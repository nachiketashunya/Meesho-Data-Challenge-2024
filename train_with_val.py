import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from collections import defaultdict
import os
from tqdm.auto import tqdm
from itertools import product
import json
from datetime import datetime
import logging
from collections import defaultdict
from tqdm.auto import tqdm
from itertools import product
import sys
import torch
from torch.utils.data import random_split, DataLoader
import numpy as np
import torch.nn.functional as F
import open_clip
import wandb
from torch.optim.lr_scheduler import MultiStepLR
import yaml
import os
import json
import argparse

def load_category_mapping():
    with open("cat_attr_map.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Load the mapping when the module is imported
CATEGORY_MAPPING = load_category_mapping()

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def create_train_val_datasets(dataset, val_ratio=0.1, seed=42):
    """
    Split dataset into training and validation sets
    """
    total_size = len(dataset)
    val_size = int(total_size * val_ratio)
    train_size = total_size - val_size
    
    # Use random_split to maintain PyTorch Dataset functionality
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return train_dataset, val_dataset

def setup_logging(config):
    """Set up logging configuration"""
    # Create logs directory if it doesn't exist
    os.makedirs(config['paths']['log_dir'], exist_ok=True)
    
    # Create a timestamp for the log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(config['paths']['log_dir'], f'vith14_qgelu_trainval_{timestamp}.log')
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return logging.getLogger(__name__)

def log_hyperparameters(logger, params):
    """Log hyperparameters in a structured format"""
    logger.info("Training hyperparameters:")
    logger.info(json.dumps(params, indent=2))


def create_clip_model(model_name, pretrained_ds, device, cache_dir=None):
    model, preprocess_train, preprocess_val = open_clip.create_model_and_transforms(
        model_name,
        device=device,
        pretrained=pretrained_ds,
        precision="fp32",  # Explicitly set precision to fp32
        cache_dir=cache_dir
    )
    # Ensure model is in fp32
    model = model.float()
    return model, preprocess_train, preprocess_val

def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:
            p.grad.data = p.grad.data.float()

def validate_model(model, clip_model, val_loader, criterion, device, logger):
    """
    Validate the model on validation set
    """
    model.eval()
    clip_model.eval()
    val_loss = 0.0
    attr_correct = defaultdict(int)
    attr_total = defaultdict(int)
    num_batches = 0
    
    with torch.no_grad():
        for images, categories, targets in tqdm(val_loader, desc="Validating Model"):
            images = images.to(device)
            batch_size = images.size(0)
            
            image_features = clip_model.encode_image(images)
            
            batch_loss = 0
            valid_predictions = 0
            batch_correct = defaultdict(int)
            batch_total = defaultdict(int)
            
            for i in range(batch_size):
                category = categories[i]
                img_features = image_features[i].unsqueeze(0)

                predictions = model(img_features, category)
                
                for key, pred in predictions.items():
                    target_val = targets[key][i]
                    if target_val != -1:
                        target_tensor = torch.tensor([target_val]).to(device)
                        loss = criterion(pred, target_tensor)
                        batch_loss += loss
                        
                        _, predicted = torch.max(pred, 1)
                        is_correct = (predicted == target_tensor).item()
                        
                        attr_correct[key] += is_correct
                        attr_total[key] += 1

                        batch_correct[key] += is_correct
                        batch_total[key] += 1
                        
                        valid_predictions += 1
            
            if valid_predictions > 0:
                val_loss += (batch_loss / valid_predictions).item()

                num_batches += 1
    
    # Calculate metrics
    avg_val_loss = val_loss / num_batches
    val_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
    
    # Calculate per-attribute accuracies
    attr_accuracies = {
        key: attr_correct[key] / attr_total[key] 
        for key in attr_correct.keys()
        if attr_total[key] > 0
    }
    
    return avg_val_loss, val_acc, attr_accuracies

# Custom collate function to handle different categories and their attributes
def custom_collate_fn(batch):
    # Separate images, categories, and targets
    images = torch.stack([item[0] for item in batch])
    categories = [item[1] for item in batch]
    
    # Initialize an empty targets dict with all possible category-attribute combinations
    targets = {}
    for category, attrs in CATEGORY_MAPPING.items():
        for attr_name in attrs.keys():
            key = f"{category}_{attr_name}"
            targets[key] = torch.full((len(batch),), -1, dtype=torch.long)
    
    # Fill in the actual values
    for batch_idx, (_, category, item_targets) in enumerate(batch):
        for key, value in item_targets.items():
            if key in targets:
                targets[key][batch_idx] = value
    
    return images, categories, targets

class ProductDataset(Dataset):
    def __init__(self, csv_path, image_dir, clip_preprocess_train, clip_preprocess_val, train=True):
        self.df = pd.read_csv(csv_path)
        self.image_dir = image_dir
        self.train = train
        
        # Load CLIP model
        # self.clip_model, self.clip_preprocess = clip.load("ViT-L/14", device="cuda")
        self.clip_preprocess_train = clip_preprocess_train
        self.clip_preprocess_val = clip_preprocess_val
        
        # Store category-wise attribute information
        self.category_attributes = CATEGORY_MAPPING
        
        # Create attribute encoders and store unique values for each attribute
        self.attribute_encoders = {}
        self.attribute_classes = {}
        
        # Initialize all possible category-attribute combinations
        for category, attributes in self.category_attributes.items():
            category_data = self.df[self.df['Category'] == category]
            
            for attr_name, attr_col in attributes.items():
                # Get unique values excluding NULL/dummy values
                unique_values = category_data[attr_col][
                    (category_data[attr_col].notna()) & 
                    (category_data[attr_col] != 'dummy')
                ].unique()
                
                key = f"{category}_{attr_name}"
                self.attribute_classes[key] = list(unique_values)
                self.attribute_encoders[key] = {
                    value: idx for idx, value in enumerate(unique_values)
                }
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        category = row['Category']
        image_path = f"{self.image_dir}/{row['id'].astype(str).zfill(6)}.jpg"
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')

            if self.train:
                image = self.clip_preprocess_train(image)
            else:
                image = self.clip_preprocess_train(image)
            
            # Initialize targets with all possible category-attribute combinations
            targets = {}
            for cat, attrs in self.category_attributes.items():
                for attr_name, attr_col in attrs.items():
                    key = f"{cat}_{attr_name}"
                    # Default value is -1 (ignore index)
                    targets[key] = -1
            
            # Fill in the actual values for this category
            category_attrs = self.category_attributes[category]
            for attr_name, attr_col in category_attrs.items():
                value = row[attr_col]
                key = f"{category}_{attr_name}"
                
                if pd.isna(value) or value == 'dummy':
                    targets[key] = -1
                else:
                    targets[key] = self.attribute_encoders[key].get(value, -1)
            
            return image, category, targets
            
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return default values with all possible category-attribute combinations
            targets = {
                f"{cat}_{attr}": -1 
                for cat in self.category_attributes
                for attr in self.category_attributes[cat].keys()
            }
            return torch.zeros((3, 336, 336)), category, targets


class CategoryAwareAttributePredictor(nn.Module):
    def __init__(self, clip_dim=512, category_attributes=None, attribute_dims=None, hidden_dim=512, dropout_rate=0.2, num_hidden_layers=1):
        super(CategoryAwareAttributePredictor, self).__init__()
        
        self.category_attributes = category_attributes
        
        # Create prediction heads for each category-attribute combination
        self.attribute_predictors = nn.ModuleDict()
        
        
        for category, attributes in category_attributes.items():
            for attr_name in attributes.keys():
                key = f"{category}_{attr_name}"
                if key in attribute_dims:
                    layers = []
                    
                    # Input layer
                    layers.append(nn.Linear(clip_dim, hidden_dim))
                    layers.append(nn.LayerNorm(hidden_dim))
                    layers.append(nn.ReLU())
                    layers.append(nn.Dropout(dropout_rate))
                    
                    # Additional hidden layers
                    for _ in range(num_hidden_layers - 1):
                        layers.append(nn.Linear(hidden_dim, hidden_dim // 2))
                        layers.append(nn.LayerNorm(hidden_dim))
                        layers.append(nn.ReLU())
                        layers.append(nn.Dropout(dropout_rate))

                        hidden_dim = hidden_dim // 2
                    # Output layer
                    layers.append(nn.Linear(hidden_dim, attribute_dims[key]))
                    
                    self.attribute_predictors[key] = nn.Sequential(*layers)
    
    def forward(self, clip_features, category):
        results = {}

        category_attrs = self.category_attributes[category]
        
        clip_features = clip_features.float()
        
        for attr_name in category_attrs.keys():
            key = f"{category}_{attr_name}"
            if key in self.attribute_predictors:
                results[key] = self.attribute_predictors[key](clip_features)
        
        return results

# Option 1: Using custom gamma values for each milestone
class CustomMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, milestones, gammas, last_epoch=-1):
        self.milestones = milestones
        self.gammas = {milestone: gamma for milestone, gamma 
                      in zip(milestones, gammas)}
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.last_epoch in self.milestones:
            gamma = self.gammas[self.last_epoch]
            return [group['lr'] * gamma for group in self.optimizer.param_groups]
        return [group['lr'] for group in self.optimizer.param_groups]

def train_model_with_validation(
        clip_model, 
        model, 
        train_loader, 
        val_loader, 
        device,  
        clip_lr,
        predictor_lr, 
        weight_decay, 
        beta1, 
        beta2, 
        hidden_dim, 
        dropout_rate, 
        num_hidden_layers, 
        milestones, 
        gamma, 
        logger, 
        patience=30, 
        num_epochs=10, 
        accumulation_steps=2, 
        using_wandb=False
    ):

    """
    Modified training function with validation, early stopping, and support for multiple CLIP models
    """
    # Log training configuration
    logger.info("Starting new training run with validation")
    log_hyperparameters(logger, {
        'clip_model': "laion",
        'clip_lr': clip_lr,
        'predictor_lr': predictor_lr,
        'weight_decay': weight_decay,
        'beta1': beta1,
        'beta2': beta2,
        'num_epochs': num_epochs,
        'patience': patience,
        'device': str(device),
        'model_architecture': str(model)
    })
    
    optimizer = optim.AdamW([
        {'params': clip_model.parameters(), 'lr': clip_lr},
        {'params': model.parameters(), 'lr': predictor_lr, 'weight_decay': weight_decay}
    ], betas=(beta1, beta2))
    
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    scaler = torch.GradScaler("cuda")

    scheduler = MultiStepLR(optimizer=optimizer, milestones=milestones, gamma=gamma)
    # scheduler = CustomMultiStepLR(
    #     optimizer,
    #     milestones=[3, 4, 5, 6],
    #     gammas=[0.1, 0.5, 0.7, 0.9]  # Different decay rates for each milestone
    # )
    
    metrics_history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'params': {
            'clip_model': "laion",
            'clip_lr': clip_lr,
            'predictor_lr': predictor_lr,
            'weight_decay': weight_decay,
            'beta1': beta1,
            'beta2': beta2,
            'hidden_dim': hidden_dim,
            'dropout_rate': dropout_rate,
            'num_hidden_layers': num_hidden_layers
        }
    }
    
    epoch_pbar = tqdm(range(num_epochs), desc='Training Progress', position=0)
    
    try:
        for epoch in epoch_pbar:
            logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
            
            # Training phase
            model.train()
            clip_model.train()
            train_loss = 0.0
            attr_correct = defaultdict(int)
            attr_total = defaultdict(int)
            
            train_pbar = tqdm(train_loader, 
                            desc=f'Epoch {epoch+1}/{num_epochs} [Train]', 
                            leave=False, 
                            position=1)
            
            num_batches = 0
            batch_metrics = defaultdict(list)

            for batch_idx, (images, categories, targets) in enumerate(train_pbar):
                try:
                    # Ensure images are in fp32
                    images = images.to(device).float()
                    batch_size = images.size(0)
                    
                    with torch.autocast('cuda'): 
                        image_features = clip_model.encode_image(images)

                        batch_loss = 0
                        batch_correct = defaultdict(int)
                        batch_total = defaultdict(int)
                        valid_predictions = 0
                        
                        for i in range(batch_size):
                            category = categories[i]
                            img_features = image_features[i].unsqueeze(0)
                
                            predictions = model(img_features, category)
                            
                            for key, pred in predictions.items():
                                target_val = targets[key][i]
                                if target_val != -1:
                                    target_tensor = torch.tensor([target_val], device=device, dtype=torch.long)
                                    loss = criterion(pred, target_tensor)
                                    batch_loss += loss
                                    
                                    _, predicted = torch.max(pred, 1)
                                    is_correct = (predicted == target_tensor).item()
                                    
                                    attr_correct[key] += is_correct
                                    attr_total[key] += 1
                                    batch_correct[key] += is_correct
                                    batch_total[key] += 1
                                    valid_predictions += 1
                                                
                    if valid_predictions > 0:
                        # Normalize the loss based on valid predictions
                        batch_loss = batch_loss / valid_predictions
                        
                        # Scale the loss by the number of accumulation steps
                        batch_loss = batch_loss / accumulation_steps
                        
                        # Accumulate gradients
                        scaler.scale(batch_loss).backward()
                        
                        # Only update weights after accumulating gradients for specified steps
                        if (batch_idx + 1) % accumulation_steps == 0:
                            # Optimizer step and zero grad only after accumulation is complete
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()

                            # Track metrics
                            train_loss += batch_loss.item() * accumulation_steps  # Multiply by accumulation_steps to get true loss
                            num_batches += 1
                            
                            # Log batch metrics
                            batch_metrics['loss'].append(batch_loss.item() * accumulation_steps)
                            batch_acc = sum(batch_correct.values()) / max(sum(batch_total.values()), 1)
                            batch_metrics['accuracy'].append(batch_acc)
                            
                            if (batch_idx+1) % 100 == 0:
                                logger.info(
                                    f"Epoch {epoch+1}, Batch {batch_idx}: "
                                    f"Loss = {batch_loss.item() * accumulation_steps:.4f}, "
                                    f"Accuracy = {batch_acc:.4%}"
                                )
                        else:
                            # For intermediate accumulation steps, just add to metrics without optimizer step
                            train_loss += batch_loss.item() * accumulation_steps
                            num_batches += 1
                
                except Exception as e:
                    logger.error(f"Error in batch {batch_idx}: {str(e)}", exc_info=True)
                    continue

            # Validation phase
            val_loss, val_acc, val_attr_accuracies = validate_model(
                model, clip_model, val_loader, criterion, device, logger
            )

            scheduler.step()
            
            # Calculate and log epoch metrics
            avg_train_loss = train_loss / max(num_batches, 1)
            train_acc = sum(attr_correct.values()) / max(sum(attr_total.values()), 1)
            
            metrics_history['train_loss'].append(avg_train_loss)
            metrics_history['train_acc'].append(train_acc)
            metrics_history['val_loss'].append(val_loss)
            metrics_history['val_acc'].append(val_acc)

            lrs = [param_group['lr'] for param_group in optimizer.param_groups]

            if using_wandb:
                wandb.log({
                    'LR/CLIP': lrs[0],
                    'LR/Model': lrs[1],
                    'Train/Loss': avg_train_loss,
                    'Train/Acc': train_acc,
                    'Val/Loss': val_loss,
                    'Val/Acc': val_acc
                })
                
            # Log per-attribute accuracies
            attr_accuracies = {
                key: attr_correct[key] / attr_total[key] 
                for key in attr_correct.keys()
            }

            logger.info(
                f"Epoch {epoch+1} Results:\n"
                f"Train Loss: {avg_train_loss:.4f}\n"
                f"Train Acc: {train_acc:.4%}\n"
                f"Attribute Training Accuracies: {json.dumps(attr_accuracies, indent=2)}\n"
                f"Val Loss: {val_loss:.4f}\n"
                f"Val Acc: {val_acc:.4%}\n"
                f"Attribute Validation Accuracies: {json.dumps(val_attr_accuracies, indent=2)}"
            )
    
    except Exception as e:
        logger.error(f"Training interrupted: {str(e)}", exc_info=True)
        raise
    
    logger.info("Training completed successfully")
    return model, clip_model, metrics_history

def main():
    # Load configuration
    parser = argparse.ArgumentParser(description='Train and Validate')
    
    parser.add_argument('--config_path', 
                       type=str,
                       help='Path to config yaml file')

    args = parser.parse_args()

    config_path = args.config_path
    config = load_config(config_path)

    torch.cuda.empty_cache()
    # Set up logging
    logger = setup_logging(config)
    logger.info("Starting hyperparameter search")

    # Log into wandb
    using_wandb = config['wandb']['using']
    if using_wandb:
        wandb.login(key=config['wandb']['api_key'])
        logger.info("WandB login successful!")
    
    # Get hyperparameter grid from config
    param_grid = {
        'clip_lr': config['optimizer']['clip_lr'],
        'predictor_lr': config['optimizer']['predictor_lr'],
        'weight_decay': config['optimizer']['weight_decay'],
        'beta1': config['optimizer']['beta1'],
        'beta2': config['optimizer']['beta2'],
        'hidden_dim': config['model']['hidden_dim'],
        'dropout_rate': config['model']['dropout_rate'],
        'num_hidden_layers': config['model']['num_hidden_layers']
    }
    
    logger.info("Hyperparameter search space:")
    logger.info(json.dumps(param_grid, indent=2))

    # Training Related Configurations
    batch_size = config['training']['batch_size']
    num_epochs = config['training']['num_epochs']
    accumulation_steps = config['training']['accumulation_steps']
    gamma = config['scheduler']['gamma']
    milestones = config['scheduler']['milestones']

    # CLIP model related configurations
    clip_dim = config['model']['clip_dim']
    model_name = config['model']['name']
    pretrained_ds = config['model']['pretrained']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    clip_model, clip_preprocess_train, clip_preprocess_val = create_clip_model(
        model_name=model_name,
        pretrained_ds=pretrained_ds,
        device=device,
        cache_dir=config['paths']['cache_dir']
    )

    logger.info("LAION CLIP model loaded successfully in fp32 precision")
    
    try:
        # Data loading setup
        # Data loading setup
        data_dir = config['paths']['data_dir']
        train_csv = os.path.join(data_dir, config['paths']['train_csv'])
        train_images = os.path.join(data_dir, config['paths']['train_images'])

        train_dataset = ProductDataset(
            csv_path=train_csv,
            image_dir=train_images,
            clip_preprocess_train=clip_preprocess_train,
            clip_preprocess_val=clip_preprocess_val,
            train=True
        )
        logger.info(f"Dataset loaded successfully with {len(train_dataset)} samples")

        # Split into train and validation sets
        train_subset, val_subset = create_train_val_datasets(train_dataset)
        logger.info(f"Dataset split: {len(train_subset)} train, {len(val_subset)} validation samples")
        
        # Create data loaders
        train_loader = DataLoader(
            train_subset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Prefetch batches
            collate_fn=custom_collate_fn
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            persistent_workers=True,  # Keep workers alive between epochs
            prefetch_factor=2,  # Prefetch batches
            collate_fn=custom_collate_fn
        )
        
        attribute_dims = {
            key: len(values) 
            for key, values in train_dataset.attribute_classes.items()
        }
        
        logger.info("Attribute dimensions:")
        logger.info(json.dumps(attribute_dims, indent=2))
        
        all_results = []
        
        # Systematic hyperparameter search
        for hidden_dim, dropout_rate, num_hidden_layers in product(
            param_grid['hidden_dim'],
            param_grid['dropout_rate'],
            param_grid['num_hidden_layers']
        ):
            model = CategoryAwareAttributePredictor(
                clip_dim=clip_dim,
                category_attributes=CATEGORY_MAPPING,
                attribute_dims=attribute_dims,
                hidden_dim=hidden_dim,
                dropout_rate=dropout_rate,
                num_hidden_layers=num_hidden_layers
            ).to(device)
            
            logger.info(f"\nInitialized model with architecture parameters:")
            logger.info(f"Hidden Dim: {hidden_dim}, Dropout: {dropout_rate}, Layers: {num_hidden_layers}")
            
            for clip_lr, predictor_lr, weight_decay, beta1, beta2 in product(
                param_grid['clip_lr'],
                param_grid['predictor_lr'],
                param_grid['weight_decay'],
                param_grid['beta1'],
                param_grid['beta2']
            ):
                try:
                    logger.info(f"\nStarting training with optimizer parameters:")
                    logger.info(
                        f"CLIP LR: {clip_lr}, Predictor LR: {predictor_lr}, "
                        f"Weight Decay: {weight_decay}, Beta1: {beta1}, Beta2: {beta2}"
                    )

                    if using_wandb:
                        config = {
                            "hidden_dim": hidden_dim,
                            "num_hidden_layers": num_hidden_layers,
                            "clip_lr": clip_lr,
                            "predictor_lr": predictor_lr,
                            "weight_decay": weight_decay,
                            "beta1": beta1,
                            "beta2": beta2
                        }

                        run_name = config['wandb']['run_name_template'].format(
                            hidden_dim=hidden_dim,
                            num_hidden_layers=num_hidden_layers
                        )
                        
                        wandb.init(
                            project=config['wandb']['project'],
                            name=run_name,
                            config=config
                        )

                    # Enable torch.compile if using PyTorch 2.0+
                    if hasattr(torch, 'compile'):
                        model = torch.compile(model)
                        clip_model = torch.compile(clip_model)
                        
                    _, _, metrics = train_model_with_validation(
                        clip_model=clip_model,
                        model=model,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        clip_lr=clip_lr,
                        predictor_lr=predictor_lr,
                        weight_decay=weight_decay,
                        beta1=beta1,
                        beta2=beta2,
                        hidden_dim=hidden_dim,
                        dropout_rate=dropout_rate,
                        num_hidden_layers=num_hidden_layers,
                        gamma=gamma,
                        milestones=milestones,
                        num_epochs=num_epochs,
                        logger=logger,
                        accumulation_steps=accumulation_steps,
                        using_wandb=using_wandb
                    )
                    
                    result = {
                        'hidden_dim': hidden_dim,
                        'dropout_rate': dropout_rate,
                        'num_hidden_layers': num_hidden_layers,
                        'clip_lr': clip_lr,
                        'predictor_lr': predictor_lr,
                        'weight_decay': weight_decay,
                        'beta1': beta1,
                        'beta2': beta2,
                        'final_accuracy': metrics['train_acc'][-1],
                        'best_accuracy': max(metrics['train_acc']),
                        'final_loss': metrics['train_loss'][-1],
                        'best_loss': min(metrics['train_loss'])
                    }
                    
                    all_results.append(result)
                    logger.info("Training run completed successfully")
                    logger.info(f"Results: {json.dumps(result, indent=2)}")
                    
                    # Save results after each run
                    with open('hyperparameter_search_results.json', 'w') as f:
                        json.dump(all_results, f, indent=2)
                    logger.info("Updated hyperparameter search results saved to file")

                    if using_wandb:
                        wandb.log(result)
                        wandb.finish()
                    
                except Exception as e:
                    logger.error(f"Error in training run: {str(e)}", exc_info=True)
                    continue
        
        logger.info("Hyperparameter search completed successfully")
        
    except Exception as e:
        logger.error(f"Fatal error in main execution: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()