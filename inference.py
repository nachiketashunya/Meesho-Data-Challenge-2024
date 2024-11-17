import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
import os
from tqdm import tqdm
import open_clip
from pathlib import Path
import json
import argparse
import torch
import time


def load_category_mapping():
    with open("cat_attr_map.json", "r", encoding="utf-8") as f:
        return json.load(f)

# Load the mapping when the module is imported
CATEGORY_MAPPING = load_category_mapping()

class ImageDataset(Dataset):
    """Dataset class for batch processing of images"""

    def __init__(self, image_paths, categories, clip_preprocess):
        self.image_paths = image_paths
        self.categories = categories
        self.clip_preprocess = clip_preprocess

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.clip_preprocess(image)
        except Exception as e:
            print(f"Error loading image {image_path}: {str(e)}")
            # Return a blank image in case of error
            image = torch.zeros(3, 224, 224)
        return image, self.categories[idx]


class CategoryAwareAttributePredictor(nn.Module):
    def __init__(
        self,
        clip_dim=512,
        category_attributes=None,
        attribute_dims=None,
        hidden_dim=512,
        dropout_rate=0.2,
        num_hidden_layers=1,
    ):
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


def predict_batch(
    images,
    categories,
    clip_model_gelu,
    clip_model_convnext,
    model_gelu,
    model_convnext,
    checkpoint_gelu,
    checkpoint_convnext,
    clip_preprocess_gelu,
    clip_preprocess_convnext,
    device="cuda",
    batch_size=32,
):
    """Process a batch of images using ensemble of two models"""
    all_predictions = []

    dataset = ImageDataset(images, categories, clip_preprocess_gelu)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,  # Prefetch 2 batches per worker
    )

    total_batches = len(dataloader)
    with torch.no_grad():
        pbar = tqdm(
            dataloader,
            total=total_batches,
            desc="Processing batches",
            unit="batch",
            position=0,
            leave=True,
        )

        for batch_images, batch_categories in pbar:
            batch_images_gelu = batch_images.to(device, non_blocking=True)
            batch_images_convnext = batch_images.to(device, non_blocking=True)

            start_time = time.time()

            # Get CLIP features for the batch
            with torch.autocast(str(device)):
                clip_features_gelu = clip_model_gelu.encode_image(batch_images_gelu)
                clip_features_convnext = clip_model_convnext.encode_image(
                    batch_images_convnext
                )

                # Process each category in the batch
                batch_predictions = []
                for idx, category in enumerate(batch_categories):
                    if category not in checkpoint_gelu["category_mapping"]:
                        batch_predictions.append({})
                        continue

                    # Get model predictions for single image
                    predictions_gelu = model_gelu(
                        clip_features_gelu[idx : idx + 1], category
                    )
                    predictions_convnext = model_convnext(
                        clip_features_convnext[idx : idx + 1], category
                    )

                    # Ensemble the predictions
                    ensemble_predictions = {}
                    for key, pred_gelu in predictions_gelu.items():
                        pred_convnext = predictions_convnext[key].to(device)
                        ensemble_predictions[key] = (
                            0.5 * pred_gelu + 0.5 * pred_convnext
                        )

                    # Convert predictions to attribute values
                    predicted_attributes = {}
                    for key, pred in ensemble_predictions.items():
                        _, predicted_idx = torch.max(pred, 1)
                        predicted_idx = predicted_idx.item()

                        attr_name = key.split("_", 1)[1]
                        attr_values = checkpoint_gelu["attribute_classes"][key]
                        if predicted_idx < len(attr_values):
                            predicted_attributes[attr_name] = attr_values[predicted_idx]

                    batch_predictions.append(predicted_attributes)

            end_time = time.time()
            batch_time = end_time - start_time
            pbar.set_postfix({"Per Image Time": f"{batch_time/batch_size:.2f}s"})

            all_predictions.extend(batch_predictions)

            # Clear GPU cache periodically
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return all_predictions


def create_clip_model_convnext(device, cache_dir=None):
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        "convnext_xxlarge",
        device=device,
        pretrained="laion2b_s34b_b82k_augreg_soup",
        precision="fp32",
        cache_dir=cache_dir,
    )
    model = model.float()
    return model, preprocess_train


def create_clip_model_gelu(device, cache_dir=None):
    model, preprocess_train, _ = open_clip.create_model_and_transforms(
        "ViT-H-14-quickgelu",
        device=device,
        pretrained="dfn5b",
        precision="fp32",  # Explicitly set precision to fp32
        cache_dir=cache_dir,
    )
    model = model.float()
    return model, preprocess_train


# Option 1: Clean the checkpoint before loading
def clean_state_dict(state_dict):
    new_state_dict = {}
    for k, v in state_dict.items():
        # Remove '_orig_mod.' prefix
        name = k.replace("_orig_mod.", "")
        new_state_dict[name] = v
    return new_state_dict


def load_models(model_path_gelu, model_path_convnext, device, cache_dir=None):
    # Load the CLIP model gelu
    checkpoint_gelu = torch.load(model_path_gelu, map_location="cpu",weights_only = False)
    clean_clip_checkpoint_gelu = clean_state_dict(
        checkpoint_gelu["clip_model_state_dict"]
    )

    clip_model_gelu, clip_preprocess_gelu = create_clip_model_gelu("cpu", cache_dir)
    clip_model_gelu.load_state_dict(clean_clip_checkpoint_gelu)
    clip_model_gelu = clip_model_gelu.to(device)
    del clean_clip_checkpoint_gelu
    torch.cuda.empty_cache()

    # Load the CLIP model convnext
    checkpoint_convnext = torch.load(model_path_convnext, map_location="cpu",weights_only = False)
    clean_clip_checkpoint_convnext = clean_state_dict(
        checkpoint_convnext["clip_model_state_dict"]
    )

    clip_model_convnext, clip_preprocess_convnext = create_clip_model_convnext(
        "cpu", cache_dir
    )
    clip_model_convnext.load_state_dict(clean_clip_checkpoint_convnext)
    clip_model_convnext = clip_model_convnext.to(device)
    del clean_clip_checkpoint_convnext
    torch.cuda.empty_cache()

    # Load the attribute predictor models
    model_gelu = CategoryAwareAttributePredictor(
        clip_dim=checkpoint_gelu["model_config"]["clip_dim"],
        category_attributes=checkpoint_gelu["dataset_info"]["category_mapping"],
        attribute_dims={
            key: len(values)
            for key, values in checkpoint_gelu["dataset_info"][
                "attribute_classes"
            ].items()
        },
        hidden_dim=checkpoint_gelu["model_config"]["hidden_dim"],
        dropout_rate=checkpoint_gelu["model_config"]["dropout_rate"],
        num_hidden_layers=checkpoint_gelu["model_config"]["num_hidden_layers"],
    ).to(device)

    model_convnext = CategoryAwareAttributePredictor(
        clip_dim=checkpoint_convnext["model_config"]["clip_dim"],
        category_attributes=checkpoint_convnext["dataset_info"]["category_mapping"],
        attribute_dims={
            key: len(values)
            for key, values in checkpoint_convnext["dataset_info"][
                "attribute_classes"
            ].items()
        },
        hidden_dim=checkpoint_convnext["model_config"]["hidden_dim"],
        dropout_rate=checkpoint_convnext["model_config"]["dropout_rate"],
        num_hidden_layers=checkpoint_convnext["model_config"]["num_hidden_layers"],
    ).to(device)

    clean_cat_checkpoint_gelu = clean_state_dict(checkpoint_gelu["model_state_dict"])
    model_gelu.load_state_dict(clean_cat_checkpoint_gelu)
    del clean_cat_checkpoint_gelu

    clean_cat_checkpoint_convnext = clean_state_dict(
        checkpoint_convnext["model_state_dict"]
    )
    model_convnext.load_state_dict(clean_cat_checkpoint_convnext)
    del clean_cat_checkpoint_convnext

    if hasattr(torch, "compile"):
        model_gelu = torch.compile(model_gelu)
        clip_model_gelu = torch.compile(clip_model_gelu)
        model_convnext = torch.compile(model_convnext)
        clip_model_convnext = torch.compile(clip_model_convnext)

    model_gelu.eval()
    clip_model_gelu.eval()
    model_convnext.eval()
    clip_model_convnext.eval()

    return (
        model_gelu,
        clip_model_gelu,
        clip_preprocess_gelu,
        checkpoint_gelu["dataset_info"],
        model_convnext,
        clip_model_convnext,
        clip_preprocess_convnext,
        checkpoint_convnext["dataset_info"],
    )


def process_csv_file(
    input_csv_path,
    image_dir,
    model_path_gelu,
    model_path_convnext,
    output_csv_path,
    batch_size=32,
    device="cuda",
    cache_dir=None,
):
    # Load the input CSV
    df = pd.read_csv(input_csv_path)

    # Validate required columns
    required_columns = ["id", "Category"]
    if not all(col in df.columns for col in required_columns):
        raise ValueError(f"Input CSV must contain columns: {required_columns}")

    # Load models from checkpoint
    (
        model_gelu,
        clip_model_gelu,
        clip_preprocess_gelu,
        checkpoint_gelu,
        model_convnext,
        clip_model_convnext,
        clip_preprocess_convnext,
        checkpoint_convnext,
    ) = load_models(model_path_gelu, model_path_convnext, device, cache_dir)
    print("Loaded both the models Successfully")

    # Prepare image paths and categories
    image_paths = [
        os.path.join(image_dir, f"{str(id_).zfill(6)}.jpg")
        for id_ in df["id"]
        if os.path.exists(os.path.join(image_dir, f"{str(id_).zfill(6)}.jpg"))
    ]
    valid_indices = [
        i
        for i, id_ in enumerate(df["id"])
        if os.path.exists(os.path.join(image_dir, f"{str(id_).zfill(6)}.jpg"))
    ]
    categories = df["Category"].iloc[valid_indices].tolist()

    print(f"Processing {len(image_paths)} valid images out of {len(df)} total entries")
    print(f"Batch size: {batch_size}")

    # Get predictions in batches

    predictions = predict_batch(
        image_paths,
        categories,
        clip_model_gelu,
        clip_model_convnext,
        model_gelu,
        model_convnext,
        checkpoint_gelu,
        checkpoint_convnext,
        clip_preprocess_gelu,
        clip_preprocess_convnext,
        device=device,
        batch_size=batch_size,
    )

    # Process results
    results = []
    pred_idx = 0
    for idx, row in df.iterrows():
        if idx in valid_indices:
            pred = predictions[pred_idx]
            pred_idx += 1
        else:
            pred = {}

        result = {"id": row["id"], "Category": row["Category"], "len": len(pred)}

        # Map the predictions to attr_1, attr_2, etc.
        category_mapping = CATEGORY_MAPPING[row["Category"]]

        # Initialize all attribute columns with None
        for i in range(1, 11):
            result[f"attr_{i}"] = "dummy"

        # Fill in the predicted attributes according to the mapping
        for attr_name, pred_value in pred.items():
            if attr_name in category_mapping:
                attr_column = category_mapping[attr_name]
                result[attr_column] = pred_value

        results.append(result)

    # Create output DataFrame
    output_df = pd.DataFrame(results)
    columns = ["id", "Category", "len"] + [f"attr_{i}" for i in range(1, 11)]
    output_df = output_df[columns]

    # Save to CSV
    output_df.to_csv(output_csv_path, index=False)
    print(f"Results saved to {output_csv_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with CLIP model")

    parser.add_argument("--input_csv", type=str, help="Path to input CSV file")

    parser.add_argument("--image_dir", type=str, help="Directory containing images")

    parser.add_argument(
        "--model_path_convnext", type=str, help="Path to CONVNEXT model checkpoint"
    )

    parser.add_argument(
        "--model_path_gelu", type=str, help="Path to GELU model checkpoint"
    )

    parser.add_argument(
        "--output_csv",
        type=str,
        default="submission.csv",
        help="Path for output CSV file",
    )

    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for processing"
    )

    parser.add_argument(
        "--cache_dir", type=str, default=None, help="Cache directory path"
    )

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Process the CSV files
    process_csv_file(
        input_csv_path=args.input_csv,
        model_path_convnext=args.model_path_convnext,
        model_path_gelu=args.model_path_gelu,
        image_dir=args.image_dir,
        output_csv_path=args.output_csv,
        batch_size=args.batch_size,
        device=device,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
