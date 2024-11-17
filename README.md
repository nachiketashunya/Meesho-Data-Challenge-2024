# Meesho Data Challenge Repository

## Overview

This repository contains the implementation for the Meesho Data Challenge. The codebase includes training pipelines with validation capabilities and inference scripts for model deployment.

## Collaborators

- [Alli Khadga Jyoth](https://www.linkedin.com/in/khadga-a/)
- [Kushal Agrawal](https://www.linkedin.com/in/kushal-agrawal-36a387168/)
- [Nachiketa Purohit](https://www.linkedin.com/in/nachiketa-purohit-2409551b1/)
- [Ritu Singh](https://www.linkedin.com/in/ritusingh46/)

## Repository Structure

The repository consists of three main Python scripts:

1. `train_with_val.py`: Training script with validation

   - Implements a 90:10 train-validation split
   - Provides validation metrics during training
   - Useful for model development and hyperparameter tuning
2. `train.py`: Full dataset training script

   - Trains the model on the complete dataset
   - Uses the same architecture as `train_with_val.py`
   - Recommended for final model training before deployment
3. `inference.py`: Model inference script

   - Handles prediction on test data
   - Supports batch processing
   - Generates submission-ready output

## Getting Started

## Environment Setup

- Install Python 3.10+ and set up a virtual environment
- Install PyTorch with CUDA support:
  ```bash
  pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
  ```
- Ensure you have a CUDA-capable GPU (minimum 8GB VRAM)
- Install all dependencies: `pip install -r requirements.txt`

### Configuration

Before running any scripts, configure the training parameters in `config.yaml`

### Training

#### Option 1: Training with Validation

This approach is recommended during the development phase:

```bash
python train_with_val.py
```

This will:

- Split the data into 90% training and 10% validation
- Provide validation metrics after each epoch

#### Option 2: Full Dataset Training

Use this for final model training:

```bash
python train.py
```

This will:

- Train on the complete dataset
- Save model checkpoints periodically
- Generate training metrics

### Inference

#### Load Model Checkpoint

Download the pre-trained model checkpoint from the following link:
[Checkpoint File](https://www.kaggle.com/models/naconda2611/vit-h-14-quickgelu-fulltrain-e4)

#### To run inference on test data:

```bash
python inference.py \
    --input_csv /path/to/input.csv \
    --image_dir /path/to/images \
    --model_path_convnext /path/to/convnext_model \
    --model_path_gelu /path/to/gelu_model \
    --output_csv output.csv \
    --batch_size 32 \
    --cache_dir /path/to/cache
```

#### Inference Parameters

- `input_csv`: Path to the test data CSV file
- `image_dir`: Directory containing test images
- `model_path_convnext`: Path to the trained OpenClip convnext-xxlarge model `.pth` file
- `model_path_gelu`: Path to the trained OpenClip ViT-H14-quickgelu model `.pth` file
- `output_csv`: Path for saving predictions
- `batch_size`: Number of images to process simultaneously
- `cache_dir`: Directory for storing temporary files

## Model Architecture

[Brief description of the model architecture, key features, and any modifications]

## Results

[Summary of model performance, key metrics, and any notable findings]

## License

[MIT](LICENSE)

## Contributing

Feel free to:

- Open issues for bugs or enhancement requests
- Submit pull requests with improvements
- Share your experimental results

## Acknowledgments

[Credit any third-party resources, datasets, or inspirations used in the project]
