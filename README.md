# CIFAR-10 Image Classification

A PyTorch implementation for CIFAR-10 image classification using a custom CNN architecture.

## Features

- Custom CNN architecture with batch normalization and dropout
- Data augmentation for improved training
- Comprehensive evaluation metrics (accuracy, precision, recall, F1-score)
- Model checkpointing with best accuracy tracking
- Learning rate scheduling
- GPU support

## Project Structure

```
CIFAR10/
├── data/                   # Dataset storage
├── model/                  # Saved model checkpoints
├── test_images/           # Sample test images
├── Data.py               # Data loading and preprocessing
├── model.py              # CNN model definition
├── train.py              # Training script
├── test.py               # Testing script
├── requirements.txt      # Python dependencies
└── README.md            # Project documentation
```

## Requirements

- Python 3.7+
- PyTorch 1.7+
- torchvision
- numpy
- scikit-learn
- Pillow (PIL)

## Usage

### Training

Run the training script:
```bash
python train.py
```

The script will:
- Automatically download the CIFAR-10 dataset
- Train the model for 30 epochs
- Save the best model

### Testing

Test the model on a single image:
```bash
python test.py
```

Modify the `img_path` variable in `test.py` to test different images.

## Model Architecture

The CNN model consists of:
- 3 Convolutional blocks with batch normalization and dropout
- Fully connected layers with dropout for classification
- 10 output classes for CIFAR-10 categories

### CIFAR-10 Classes
- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

### Test_images
Comes from the Internet