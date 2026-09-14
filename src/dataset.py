"""
This script loads the processed tomato images for training.

The images are stored in three folders:
- green
- turning
- ripe

PyTorch automatically uses these folder names as class labels.
"""

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# Load training images
dataset = datasets.ImageFolder(
    "data/processed/train",
    transform=transform
)


# Split training data into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size]
)


# Create DataLoaders
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False
)


# Print information
print("Classes:", dataset.classes)
print("Total images:", len(dataset))
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# Check one batch
images, labels = next(iter(train_loader))

print("Image batch shape:", images.shape)
print("Label batch shape:", labels.shape)
