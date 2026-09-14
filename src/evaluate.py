"""
Test the trained tomato CNN.

This script:
1. Loads the test tomato images.
2. Loads the trained CNN.
3. Lets the CNN predict each image.
4. Checks how many predictions are correct.
5. Prints the final test accuracy.
"""

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model import TomatoCNN


# Resize the test images
# and change them into numbers that PyTorch can use
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


# Load the test images
test_dataset = datasets.ImageFolder(
    "data/processed/test",
    transform=transform
)


# Give the test images to the CNN 32 at a time
test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)


# Create the same CNN structure
model = TomatoCNN()


# Load what the CNN learned during training
model.load_state_dict(
    torch.load("tomato_cnn.pth")
)


# Tell PyTorch that we are testing, not training
model.eval()


# Count the results
number_correct = 0
number_total = 0


# We do not need to train or update the CNN here
with torch.no_grad():

    for images, correct_labels in test_loader:

        # Ask the CNN to predict the tomato classes
        predictions = model(images)

        # Choose the class with the highest score
        predicted_labels = torch.argmax(
            predictions,
            dim=1
        )

        # Add the number of test images
        number_total = number_total + len(correct_labels)

        # Count how many predictions were correct
        number_correct = number_correct + (
            predicted_labels == correct_labels
        ).sum().item()


# Calculate accuracy
accuracy = number_correct / number_total


print("Number of test images:", number_total)
print("Correct predictions:", number_correct)
print("Test accuracy:", round(accuracy * 100, 2), "%")
