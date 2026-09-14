"""
Train the tomato ripeness CNN.

The model learns from tomato images and their correct labels.

Steps:
1. Give images to the model.
2. Get predictions.
3. Compare predictions with the correct answers.
4. Calculate the error.
5. Update the model.
6. Repeat.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from dataset import train_loader, val_loader
from model import TomatoCNN


# Create the CNN
model = TomatoCNN()


# Measure how wrong the prediction is
loss_function = nn.CrossEntropyLoss()


# Update the model when it makes mistakes
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)


# Train using the whole training dataset 5 times
number_of_epochs = 5


for epoch in range(number_of_epochs):

    print("Starting epoch", epoch + 1)

    # --------------------
    # Training
    # --------------------

    model.train()

    total_loss = 0

    for images, correct_labels in train_loader:

        # Remove information from the previous batch
        optimizer.zero_grad()

        # Give the images to the CNN
        predictions = model(images)

        # Check how different the predictions are from the correct answers
        loss = loss_function(
            predictions,
            correct_labels
        )

        # Calculate how the model should change
        loss.backward()

        # Update the model
        optimizer.step()

        total_loss = total_loss + loss.item()


    # Calculate average training loss
    average_loss = total_loss / len(train_loader)


    # --------------------
    # Validation
    # --------------------

    model.eval()

    correct_predictions = 0
    total_images = 0

    with torch.no_grad():

        for images, correct_labels in val_loader:

            # Make predictions
            predictions = model(images)

            # Choose the class with the highest score
            predicted_labels = torch.argmax(
                predictions,
                dim=1
            )

            # Count number of images
            total_images = total_images + correct_labels.size(0)

            # Count correct predictions
            correct_predictions = correct_predictions + (
                predicted_labels == correct_labels
            ).sum().item()


    validation_accuracy = (
        correct_predictions / total_images
    )


    print("Loss:", round(average_loss, 4))

    print(
        "Validation accuracy:",
        round(validation_accuracy * 100, 2),
        "%"
    )


# Save the trained model
torch.save(
    model.state_dict(),
    "tomato_cnn.pth"
)

print("Training finished")
print("Model saved")
