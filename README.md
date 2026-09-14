# Tomato Ripeness Classifier 🍅

This is a small computer vision and deep learning project.

The goal is to classify tomatoes into three ripeness stages:

* Green
* Turning
* Ripe

I made this project to practice the basic steps of image classification using a Convolutional Neural Network (CNN) with PyTorch.

## Dataset

I used the Laboro Tomato dataset. (Find the link under "Reference")

The original images contain multiple tomatoes.

The dataset also includes annotations that show:

* where each tomato is in the image
* the ripeness class of each tomato

I used these annotations to crop each tomato from the original images.

The original class names were changed to simpler names:

```text
b_green -> green
b_half_ripened -> turning
b_fully_ripened -> ripe
```

The processed training dataset contains:

```text
Green: 1511
Turning: 506
Ripe: 343
```

Total: 2360 images

I randomly split these images into:

```text
1888 images for training
472 images for validation
```

The test dataset contains 550 images.

## Image Preparation

Before giving the images to the CNN, I:

1. Resized each image to 128 x 128
2. Converted the images into PyTorch tensors
3. Loaded the images in batches of 32

## CNN Model

I created a simple CNN with PyTorch.

The model follows this basic flow:

```text
Tomato image
    ↓
Convolution
    ↓
ReLU
    ↓
Max Pooling
    ↓
Convolution
    ↓
ReLU
    ↓
Max Pooling
    ↓
Fully Connected Layers
    ↓
Green / Ripe / Turning
```

The convolution layers help the model find features in the tomato images.

The final layers use those features to choose one of the three classes.

## Training

I trained the model for 5 epochs.

I used:

* Cross Entropy Loss
* Adam optimizer
* Learning rate: 0.001
* Batch size: 32

The results were:

| Epoch |   Loss | Validation Accuracy |
| ----- | -----: | ------------------: |
| 1     | 0.6669 |              77.33% |
| 2     | 0.4902 |              79.24% |
| 3     | 0.4364 |              83.90% |
| 4     | 0.4171 |              85.38% |
| 5     | 0.3910 |              85.81% |

## Test Result

After training, I tested the model with 550 test images.

```text
Correct predictions: 483 / 550
Test accuracy: 87.82%
```

## Technologies

* Python
* PyTorch
* torchvision
* Pillow

## Files

```text
tomato-ripeness-classifier/
├── data/                  # not included in GitHub
│   ├── laboro_big/
│   └── processed/
├── src/
│   ├── crop_tomatoes.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── .gitignore
└── README.md
```

## What I Learned

Through this project, I practiced:

* preparing image data
* using annotations and bounding boxes
* cropping images
* using PyTorch datasets and DataLoaders
* creating a simple CNN
* training a CNN
* checking validation accuracy
* testing a trained model

## Future Work

In the future, I would like to try:

* creating a confusion matrix
* testing the model with my own tomato photos
* using data augmentation
* running a smaller model on Arduino or another edge device

## Reference

- [Laboro Tomato Dataset](https://github.com/laboroai/LaboroTomato)
- [IBM - What are Convolutional Neural Networks?](https://www.ibm.com/think/topics/convolutional-neural-networks)
- [TensorFlow - Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)
