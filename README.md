
# MLP from Scratch

This project implements a **Multilayer Perceptron (MLP)** from scratch using **NumPy**. It is trained on the **CIFAR-10** dataset and demonstrates how to build, train, and evaluate a basic MLP model without using deep learning libraries like TensorFlow or PyTorch.

## Features
- Fully connected MLP built with customizable hidden layers.
- Forward and backward propagation implemented manually.
- **ReLU** and **Softmax** activation functions.
- Batch gradient descent with support for multiple epochs.
- Accuracy and training time visualization.
- CIFAR-10 dataset image classification.

## Dataset
The [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset consists of 60,000 32x32 color images in 10 classes:
- Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck.

The dataset is split into 50,000 training images and 10,000 test images.

## How It Works
1. **Data Preprocessing**:
   - CIFAR-10 data is loaded and normalized.
2. **Network Architecture**:
   - The MLP is created with an input layer, customizable hidden layers, and an output layer.
3. **Forward Pass**:
   - Computes the outputs for each layer.
4. **Backward Pass**:
   - Gradients are calculated using backpropagation to update the weights.
5. **Training**:
   - The model is trained on batches, with accuracy and loss tracked per epoch.

## Results
- **Accuracy**: Both training and test accuracy are logged for each epoch and plotted.
- **Training Time**: Each epoch's duration is recorded and displayed in a graph.
- **Sample Predictions**: The model's predictions on sample images from the CIFAR-10 dataset are shown with their actual and predicted labels.

### Accuracy and Time Plots
- A plot of accuracy over epochs.
- A plot showing the time taken for each epoch.


## Customization
You can customize the following parameters:
- **Number of hidden layers**: Adjust the number of layers and their sizes.
- **Activation functions**: Modify the ReLU and Softmax functions.
- **Batch size** and **Epochs**: Set these according to your hardware capabilities.

## Example
```python
# Example of how to create an MLP with 2 hidden layers
layers = [3072, 128, 64, 10]  # Input layer, 2 hidden layers, output layer
```

