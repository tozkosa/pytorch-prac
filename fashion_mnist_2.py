"""From official tutorial"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import fashion_mnist

if __name__ == "__main__":
    model = fashion_mnist.NeuralNetwork()

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot"
    ]

    test_data = datasets.FashionMNIST(
        root="../data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    print(f"shape of x: {x.shape}")
    with torch.no_grad():
        pred = model(x)
        print(f"shape of pred: {pred.shape}")
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f"Predicted: {predicted}, Actual: {actual}")
    



