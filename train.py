import random
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from data.dataset import get_dataloaders
from model.cnn import CNNModel


def set_seed(seed = 42) -> None:
    """
    Set the seed for full experiment reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_epoch(model: nn.Module,
                device: torch.device,
                criterion: nn.Module,
                train_loader: DataLoader,
                optimizer: Optimizer
                ) -> Tuple[float, float]:
    """
    Trains the model for one epoch.
    :return: Tuple (average loss, accuracy)
    """
    model.train()

    running_loss = 0.0
    correct = 0.0
    total = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Zeroing gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Statistic
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)

        correct += torch.sum(predicted == labels).item()
        total += labels.size(0)

    if total == 0:
        raise RuntimeError("Empty training set")

    return running_loss / total , correct / total

def eval_epoch(model: nn.Module,
               device: torch.device,
               criterion: nn.Module,
               val_loader: DataLoader
               ) -> Tuple[float, float]:
    """
    Model evaluation.
    :return: Tuple (average loss, accuracy)
    """
    model.eval()

    running_loss = 0.0
    correct = 0.0
    total = 0.0


    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct += torch.sum(predicted == labels).item()
            total += labels.size(0)

    if total == 0:
        raise RuntimeError("Empty validation set")

    return running_loss / total , correct / total


def plot_learning_curves(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))

    #Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Valid Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    #Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Valid Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()


def train_model() -> None:
    """
    Main training loop and progress visualization.
    """
    try:
        set_seed()
        device = torch.device('cpu')

        train_loader, val_loader, _ = get_dataloaders()
        model = CNNModel().to(device)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        epochs = 10
        best_val_acc = 0.0

        history = {
            "train_loss": [],
            "val_loss": [],
            "train_accuracy": [],
            "val_accuracy": []
        }


        for epoch in range(epochs):
            train_loss, train_acc = train_epoch(
                model, device, criterion, train_loader, optimizer)

            val_loss, val_acc = eval_epoch(
                model, device, criterion, val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_accuracy"].append(train_acc)
            history["val_accuracy"].append(val_acc)


            print(
                f"Epoch [{epoch + 1}/{epochs}]\t"
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(),
                           f'./model/best_model.pth')

        plot_learning_curves(history)
        print("Training Finished")

    except Exception as e:
        print(f"[ERROR] An error occurred during training: {e}")
        raise


if __name__ == '__main__':
    train_model()
