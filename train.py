import random
import numpy as np
import torch

from data.dataset import get_dataloaders
from model.cnn import CNNModel


def set_seed(seed = 42):
    """
    Set the seed for full experiment reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_epoch(model, device, criterion, train_loader, optimizer):
    """
    Trains the model for one epoch.
    Returns the average loss and accuracy.
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

    return running_loss / total , correct / total

def eval_epoch(model, device, criterion, val_loader):
    """
    Model evaluation.
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

    return running_loss / total , correct / total

def main():
    set_seed()
    device = torch.device('cpu')
    train_loader, val_loader, _ = get_dataloaders()
    model = CNNModel().to(device)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    best_val_acc = 0.0

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, device, criterion, train_loader, optimizer)

        val_loss, val_acc = eval_epoch(
            model, device, criterion, val_loader)

        print(
            f"Epoch [{epoch + 1}/{epochs}]\t"
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(),
                       f'./model/best_model.pth')

    print("Training Finished")


if __name__ == '__main__':
    main()
