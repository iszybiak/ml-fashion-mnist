import os

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

from data.dataset import get_dataloaders
from model.cnn import CNNModel

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

def plot_confusion_matrix(y_true, y_pred):
    """
    Plots the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues", xticks_rotation=45)
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.show()

def show_misclassified_samples(images, true_labels, pred_labels,
                               max_samples : int = 8) -> None:
    """
    Displays misclassified images from the test set.
    """
    plt.figure(figsize=(12, 4))
    count = 0

    for i in range(len(images)):
        if true_labels[i] == pred_labels[i]:
            plt.subplot(2, max_samples // 2, count + 1)
            plt.imshow(images[i].squeeze(), cmap="gray")
            plt.title(
                f"True: {CLASS_NAMES[true_labels[i]]}\n"
                f"False: {CLASS_NAMES[pred_labels[i]]}"
            )
            plt.axis('off')
            count += 1
            if count >= max_samples:
                break

    plt.suptitle("Misclassified samples")
    plt.tight_layout()
    plt.show()


def evaluate_model(model_path: str = "./model/best_model.pth") -> None:
    """
    Evaluates a trained CNNModel on the test dataset.
    """
    try:
        # Davice
        device = torch.device('cpu')

        #DataLoader
        _, _, test_loader = get_dataloaders()
        if len(test_loader) == 0:
            raise ValueError("Test loader is empty!")

        # Initialize the model
        model = CNNModel().to(device)
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        # Load model weights
        model.load_state_dict(torch.load(model_path))
        model.eval()

        criterion = nn.CrossEntropyLoss()

        correct = 0
        total = 0
        test_loss = 0

        all_predictions = []
        all_labels = []
        all_images = []

        # Test loop
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)

                # For plotting
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_images.extend(images.cpu().numpy())

                correct += torch.sum(predicted == labels).item()
                total += labels.size(0)

        if total == 0:
            raise ValueError("Test dataset is empty!")

        accuracy = correct / total
        avg_loss = test_loss / total

        print(f"Test accuracy: {accuracy}")
        print(f"Test loss: {avg_loss}")

        print('\nClassification Report')

        report = classification_report(
            all_labels,
            all_predictions,
            target_names=CLASS_NAMES,
            digits=4
        )
        print(report)

        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_predictions)

        # Show misclassified samples
        show_misclassified_samples(
            np.array(all_images),
            np.array(all_labels),
            np.array(all_predictions)
        )

    except FileNotFoundError as fnt_error:
        print(f"Error: {fnt_error}")
    except ValueError as val_error:
        print(f"Error: {val_error}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    evaluate_model()
