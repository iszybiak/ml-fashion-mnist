import os

import torch
import torch.nn as nn

from data.dataset import get_dataloaders
from model.cnn import CNNModel

def evaluate_model(model_path: str = "best_model.pth") -> None:
    """
    Evaluates a trained CNNModel on the test dataset.
    :param model_path:
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

        # Test loop
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                test_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs.data, 1)
                correct += torch.sum(predicted == labels).item()
                total += labels.size(0)

        if total == 0:
            raise ValueError("Test dataset is empty!")

        accuracy = correct / total
        avg_loss = test_loss / total

        print(f"Test accuracy: {accuracy}")
        print(f"Test loss: {avg_loss}")

    except FileNotFoundError as fnt_error:
        print(f"Error: {fnt_error}")
    except ValueError as val_error:
        print(f"Error: {val_error}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    evaluate_model()
