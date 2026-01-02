# Script for loading a trained CNN model and running inference
# on a single image. The image should be converted to grayscale,
# resized to 28x28 and normalized the same way as during training.

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image, ImageOps
from model.cnn import CNNModel
import matplotlib.pyplot as plt
import argparse


# Normalization parameters (same as during training)
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
def load_model(model_path: str, device: torch.device) -> nn.Module:
    """
    Loads the trained CNN model from the specified path.
    """
    model = CNNModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(image_path: str) -> torch.Tensor:
    """
    Preprocesses the input image for model inference.
    The displayed image and the pre-tensor image are identical.
    """

    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
    ])

    image = Image.open(image_path).convert("L")  # L = grayscale
    image = ImageOps.invert(image)
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension
    return image
    # img = Image.open(image_path).convert("L")  # L = grayscale


def predict_image(model: nn.Module, image_tensor: torch.Tensor,
                    device: torch.device) -> tuple[int, float]:
        """
        Predicts the class of the input image tensor using the model.
        """
        image_tensor = image_tensor.to(device)
        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probs, 1)
        return predicted.item(), confidence.item()


def main():
    parser = argparse.ArgumentParser(description="Predict image class using trained CNN model.")
    parser.add_argument("image_path", type=str)
    parser.add_argument("--model_path", type=str, default="./model/best_model.pth",
                        help="Path to the trained model file.")
    args = parser.parse_args()

    # Device configuration
    device = torch.device('cpu')

    # Load model
    model = load_model(args.model_path, device)

    # Preprocess image
    image_tensor = preprocess_image(args.image_path)

    # Predict
    predicted_class, confidence = predict_image(model, image_tensor, device)

    print(f"Predicted class: {CLASS_NAMES[predicted_class]}")
    print(f"Confidence: {confidence * 100:.2f}%")

    # Optionally display the image
    img = Image.open(args.image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {CLASS_NAMES[predicted_class]}")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()