from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

DATA_DIR = './data'
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)
TRAIN_SPLIT = 0.8

def get_dataloaders(batch_size: int = 64) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Creates DataLoaders for the training, validation, and test sets
    for Fashion-MNIST.
    """
    try:
        # Transformations for the training set (with augmentation)
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])

        # Transformations for validation and test (without augmentation)
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(NORMALIZE_MEAN, NORMALIZE_STD)
        ])

        # Download training set
        train_dataset = datasets.FashionMNIST(
            root=DATA_DIR,
            train=True,
            download=True,
            transform=train_transform
        )

        # Download test set
        test_dataset = datasets.FashionMNIST(
            root=DATA_DIR,
            train=False,
            download=True,
            transform=test_transform
        )

    except Exception as e:
        raise RuntimeError(
            "Error retrieving or initializing Fashion-MNIST collection."
        ) from e


    #Split data into train and validate
    train_size = int(TRAIN_SPLIT * len(train_dataset))
    val_size = len(train_dataset) - train_size

    train_subset, val_subset = random_split(
        train_dataset,
[train_size, val_size])

    #Disabling augmentation for valid
    val_subset.dataset.transform = test_transform

    #DataLoaders
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader