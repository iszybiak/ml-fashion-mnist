from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int = 64):
    """
    Creates DataLoaders for the training, validation, and test sets
    for Fashion-MNIST.
    """

    # Transformations for the training set (with augmentation)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Transformations for validation and test (without augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Download training set
    train = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    # Download test set
    test = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    #Split data into train and validate
    train_size = int(0.80 * len(train))
    val_size = len(train) - train_size

    train_dataset, val_dataset = random_split(
        train, [train_size, val_size])

    #Disabling augmentation for valid
    val_dataset.dataset.transform = test_transform

    #DataLoaders
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader