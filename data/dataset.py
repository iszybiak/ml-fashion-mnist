from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

def get_dataloaders(batch_size: int = 64):
    #Transfroms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    #Dataet
    train = datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test = datasets.FashionMNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    #Split data into train and validate
    train_size = int(0.80 * len(train))
    val_size = len(train) - train_size

    train_dataset, val_dataset = random_split(
        train, [train_size, val_size])


    #DataLoader
    train_loader = DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size,shuffle=False)
    test_loader = DataLoader(test,batch_size=batch_size,shuffle=False)

    return train_loader, val_loader, test_loader