import random
import numpy as np
import torch

from data.dataset import get_dataloaders

#Setting seed
def set_seed(seed = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():
    set_seed()
    device = torch.device('cpu')
    train_loader, val_loader, _ = get_dataloaders()


if __name__ == '__main__':
    main()
