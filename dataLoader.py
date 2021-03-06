from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import random_split, DataLoader
from torchvision.utils import make_grid

def get_dataset():
    return MNIST('data/', download=True, transform=ToTensor())
    
def get_dataloader(val_size, batch_size=128):
    # return train_loader, val_loader
    dataset = get_dataset()
    set_size = len(dataset)
    train_size = set_size - val_size
    train_set, val_set = random_split(dataset, (train_size, val_size))
    train_loader = DataLoader(train_set, batch_size, shuffle=True, num_workers=5, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size, num_workers=5, pin_memory=True)
    return train_loader, val_loader
    

if __name__ == '__main__':
    dataset = get_dataset()
    image, target = dataset[0]   
    plt.imshow(image[0], cmap='gray')
    plt.show()
    print(f"Label: {target}")
    train_ld, val_ld = get_dataloader(10000)
    for batch in train_ld:
        images, targets = batch
        plt.figure(figsize=(16, 8))
        plt.axis('off')
        plt.imshow(make_grid(images, nrow=16).permute((1, 2, 0)))
        plt.show()
        break
