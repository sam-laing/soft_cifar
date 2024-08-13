import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import random

class SVHNOneHotDataset(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.svhn = datasets.SVHN(root=root, download=True, transform=transform)
        self.num_classes = 10  

    def __len__(self):
        return len(self.svhn)

    def __getitem__(self, idx):
        image, label = self.svhn[idx]
        one_hot_label = F.one_hot(torch.tensor(label), num_classes=self.num_classes).float()
        return image, one_hot_label

def get_svhn_loader(subset_size=1000, random_seed=66):
    """Builds and returns Dataloader for a subset of the SVHN dataset."""
    random.seed(random_seed) 
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    svhn_dataset = SVHNOneHotDataset(root="/home/slaing/Downloads/", transform=transform)
    
    # Create a subset of the dataset
    indices = list(range(len(svhn_dataset)))
    random.shuffle(indices)
    subset_indices = indices[:subset_size]
    svhn_subset = Subset(svhn_dataset, subset_indices)

    svhn_loader = DataLoader(dataset=svhn_subset,
                             batch_size=128,
                             shuffle=True)
    
    return svhn_loader

# Example usage

if __name__ == "__main__":
    svhn_loader = get_svhn_loader()
    print(svhn_loader)
    for images, labels in svhn_loader:
        print(images.shape)
        print(labels.shape)
        break