from reader import ReaderSoft
from torch.utils.data import DataLoader
# import datasets from torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms
from PIL import Image
import os
import sys
import torch
import os

class CIFAR10Soft(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)

        return image, label
    
def make_reader(path=None):
    reader = ReaderSoft(path)
    # make a probability distribution for each row of soft labels
    reader.soft_labels = reader.soft_labels/reader.soft_labels.sum(axis=1, keepdims=True)

    imgs = []
    for img in reader.filepath_to_imgid.keys():
        img = plt.imread(os.path.join(reader.root, img))
        img = (img*255).astype(np.uint8)
        imgs.append(img)

    imgs_array = np.array(imgs)
    reader.data = imgs_array

    return reader


def make_datasets(reader: ReaderSoft, split_ratio:list = [0.8, 0.05, 0.15], use_hard_labels:bool = False, do_augmentation:bool = True, entropy_threshold:float = None, seed=None):
    if entropy_threshold is not None:
        """   
        if the entropy of a sample is sufficiently low (below threshold),
        then just use the hard label for that sample
        otherwise use the soft label
        """
        assert use_hard_labels == False, "Cannot use hard labels and entropy thresholding at the same time"
        @torch.no_grad()
        def entropy(p):
            return torch.sum(-p * torch.nan_to_num(torch.log(p)), dim = -1)

        # get the entropy of the soft labels
        entropies = entropy(reader.soft_labels)

        # get the indices of the images with lower than threshold entropy
        lower_entropy_indices = torch.where(entropies < entropy_threshold)[0]


        hard_labels = reader.soft_labels.argmax(axis=1)
        reader.soft_labels[lower_entropy_indices] = torch.nn.functional.one_hot(hard_labels[lower_entropy_indices], num_classes=reader.soft_labels.size(1)).float()



    N = reader.data.shape[0]
    N_train, N_val, N_test = [int(r * N) for r in split_ratio]
    idx = range(N)
    if (seed is not None) and (seed != 666) :
        np.random.seed(seed)
        idx = np.random.permutation(idx)

    train_indices = idx[:N_train]
    val_indices = idx[N_train:N_train + N_val]
    test_indices = idx[N_train + N_val:]


    train_mean = reader.data[train_indices].mean(axis=(0,1,2))/255
    train_std = reader.data[train_indices].std(axis=(0,1,2))/255

    if do_augmentation:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)
        ])

    non_train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(train_mean, train_std)
    ])

    if use_hard_labels:
        max_indices = reader.soft_labels.argmax(axis=1).numpy()

        hard_labels = np.zeros_like(reader.soft_labels)
        np.put_along_axis(hard_labels, max_indices[:, np.newaxis], 1, axis=1)
        hard_labels = torch.tensor(hard_labels, dtype=torch.float)  # Convert to PyTorch tensor with appropriate dtype

        train_dataset = CIFAR10Soft(reader.data[train_indices], hard_labels[train_indices], transform=transform)
        val_dataset = CIFAR10Soft(reader.data[val_indices], hard_labels[val_indices], transform=non_train_transform)
        test_dataset = CIFAR10Soft(reader.data[test_indices], hard_labels[test_indices], transform=non_train_transform)
    else:
        train_dataset = CIFAR10Soft(reader.data[train_indices], reader.soft_labels[train_indices], transform=transform)
        val_dataset = CIFAR10Soft(reader.data[val_indices], reader.soft_labels[val_indices], transform=non_train_transform)
        test_dataset = CIFAR10Soft(reader.data[test_indices], reader.soft_labels[test_indices], transform=non_train_transform)

    return train_dataset, val_dataset, test_dataset

def make_loaders(reader: ReaderSoft, batch_size:int = 128, split_ratio:list = [0.8, 0.05, 0.15], use_hard_labels:bool = False, do_augmentation:bool = True, entropy_threshold:float = None, seed=None):
    train_dataset, val_dataset, test_dataset = make_datasets(reader, split_ratio, use_hard_labels, do_augmentation, entropy_threshold, seed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

