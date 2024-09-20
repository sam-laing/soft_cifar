import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import torch

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    labels = dict[b'labels']
    images = images.reshape(-1, 32, 32, 3).astype("float32")
    return images, labels

def load_cifar10(root):
    train_images = []
    train_labels = []
    for i in range(1, 6):
        file = os.path.join(root, f'data_batch_{i}')
        images, labels = load_cifar10_batch(file)
        train_images.append(images)
        train_labels.extend(labels)
    
    train_images = np.concatenate(train_images, axis=0)
    train_labels = np.array(train_labels)

    max_indices = train_labels
    labels = np.zeros((len(max_indices), 10))
    np.put_along_axis(labels, max_indices[:, np.newaxis], 1, axis=1)
    labels = torch.tensor(labels, dtype=torch.float)  

    return (train_images, labels)

class CustomCIFAR10(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].astype(np.uint8)  # Convert to uint8
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def make_larger_hard_datasets(path, split_ratio=[0.7, 0.15, 0.15], do_augmentation=True, seed=99):
    data, labels = load_cifar10(path)

    N = len(data)
    N_train, N_val = int(N * split_ratio[0]), int(N * split_ratio[1])
    N_test = N - N_train - N_val

    np.random.seed(seed)
    indices = np.random.permutation(N)

    train_indices = indices[:N_train]
    val_indices = indices[N_train:N_train + N_val]
    test_indices = indices[N_train + N_val:]

    if do_augmentation:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.2, hue=0.15),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])

    non_train_transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    train_dataset = CustomCIFAR10(images=data[train_indices], labels=labels[train_indices], transform=transform)
    val_dataset = CustomCIFAR10(images=data[val_indices], labels=labels[val_indices], transform=non_train_transform)
    test_dataset = CustomCIFAR10(images=data[test_indices], labels=labels[test_indices], transform=non_train_transform)

    return train_dataset, val_dataset, test_dataset

def make_larger_hard_loaders(path, split_ratio=[0.7, 0.15, 0.15], do_augmentation=True, batch_size=64):
    train_dataset, val_dataset, test_dataset = make_larger_hard_datasets(path, split_ratio, do_augmentation)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    data, labels = load_cifar10('/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py')
    print(data.shape, labels.shape)

    path = "/mnt/qb/work/oh/owl886/datasets/cifar-10-batches-py"

    train_dataset, val_dataset, test_dataset = make_larger_hard_datasets(path)
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    train_loader, val_loader, test_loader = make_larger_hard_loaders(path)
    for x, y in train_loader:
        print(x.shape, y.shape)
        break

    '''
    from model_structure import make_resnet_cifar
    model = make_resnet_cifar(depth=20)
    sample, slabel = train_dataset[0]
    print(model(sample.unsqueeze(0)).shape)

    for i, (input, label) in enumerate(train_loader):
        print(model(input).shape)
        break
    '''