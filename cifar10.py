import os
import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image

def load_cifar10_batch(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    images = dict[b'data']
    labels = dict[b'labels']
    images = images.reshape(-1, 3, 32, 32).astype("float32")
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
    
    test_file = os.path.join(root, 'test_batch')
    test_images, test_labels = load_cifar10_batch(test_file)
    
    return (train_images, train_labels), (test_images, test_labels)

class CustomCIFAR10(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].transpose(1, 2, 0)  # Convert from CHW to HWC
        image = Image.fromarray(np.uint8(image))  # Convert to PIL Image
        
        if self.transform:
            image = self.transform(image)
        
        label = self.labels[idx]
        return image, label


if __name__ == '__main__':
    # Load the CIFAR-10 data from files
    train_data, test_data = load_cifar10('/home/slaing/ML/2nd_year/sem2/research/data/cifar-10-batches-py')

    print(len(train_data[0]), len(test_data[0]))

    # Define transformations (optional)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    # Create custom dataset instances
    train_dataset = CustomCIFAR10(images=train_data[0], labels=train_data[1], transform=transform)
    test_dataset = CustomCIFAR10(images=test_data[0], labels=test_data[1], transform=transform)

    # Create DataLoader instances
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


