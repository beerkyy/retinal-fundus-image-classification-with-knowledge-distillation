# This file contains loading, preprocessing, inspecting and batching up code for retinal-fundus images for a 4-class classification problem 
# EE6892 Advanced Deep Learning on Edge Final Project
#
# Sources + custom implementation 
#
# @inproceedings{paszke2019pytorch,
#   author    = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Others},
#   title     = {{PyTorch}: An Imperative Style, High-Performance Deep Learning Library},
#   booktitle = {Advances in Neural Information Processing Systems 32},
#   pages     = {8024--8035},
#   year      = {2019}
# }
#
# @misc{torchvision,
#   author    = {{Torchvision} Contributors},
#   title     = {{Torchvision}: PyTorch’s Computer Vision Library},
#   howpublished = {\url{https://github.com/pytorch/vision}},
#   year      = {2020}
# }

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# List of class names exposed for import
class_names = ['Cataract', 'DR', 'Glaucoma', 'Normal']

class RetinalFundusDataset(Dataset):
    """Dataset class for retinal fundus images
       Custom 4-class classification dataset"""
    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir (str): Parent directory containing 'train', 'val', 'test' subfolders
            split (str): One of ['train', 'val', 'test']
            transform (callable, optional): Transformation pipeline to apply
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.classes = class_names
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        self.image_paths = []
        self.labels = []

        # Walk through each class directory
        for cls in self.classes:
            cls_dir = os.path.join(self.root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    path = os.path.join(cls_dir, fname)
                    self.image_paths.append(path)
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_transforms(img_size=224):
    """Define training and validation/test transforms"""
    # Training augmentations
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(12),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Validation/Test (no augmentation)
    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return {'train': train_transform, 'val': val_transform, 'test': val_transform}


def get_dataloaders(data_dir, batch_size=32, img_size=224, num_workers=4, pin_memory=True):
    """Create DataLoaders for train / val / test splits"""
    transforms_dict = get_data_transforms(img_size)
    datasets = {
        split: RetinalFundusDataset(root_dir=data_dir, split=split, transform=transforms_dict[split])
        for split in ['train', 'val', 'test']
    }
    loaders = {
        split: DataLoader(
            dataset=datasets[split], batch_size=batch_size,
            shuffle=(split=='train'), num_workers=num_workers,
            pin_memory=pin_memory
        )
        for split in ['train', 'val', 'test']
    }
    return loaders


def visualize_dataset_samples(dataloader, num_samples=5, classes=None):
    """Plot a few samples from a DataLoader"""
    if classes is None:
        classes = class_names
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(1, min(num_samples, len(images)), figsize=(15,3))
    for i in range(min(num_samples, len(images))):
        img = images[i].numpy().transpose(1,2,0)
        mean = np.array([0.485,0.456,0.406])
        std = np.array([0.229,0.224,0.225])
        img = np.clip(img*std + mean, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(classes[labels[i]])
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def get_class_distribution(dataloader):
    """Return dict of class index -> count"""
    counts = {}
    for _, labels in dataloader:
        for l in labels:
            l_int = int(l)
            counts[l_int] = counts.get(l_int, 0) + 1
    return counts
