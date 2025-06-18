#!/usr/bin/env python3
"""
Data module for creating datasets and dataloaders
"""
import os
import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from collections import Counter


def create_transforms(args):
    """Create training and validation/test transforms"""
    # Standard ImageNet normalization - common for pretrained models
    image_mean = (0.485, 0.456, 0.406)
    image_std = (0.229, 0.224, 0.225)
    
    # Augmentations for training
    train_transform = transforms.Compose([
        # Randomly crop a region and resize it to the target input size
        transforms.RandomResizedCrop(args.input_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),  # Fundus images can be flipped in any direction
        transforms.ColorJitter(
            brightness=args.color_jitter,
            contrast=args.color_jitter,
            saturation=args.color_jitter
        ),
        transforms.ToTensor(),
        # Normalize pixel values using ImageNet stats
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    # Simpler transforms for validation/testing
    val_transform = transforms.Compose([
        # Resize the image (maintaining aspect ratio) so shorter side is input_size + 32
        transforms.Resize(args.input_size + 32),  
        # Crop the center of the image to the target input size
        transforms.CenterCrop(args.input_size),
        transforms.ToTensor(),
        # Normalize pixel values using ImageNet stats
        transforms.Normalize(mean=image_mean, std=image_std)
    ])
    
    return train_transform, val_transform


def create_dataloaders(args):
    """Create train, validation, and test dataloaders"""
    train_transform, val_transform = create_transforms(args)
    
    # Create datasets
    train_path = os.path.join(args.data_path, 'train')
    val_path = os.path.join(args.data_path, 'val')
    test_path = os.path.join(args.data_path, 'test')
    
    # Verify paths exist
    for path, name in [(train_path, "train"), (val_path, "validation"), (test_path, "test")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{name} dataset path not found: {path}")
    
    # Create datasets
    train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_path, transform=val_transform)
    test_dataset = datasets.ImageFolder(test_path, transform=val_transform)
    
    # Get class names and verify consistency
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # Check class consistency
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        print("WARNING: Train and validation class mappings differ!")
        print(f"Train: {train_dataset.class_to_idx}")
        print(f"Val: {val_dataset.class_to_idx}")
    
    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        print("WARNING: Train and test class mappings differ!")
        print(f"Train: {train_dataset.class_to_idx}")
        print(f"Test: {test_dataset.class_to_idx}")
    
    # Print dataset info
    print(f"Found {len(train_dataset)} training images in {num_classes} classes")
    print(f"Found {len(val_dataset)} validation images")
    print(f"Found {len(test_dataset)} test images")

    # --- Calculate Class Weights for Training Set ---
    # Get the list of target indices (labels) for the training dataset
    train_targets = train_dataset.targets
    # Count occurrences of each class index
    train_class_counts = Counter(train_targets) 
    # Calculate total number of training samples
    total_train_samples = len(train_dataset)
    
    # Calculate weights for each class
    class_weights = []
    
    for i in range(num_classes):
        # Get count for class i, handle potential zero count 
        count = train_class_counts.get(i, 1) # Use 1 to avoid division by zero
        # Calculate weight using the formula: total_samples / (num_classes * samples_in_class)
        weight = total_train_samples / (num_classes * count)
        class_weights.append(weight)
        
    # Convert weights to a PyTorch tensor
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    # -------------------------------------------------
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory
    )
    
    # Return dataloaders, class info, and class weights
    return train_loader, val_loader, test_loader, num_classes, class_names, class_weights_tensor