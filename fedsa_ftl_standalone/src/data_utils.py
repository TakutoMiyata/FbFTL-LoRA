"""
Data utilities for CIFAR-10/CIFAR-100 with non-IID split
Enhanced with advanced data augmentation including Mixup and CutMix
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Tuple, Dict, Optional, Callable
import random


def get_cifar_transforms(model_type='vgg', augmentation_config=None, use_cifar_resnet=False):
    """Get CIFAR data transforms based on model type with enhanced augmentation
    
    Args:
        model_type: 'vgg', 'vit', 'resnet', or other model types
        augmentation_config: Dict with augmentation settings
        use_cifar_resnet: If True, use 32x32 for CIFAR-optimized ResNet
    """
    # Default augmentation config
    if augmentation_config is None:
        augmentation_config = {}
    
    # Build training transforms based on configuration
    train_transforms = []
    
    # Resize based on model requirements
    if use_cifar_resnet:
        # CIFAR-optimized ResNet uses native 32x32 resolution
        pass  # No resize needed
    else:
        # ImageNet-pretrained models need 224x224
        train_transforms.append(transforms.Resize((224, 224)))
    
    # Random Crop with padding (if enabled)
    if augmentation_config.get('random_crop', {}).get('enabled', False):
        crop_config = augmentation_config['random_crop']
        if use_cifar_resnet:
            # For CIFAR-optimized ResNet, crop at 32x32
            train_transforms.append(
                transforms.RandomCrop(32, padding=crop_config.get('padding', 4))
            )
        else:
            # For ImageNet models, crop at 224x224
            train_transforms.append(
                transforms.RandomCrop(224, padding=crop_config.get('padding', 4))
            )
    
    # Horizontal Flip
    if augmentation_config.get('horizontal_flip', {}).get('enabled', True):
        flip_prob = augmentation_config.get('horizontal_flip', {}).get('prob', 0.5)
        train_transforms.append(transforms.RandomHorizontalFlip(p=flip_prob))
    
    # Color Jitter
    if augmentation_config.get('color_jitter', {}).get('enabled', False):
        jitter_config = augmentation_config['color_jitter']
        train_transforms.append(
            transforms.ColorJitter(
                brightness=jitter_config.get('brightness', 0.2),
                contrast=jitter_config.get('contrast', 0.2),
                saturation=jitter_config.get('saturation', 0.2),
                hue=jitter_config.get('hue', 0.1)
            )
        )
    
    # Random Rotation
    if augmentation_config.get('random_rotation', {}).get('enabled', True):
        degrees = augmentation_config.get('random_rotation', {}).get('degrees', 10)
        train_transforms.append(transforms.RandomRotation(degrees))
    
    # Random Erasing (similar to CutOut)
    if augmentation_config.get('random_erasing', {}).get('enabled', False):
        # This will be applied after ToTensor
        pass
    
    # Convert to tensor
    train_transforms.append(transforms.ToTensor())
    
    # Random Erasing (must be after ToTensor)
    if augmentation_config.get('random_erasing', {}).get('enabled', False):
        erase_config = augmentation_config['random_erasing']
        train_transforms.append(
            transforms.RandomErasing(
                p=erase_config.get('prob', 0.5),
                scale=erase_config.get('scale', (0.02, 0.33)),
                ratio=erase_config.get('ratio', (0.3, 3.3))
            )
        )
    
    # Normalize (ImageNet statistics)
    train_transforms.append(
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    )
    
    transform_train = transforms.Compose(train_transforms)
    
    # Test transforms (minimal augmentation)
    if use_cifar_resnet:
        # CIFAR-optimized ResNet: no resize needed
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
    else:
        # ImageNet models: resize to 224x224
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        ])
    
    return transform_train, transform_test


# ====================== Mixup and CutMix Implementation ======================

class MixupCutmixCollate:
    """Collate function that applies Mixup or CutMix to batches"""
    
    def __init__(self, mixup_alpha=0.2, cutmix_alpha=1.0, 
                 mixup_prob=0.5, cutmix_prob=0.5, num_classes=100):
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.num_classes = num_classes
        self.use_mixup = mixup_alpha > 0 and mixup_prob > 0
        self.use_cutmix = cutmix_alpha > 0 and cutmix_prob > 0
    
    def __call__(self, batch):
        """Apply Mixup or CutMix to a batch"""
        # Standard collate
        images = torch.stack([item[0] for item in batch])
        targets = torch.tensor([item[1] for item in batch])
        
        if not self.training or (not self.use_mixup and not self.use_cutmix):
            return images, targets
        
        # Decide which augmentation to use
        use_cutmix = self.use_cutmix and np.random.rand() < self.cutmix_prob
        use_mixup = self.use_mixup and not use_cutmix and np.random.rand() < self.mixup_prob
        
        if use_cutmix:
            images, targets = self.cutmix(images, targets)
        elif use_mixup:
            images, targets = self.mixup(images, targets)
        
        return images, targets
    
    def mixup(self, images, targets):
        """Apply Mixup augmentation"""
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size)
        
        # Mix images
        mixed_images = lam * images + (1 - lam) * images[index]
        
        # Convert targets to one-hot and mix
        targets_a = torch.nn.functional.one_hot(targets, self.num_classes).float()
        targets_b = torch.nn.functional.one_hot(targets[index], self.num_classes).float()
        mixed_targets = lam * targets_a + (1 - lam) * targets_b
        
        return mixed_images, mixed_targets
    
    def cutmix(self, images, targets):
        """Apply CutMix augmentation"""
        batch_size = images.size(0)
        
        # Sample lambda from Beta distribution
        lam = np.random.beta(self.cutmix_alpha, self.cutmix_alpha)
        
        # Random shuffle for mixing
        index = torch.randperm(batch_size)
        
        # Get image dimensions
        _, _, H, W = images.shape
        
        # Sample random box
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)
        
        # Uniform sampling of box center
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        # Box boundaries
        x1 = np.clip(cx - cut_w // 2, 0, W)
        y1 = np.clip(cy - cut_h // 2, 0, H)
        x2 = np.clip(cx + cut_w // 2, 0, W)
        y2 = np.clip(cy + cut_h // 2, 0, H)
        
        # Apply CutMix
        mixed_images = images.clone()
        mixed_images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
        
        # Adjust lambda based on actual box area
        lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
        
        # Mix targets
        targets_a = torch.nn.functional.one_hot(targets, self.num_classes).float()
        targets_b = torch.nn.functional.one_hot(targets[index], self.num_classes).float()
        mixed_targets = lam * targets_a + (1 - lam) * targets_b
        
        return mixed_images, mixed_targets
    
    def set_training(self, mode):
        """Set training mode for enabling/disabling augmentation"""
        self.training = mode


def load_cifar_data(dataset_name='cifar100', data_dir='./data', model_type='vgg', augmentation_config=None, use_cifar_resnet=False):
    """Load CIFAR dataset with augmentation support
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        data_dir: Directory to store/load data
        model_type: 'vgg', 'vit', 'resnet' for appropriate transforms
        augmentation_config: Configuration for data augmentation
        use_cifar_resnet: If True, use 32x32 for CIFAR-optimized ResNet
    """
    transform_train, transform_test = get_cifar_transforms(model_type, augmentation_config, use_cifar_resnet)
    
    if dataset_name.lower() == 'cifar100':
        trainset = torchvision.datasets.CIFAR100(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR100(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    elif dataset_name.lower() == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform_train
        )
        testset = torchvision.datasets.CIFAR10(
            root=data_dir, train=False, download=True, transform=transform_test
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Use 'cifar10' or 'cifar100'")
    
    return trainset, testset


def create_non_iid_splits(dataset, num_clients: int, alpha: float = 0.5) -> List[List[int]]:
    """
    Create non-IID data splits using Dirichlet distribution
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet distribution parameter (smaller = more heterogeneous)
    
    Returns:
        List of indices for each client
    """
    num_samples = len(dataset)
    labels = np.array([dataset[i][1] for i in range(num_samples)])
    num_classes = len(np.unique(labels))
    
    # Sort indices by label
    label_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Dirichlet distribution for each class
    client_indices = [[] for _ in range(num_clients)]
    
    for class_idx in range(num_classes):
        class_indices = label_indices[class_idx]
        np.random.shuffle(class_indices)
        
        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_indices)).astype(int)[:-1]
        
        # Split indices according to proportions
        class_splits = np.split(class_indices, proportions)
        
        for client_idx, split in enumerate(class_splits):
            client_indices[client_idx].extend(split.tolist())
    
    # Shuffle indices for each client
    for client_idx in range(num_clients):
        np.random.shuffle(client_indices[client_idx])
    
    return client_indices


def create_iid_splits(dataset, num_clients: int) -> List[List[int]]:
    """
    Create IID data splits
    
    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
    
    Returns:
        List of indices for each client
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    
    # Split evenly among clients
    splits = np.array_split(indices, num_clients)
    client_indices = [split.tolist() for split in splits]
    
    return client_indices


def get_client_dataloader(dataset, client_indices: List[int], batch_size: int, 
                          shuffle: bool = True, collate_fn: Optional[Callable] = None,
                          num_workers: int = 0) -> DataLoader:
    """
    Create DataLoader for a specific client with optional Mixup/CutMix
    
    Args:
        dataset: Full dataset
        client_indices: Indices for this client
        batch_size: Batch size
        shuffle: Whether to shuffle data
        collate_fn: Optional collate function for Mixup/CutMix
        num_workers: Number of workers for data loading
    
    Returns:
        DataLoader for the client
    """
    client_dataset = Subset(dataset, client_indices)
    return DataLoader(
        client_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


def analyze_data_distribution(dataset, client_indices: List[List[int]], num_classes: int = 100):
    """
    Analyze and print data distribution across clients
    
    Args:
        dataset: Full dataset
        client_indices: Indices for each client
        num_classes: Number of classes (100 for CIFAR-100, 10 for CIFAR-10)
    """
    print("\nData Distribution Analysis:")
    print("-" * 50)
    
    for client_id, indices in enumerate(client_indices):
        labels = [dataset[idx][1] for idx in indices]
        unique, counts = np.unique(labels, return_counts=True)
        
        print(f"Client {client_id}: {len(indices)} samples")
        distribution = dict(zip(unique, counts))
        
        # For CIFAR-100, show only non-zero classes to avoid clutter
        if num_classes > 10:
            print(f"  Classes present: {len(unique)} out of {num_classes}")
            for class_id in unique:
                count = distribution[class_id]
                percentage = (count / len(indices)) * 100
                print(f"    Class {class_id}: {count:4d} ({percentage:5.1f}%)")
        else:
            # Print distribution for all classes (CIFAR-10)
            for class_id in range(num_classes):
                count = distribution.get(class_id, 0)
                percentage = (count / len(indices)) * 100 if len(indices) > 0 else 0
                print(f"  Class {class_id}: {count:4d} ({percentage:5.1f}%)")
        print()


def prepare_federated_data(config: Dict):
    """
    Prepare federated data based on configuration with augmentation support
    
    Args:
        config: Configuration dictionary
    
    Returns:
        trainset, testset, client_train_indices, client_test_indices
    """
    # Set random seed for reproducibility
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    
    # Load CIFAR data with augmentation config
    dataset_name = config.get('dataset_name', 'cifar100')
    model_type = config.get('model_type', 'vgg')
    augmentation_config = config.get('augmentations', {})
    use_cifar_resnet = config.get('use_cifar_resnet', False)
    
    trainset, testset = load_cifar_data(
        dataset_name, 
        config.get('data_dir', './data'), 
        model_type,
        augmentation_config,
        use_cifar_resnet
    )
    
    # Determine number of classes based on dataset
    num_classes = 100 if dataset_name.lower() == 'cifar100' else 10
    
    # Create data splits
    num_clients = config.get('num_clients', 10)
    data_split = config.get('data_split', 'non_iid')
    
    # Create train splits
    if data_split == 'non_iid':
        alpha = config.get('alpha', 0.5)
        client_train_indices = create_non_iid_splits(trainset, num_clients, alpha)
        # Create test splits with the same distribution
        # Use a different seed offset to ensure different samples but same distribution
        np.random.seed(config.get('seed', 42) + 1000)
        client_test_indices = create_non_iid_splits(testset, num_clients, alpha)
        # Reset seed
        np.random.seed(config.get('seed', 42))
    else:
        client_train_indices = create_iid_splits(trainset, num_clients)
        client_test_indices = create_iid_splits(testset, num_clients)
    
    # Analyze distribution if verbose
    if config.get('verbose', False):
        print("\n=== Training Data Distribution ===")
        analyze_data_distribution(trainset, client_train_indices, num_classes)
        print("\n=== Test Data Distribution ===")
        analyze_data_distribution(testset, client_test_indices, num_classes)
    
    return trainset, testset, client_train_indices, client_test_indices