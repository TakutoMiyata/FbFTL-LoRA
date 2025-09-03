"""
Data utilities for CIFAR-10/CIFAR-100 with non-IID split
"""

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from typing import List, Tuple, Dict


def get_cifar_transforms(model_type='vgg'):
    """Get CIFAR data transforms based on model type
    
    Args:
        model_type: 'vgg' for VGG models, 'vit' for Vision Transformers
    """
    if model_type == 'vgg':
        # VGG-16 expects 224x224 images, ImageNet normalization
        transform_train = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        transform_test = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    elif model_type == 'vit':
        # ViT works with 32x32 for CIFAR, use CIFAR normalization
        transform_train = transforms.Compose([
            # Keep original 32x32 size for CIFAR
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR stats
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616])  # CIFAR stats
        ])
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'vgg' or 'vit'")
    
    return transform_train, transform_test


def load_cifar_data(dataset_name='cifar100', data_dir='./data', model_type='vgg'):
    """Load CIFAR dataset
    
    Args:
        dataset_name: 'cifar10' or 'cifar100'
        data_dir: Directory to store/load data
        model_type: 'vgg' or 'vit' for appropriate transforms
    """
    transform_train, transform_test = get_cifar_transforms(model_type)
    
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
                          shuffle: bool = True) -> DataLoader:
    """
    Create DataLoader for a specific client
    
    Args:
        dataset: Full dataset
        client_indices: Indices for this client
        batch_size: Batch size
        shuffle: Whether to shuffle data
    
    Returns:
        DataLoader for the client
    """
    client_dataset = Subset(dataset, client_indices)
    return DataLoader(client_dataset, batch_size=batch_size, shuffle=shuffle)


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
    Prepare federated data based on configuration
    
    Args:
        config: Configuration dictionary
    
    Returns:
        trainset, testset, client_indices
    """
    # Set random seed for reproducibility
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))
    
    # Load CIFAR data (CIFAR-100 by default, CIFAR-10 optional)
    dataset_name = config.get('dataset_name', 'cifar100')
    # Detect model type from config or default to 'vgg' for backward compatibility
    model_type = config.get('model_type', 'vgg')
    trainset, testset = load_cifar_data(dataset_name, config.get('data_dir', './data'), model_type)
    
    # Determine number of classes based on dataset
    num_classes = 100 if dataset_name.lower() == 'cifar100' else 10
    
    # Create data splits
    num_clients = config.get('num_clients', 10)
    data_split = config.get('data_split', 'non_iid')
    
    if data_split == 'non_iid':
        alpha = config.get('alpha', 0.5)
        client_indices = create_non_iid_splits(trainset, num_clients, alpha)
    else:
        client_indices = create_iid_splits(trainset, num_clients)
    
    # Analyze distribution if verbose
    if config.get('verbose', False):
        analyze_data_distribution(trainset, client_indices, num_classes)
    
    return trainset, testset, client_indices