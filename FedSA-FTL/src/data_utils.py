"""
Dataset utilities for FedSA-FTL experiments
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from typing import List, Tuple, Dict, Optional
import logging
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DirichletDataSplitter:
    """
    Split data among clients using Dirichlet distribution for non-IID simulation
    """
    
    def __init__(self, alpha: float = 0.5, min_samples_per_client: int = 10):
        """
        Args:
            alpha: Dirichlet concentration parameter (lower = more non-IID)
            min_samples_per_client: Minimum samples per client
        """
        self.alpha = alpha
        self.min_samples_per_client = min_samples_per_client
    
    def split_dataset(self, 
                     dataset: Dataset, 
                     num_clients: int, 
                     num_classes: int) -> List[List[int]]:
        """
        Split dataset indices among clients using Dirichlet distribution
        
        Args:
            dataset: PyTorch dataset
            num_clients: Number of clients
            num_classes: Number of classes in dataset
            
        Returns:
            List of lists, where each inner list contains indices for one client
        """
        # Get labels from dataset
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            # Extract labels by iterating through dataset
            labels = []
            for _, label in dataset:
                labels.append(label)
            labels = np.array(labels)
        
        # Create index arrays for each class
        class_indices = {}
        for class_id in range(num_classes):
            class_indices[class_id] = np.where(labels == class_id)[0]
        
        # Generate Dirichlet distribution for each class
        client_indices = [[] for _ in range(num_clients)]
        
        for class_id in range(num_classes):
            class_idx = class_indices[class_id]
            np.random.shuffle(class_idx)
            
            # Sample from Dirichlet distribution
            proportions = np.random.dirichlet(np.repeat(self.alpha, num_clients))
            
            # Convert proportions to sample counts
            proportions = np.array([p * len(class_idx) for p in proportions])
            proportions = proportions.astype(int)
            
            # Ensure all samples are distributed
            proportions[-1] += len(class_idx) - proportions.sum()
            
            # Distribute samples
            start_idx = 0
            for client_id in range(num_clients):
                end_idx = start_idx + proportions[client_id]
                client_indices[client_id].extend(class_idx[start_idx:end_idx])
                start_idx = end_idx
        
        # Shuffle indices for each client and ensure minimum samples
        final_client_indices = []
        for client_id in range(num_clients):
            indices = client_indices[client_id]
            np.random.shuffle(indices)
            
            # Ensure minimum samples per client
            if len(indices) < self.min_samples_per_client:
                logger.warning(f"Client {client_id} has only {len(indices)} samples, "
                             f"minimum is {self.min_samples_per_client}")
            
            final_client_indices.append(indices)
        
        return final_client_indices
    
    def get_client_class_distribution(self, 
                                    dataset: Dataset, 
                                    client_indices: List[List[int]], 
                                    num_classes: int) -> Dict[int, Dict[int, int]]:
        """
        Get class distribution for each client
        
        Args:
            dataset: PyTorch dataset
            client_indices: List of client index lists
            num_classes: Number of classes
            
        Returns:
            Dictionary mapping client_id to class distribution
        """
        # Get labels
        if hasattr(dataset, 'targets'):
            labels = np.array(dataset.targets)
        elif hasattr(dataset, 'labels'):
            labels = np.array(dataset.labels)
        else:
            labels = []
            for _, label in dataset:
                labels.append(label)
            labels = np.array(labels)
        
        client_distributions = {}
        
        for client_id, indices in enumerate(client_indices):
            client_labels = labels[indices]
            class_counts = {}
            
            for class_id in range(num_classes):
                class_counts[class_id] = np.sum(client_labels == class_id)
            
            client_distributions[client_id] = class_counts
        
        return client_distributions


def create_cifar10_dataloaders(data_dir: str = './data',
                              num_clients: int = 10,
                              alpha: float = 0.5,
                              batch_size: int = 32,
                              test_batch_size: int = 128) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Create CIFAR-10 dataloaders for federated learning
    
    Args:
        data_dir: Directory to store/load data
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
        batch_size: Training batch size
        test_batch_size: Test batch size
        
    Returns:
        Tuple of (client_train_loaders, global_val_loader, global_test_loader)
    """
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_test)
    
    # Split training data for validation
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))), test_size=0.1, random_state=42, stratify=train_dataset.targets
    )
    
    # Create validation dataset
    val_dataset = Subset(datasets.CIFAR10(root=data_dir, train=True, download=False, transform=transform_test), val_indices)
    train_dataset = Subset(train_dataset, train_indices)
    
    # Split training data among clients
    splitter = DirichletDataSplitter(alpha=alpha)
    client_indices = splitter.split_dataset(train_dataset, num_clients, num_classes=10)
    
    # Create client dataloaders
    client_train_loaders = []
    for indices in client_indices:
        client_subset = Subset(train_dataset, indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        client_train_loaders.append(client_loader)
    
    # Create global validation and test loaders
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    # Log data distribution
    logger.info(f"Created CIFAR-10 federated setup:")
    logger.info(f"  - {num_clients} clients with alpha={alpha}")
    logger.info(f"  - Training samples per client: {[len(indices) for indices in client_indices]}")
    logger.info(f"  - Validation samples: {len(val_dataset)}")
    logger.info(f"  - Test samples: {len(test_dataset)}")
    
    return client_train_loaders, val_loader, test_loader


def create_cifar100_dataloaders(data_dir: str = './data',
                               num_clients: int = 10,
                               alpha: float = 0.5,
                               batch_size: int = 32,
                               test_batch_size: int = 128) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
    """
    Create CIFAR-100 dataloaders for federated learning
    """
    # Data transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
    ])
    
    # Load datasets
    train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform_test)
    
    # Split training data for validation
    train_indices, val_indices = train_test_split(
        list(range(len(train_dataset))), test_size=0.1, random_state=42, stratify=train_dataset.targets
    )
    
    val_dataset = Subset(datasets.CIFAR100(root=data_dir, train=True, download=False, transform=transform_test), val_indices)
    train_dataset = Subset(train_dataset, train_indices)
    
    # Split training data among clients
    splitter = DirichletDataSplitter(alpha=alpha)
    client_indices = splitter.split_dataset(train_dataset, num_clients, num_classes=100)
    
    # Create client dataloaders
    client_train_loaders = []
    for indices in client_indices:
        client_subset = Subset(train_dataset, indices)
        client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=2)
        client_train_loaders.append(client_loader)
    
    # Create global validation and test loaders
    val_loader = DataLoader(val_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=2)
    
    logger.info(f"Created CIFAR-100 federated setup:")
    logger.info(f"  - {num_clients} clients with alpha={alpha}")
    logger.info(f"  - Training samples per client: {[len(indices) for indices in client_indices]}")
    logger.info(f"  - Validation samples: {len(val_dataset)}")
    logger.info(f"  - Test samples: {len(test_dataset)}")
    
    return client_train_loaders, val_loader, test_loader


class GLUEDataProcessor:
    """
    Process GLUE tasks for federated learning
    """
    
    def __init__(self, task_name: str, data_dir: str = './data'):
        self.task_name = task_name.lower()
        self.data_dir = data_dir
        
    def create_federated_dataloaders(self, 
                                   num_clients: int = 10,
                                   alpha: float = 0.5,
                                   batch_size: int = 16,
                                   max_length: int = 128) -> Tuple[List[DataLoader], DataLoader, DataLoader]:
        """
        Create federated dataloaders for GLUE tasks
        
        Note: This is a simplified implementation. For full GLUE support,
        you would need to integrate with transformers library and handle
        tokenization properly.
        """
        try:
            from transformers import AutoTokenizer
            from datasets import load_dataset
        except ImportError:
            raise ImportError("transformers and datasets libraries are required for GLUE tasks")
        
        # Load dataset
        if self.task_name == 'mnli':
            dataset = load_dataset('glue', 'mnli')
            num_classes = 3
        elif self.task_name == 'qnli':
            dataset = load_dataset('glue', 'qnli')
            num_classes = 2
        elif self.task_name == 'sst2':
            dataset = load_dataset('glue', 'sst2')
            num_classes = 2
        else:
            raise ValueError(f"Unsupported GLUE task: {self.task_name}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        
        # Tokenize function
        def tokenize_function(examples):
            if self.task_name == 'mnli':
                result = tokenizer(examples['premise'], examples['hypothesis'], 
                                 truncation=True, padding='max_length', max_length=max_length)
            elif self.task_name == 'qnli':
                result = tokenizer(examples['question'], examples['sentence'],
                                 truncation=True, padding='max_length', max_length=max_length)
            elif self.task_name == 'sst2':
                result = tokenizer(examples['sentence'],
                                 truncation=True, padding='max_length', max_length=max_length)
            
            result['labels'] = examples['label']
            return result
        
        # Tokenize datasets
        train_dataset = dataset['train'].map(tokenize_function, batched=True)
        val_dataset = dataset['validation'].map(tokenize_function, batched=True)
        test_dataset = dataset['test'].map(tokenize_function, batched=True) if 'test' in dataset else val_dataset
        
        # Convert to torch format
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
        # Split training data among clients (simplified - would need proper implementation)
        train_size = len(train_dataset)
        client_sizes = [train_size // num_clients] * num_clients
        
        # Create client dataloaders (simplified)
        client_train_loaders = []
        start_idx = 0
        for client_size in client_sizes:
            end_idx = start_idx + client_size
            client_subset = train_dataset.select(range(start_idx, end_idx))
            client_loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True)
            client_train_loaders.append(client_loader)
            start_idx = end_idx
        
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return client_train_loaders, val_loader, test_loader


def get_dataset_info(dataset_name: str) -> Dict[str, int]:
    """
    Get information about dataset
    
    Args:
        dataset_name: Name of dataset
        
    Returns:
        Dictionary with dataset information
    """
    dataset_info = {
        'cifar10': {'num_classes': 10, 'input_channels': 3, 'input_size': 32},
        'cifar100': {'num_classes': 100, 'input_channels': 3, 'input_size': 32},
        'mnli': {'num_classes': 3, 'sequence_length': 128},
        'qnli': {'num_classes': 2, 'sequence_length': 128},
        'sst2': {'num_classes': 2, 'sequence_length': 128}
    }
    
    if dataset_name.lower() not in dataset_info:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return dataset_info[dataset_name.lower()]


def analyze_data_distribution(client_loaders: List[DataLoader], num_classes: int) -> Dict[str, float]:
    """
    Analyze the data distribution across clients
    
    Args:
        client_loaders: List of client dataloaders
        num_classes: Number of classes
        
    Returns:
        Dictionary with distribution statistics
    """
    client_class_counts = []
    total_samples = 0
    
    for client_id, loader in enumerate(client_loaders):
        class_counts = np.zeros(num_classes)
        client_samples = 0
        
        for _, targets in loader:
            if isinstance(targets, torch.Tensor):
                targets = targets.numpy()
            elif hasattr(targets, 'labels'):
                targets = targets.labels.numpy()
            
            for target in targets:
                class_counts[target] += 1
                client_samples += 1
        
        client_class_counts.append(class_counts)
        total_samples += client_samples
    
    client_class_counts = np.array(client_class_counts)
    
    # Calculate non-IID metrics
    # 1. Class distribution variance across clients
    class_distributions = client_class_counts / client_class_counts.sum(axis=1, keepdims=True)
    class_variance = np.var(class_distributions, axis=0).mean()
    
    # 2. Number of classes per client
    classes_per_client = (client_class_counts > 0).sum(axis=1)
    avg_classes_per_client = classes_per_client.mean()
    
    # 3. Sample distribution across clients
    samples_per_client = client_class_counts.sum(axis=1)
    sample_variance = np.var(samples_per_client)
    
    return {
        'total_clients': len(client_loaders),
        'total_samples': int(total_samples),
        'avg_samples_per_client': float(samples_per_client.mean()),
        'sample_variance': float(sample_variance),
        'avg_classes_per_client': float(avg_classes_per_client),
        'class_distribution_variance': float(class_variance),
        'samples_per_client': samples_per_client.tolist(),
        'classes_per_client': classes_per_client.tolist()
    }
