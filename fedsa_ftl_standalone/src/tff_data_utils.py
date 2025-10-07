"""
TensorFlow Federated (TFF) CIFAR-100 data utilities for PyTorch
Converts TFF's hierarchical LDA non-IID dataset to PyTorch format
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Optional
import random

# Import TFF with error handling
try:
    import tensorflow_federated as tff
    import tensorflow as tf
    TFF_AVAILABLE = True
except ImportError:
    TFF_AVAILABLE = False
    print("Warning: TensorFlow Federated not available. Install with: pip install tensorflow-federated")


# ImageNet normalization for transfer learning
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class TFFCifar100Dataset(Dataset):
    """PyTorch Dataset wrapper for TFF CIFAR-100 client data"""

    def __init__(self, client_data, transform=None):
        """
        Args:
            client_data: TFF ClientData object for a single client
            transform: Optional transform to apply to images
        """
        self.transform = transform

        # Convert TFF dataset to numpy arrays
        self.images = []
        self.labels = []

        # Create dataset from TFF client data
        for batch in client_data:
            # TFF CIFAR-100 format: {'coarse_label', 'image', 'label'}
            images = batch['image'].numpy()
            labels = batch['label'].numpy()

            # Handle both batched and unbatched data
            if images.ndim == 3:  # Single image (H, W, C)
                self.images.append(images[np.newaxis, ...])  # Add batch dimension
            else:  # Batched images (N, H, W, C)
                self.images.append(images)
            
            if labels.ndim == 0:  # Single label (scalar)
                self.labels.append(np.array([labels]))  # Convert to 1D array
            else:  # Batched labels (N,)
                self.labels.append(labels)

        # Concatenate all batches
        if self.images:
            self.images = np.concatenate(self.images, axis=0)
            self.labels = np.concatenate(self.labels, axis=0)
        else:
            self.images = np.array([])
            self.labels = np.array([])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to PIL Image for transforms
        from PIL import Image
        if image.dtype == np.uint8:
            image = Image.fromarray(image)
        else:
            # Ensure values are in [0, 255] range
            image = np.clip(image * 255, 0, 255).astype(np.uint8)
            image = Image.fromarray(image)

        if self.transform:
            image = self.transform(image)

        return image, label


def load_tff_cifar100(train_client_ids: Optional[List[str]] = None,
                      test_client_ids: Optional[List[str]] = None,
                      input_size: int = 224,
                      augment_train: bool = True):
    """
    Load TFF CIFAR-100 dataset with hierarchical LDA non-IID split

    Args:
        train_client_ids: List of training client IDs to load (None = all 500)
        test_client_ids: List of test client IDs to load (None = all 100)
        input_size: Input image size for models
        augment_train: Whether to apply data augmentation for training

    Returns:
        train_datasets: Dict mapping client_id to PyTorch Dataset
        test_datasets: Dict mapping client_id to PyTorch Dataset
        client_info: Dict with dataset metadata
    """
    if not TFF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow Federated is not installed. "
            "Install with: pip install tensorflow-federated"
        )

    # Load TFF CIFAR-100 data
    print("Loading TFF CIFAR-100 dataset (hierarchical LDA non-IID)...")
    train_data, test_data = tff.simulation.datasets.cifar100.load_data()

    # Get all available client IDs
    all_train_ids = train_data.client_ids
    all_test_ids = test_data.client_ids

    print(f"Available training clients: {len(all_train_ids)}")
    print(f"Available test clients: {len(all_test_ids)}")

    # Select client IDs
    if train_client_ids is None:
        train_client_ids = all_train_ids
    if test_client_ids is None:
        test_client_ids = all_test_ids

    # Build transforms
    if augment_train:
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(input_size, scale=(0.5, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize(int(input_size * 256 / 224)),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    test_transform = transforms.Compose([
        transforms.Resize(int(input_size * 256 / 224)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    # Create PyTorch datasets for each client
    train_datasets = {}
    test_datasets = {}

    print(f"Creating PyTorch datasets for {len(train_client_ids)} training clients...")
    for client_id in train_client_ids:
        client_data = train_data.create_tf_dataset_for_client(client_id)
        train_datasets[client_id] = TFFCifar100Dataset(client_data, transform=train_transform)

    print(f"Creating PyTorch datasets for {len(test_client_ids)} test clients...")
    for client_id in test_client_ids:
        client_data = test_data.create_tf_dataset_for_client(client_id)
        test_datasets[client_id] = TFFCifar100Dataset(client_data, transform=test_transform)

    # Collect metadata
    client_info = {
        'num_train_clients': len(train_client_ids),
        'num_test_clients': len(test_client_ids),
        'train_client_ids': train_client_ids,
        'test_client_ids': test_client_ids,
        'total_available_train_clients': len(all_train_ids),
        'total_available_test_clients': len(all_test_ids),
        'split_method': 'hierarchical_LDA_non_IID'
    }

    return train_datasets, test_datasets, client_info


def select_fixed_test_clients(num_test_clients: int = 30, seed: int = 42):
    """
    Select a fixed set of test clients from TFF CIFAR-100

    Args:
        num_test_clients: Number of test clients to select (default: 30)
        seed: Random seed for reproducibility

    Returns:
        List of selected test client IDs
    """
    if not TFF_AVAILABLE:
        raise RuntimeError("TensorFlow Federated is not installed")

    # Load only test data to get client IDs
    _, test_data = tff.simulation.datasets.cifar100.load_data()
    all_test_ids = list(test_data.client_ids)

    # Random selection with fixed seed
    random.seed(seed)
    selected_test_ids = random.sample(all_test_ids, min(num_test_clients, len(all_test_ids)))

    print(f"Selected {len(selected_test_ids)} test clients (seed={seed}): {selected_test_ids[:5]}...")

    return selected_test_ids


def select_training_clients(num_train_clients: int = 10, seed: int = 42):
    """
    Select training clients from TFF CIFAR-100

    Args:
        num_train_clients: Number of training clients (5 or 10 recommended)
        seed: Random seed for reproducibility

    Returns:
        List of selected training client IDs
    """
    if not TFF_AVAILABLE:
        raise RuntimeError("TensorFlow Federated is not installed")

    train_data, _ = tff.simulation.datasets.cifar100.load_data()
    all_train_ids = list(train_data.client_ids)

    # Random selection with fixed seed
    random.seed(seed)
    selected_train_ids = random.sample(all_train_ids, min(num_train_clients, len(all_train_ids)))

    print(f"Selected {len(selected_train_ids)} training clients (seed={seed})")

    return selected_train_ids


def get_tff_dataloader(dataset: TFFCifar100Dataset,
                       batch_size: int,
                       shuffle: bool = True,
                       num_workers: int = 0,
                       pin_memory: bool = True):
    """
    Create DataLoader for TFF CIFAR-100 dataset

    Args:
        dataset: TFFCifar100Dataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of workers for data loading
        pin_memory: Whether to use pinned memory

    Returns:
        PyTorch DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )


def analyze_tff_distribution(datasets: Dict[str, TFFCifar100Dataset],
                             dataset_type: str = "train",
                             num_classes: int = 100):
    """
    Analyze and print data distribution across TFF clients

    Args:
        datasets: Dictionary mapping client_id to Dataset
        dataset_type: "train" or "test"
        num_classes: Number of classes (100 for CIFAR-100)
    """
    print(f"\n{'='*60}")
    print(f"TFF CIFAR-100 {dataset_type.capitalize()} Data Distribution Analysis")
    print(f"{'='*60}")

    for client_id, dataset in list(datasets.items())[:10]:  # Show first 10 clients
        labels = dataset.labels
        unique, counts = np.unique(labels, return_counts=True)

        print(f"\nClient {client_id}: {len(labels)} samples")
        print(f"  Classes present: {len(unique)} out of {num_classes}")

        # Show top 5 most frequent classes
        top_5_idx = np.argsort(counts)[-5:][::-1]
        for idx in top_5_idx:
            class_id = unique[idx]
            count = counts[idx]
            percentage = (count / len(labels)) * 100
            print(f"    Class {class_id}: {count:4d} ({percentage:5.1f}%)")

    if len(datasets) > 10:
        print(f"\n... ({len(datasets) - 10} more clients)")

    print(f"{'='*60}\n")


def prepare_tff_federated_data(config: Dict):
    """
    Prepare TFF CIFAR-100 federated data based on configuration

    Args:
        config: Configuration dictionary

    Returns:
        train_datasets: Dict of training datasets per client
        test_datasets: Dict of test datasets per client
        client_info: Metadata about clients
    """
    # Set random seed
    seed = config.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    # Get configuration
    num_train_clients = config.get('num_clients', 10)
    num_test_clients = config.get('num_test_clients', 30)
    input_size = config.get('input_size', 224)
    augmentation_config = config.get('augmentations', {})

    # Determine if augmentation is enabled
    augment_train = any(
        aug_config.get('enabled', False)
        for aug_config in augmentation_config.values()
        if isinstance(aug_config, dict)
    )

    # Select clients (use same IDs for train and test to maintain non-IID consistency)
    train_client_ids = select_training_clients(num_train_clients, seed)
    # Use same client IDs for test data (TFF convention: same client has both train and test splits)
    test_client_ids = train_client_ids

    print(f"\n📊 Using same client IDs for train and test (TFF convention)")
    print(f"   Selected {len(train_client_ids)} clients: {train_client_ids[:5]}{'...' if len(train_client_ids) > 5 else ''}")

    # Load data
    train_datasets, test_datasets, client_info = load_tff_cifar100(
        train_client_ids=train_client_ids,
        test_client_ids=test_client_ids,
        input_size=input_size,
        augment_train=augment_train
    )

    # Analyze distribution if verbose
    if config.get('verbose', False):
        analyze_tff_distribution(train_datasets, "train")
        analyze_tff_distribution(test_datasets, "test")

    print(f"\n✅ TFF CIFAR-100 data prepared:")
    print(f"  Training clients: {num_train_clients}")
    print(f"  Test clients: {num_train_clients} (same as training)")
    print(f"  Split method: Hierarchical LDA (non-IID)")
    print(f"  Each client has ~100 train samples + ~100 test samples")

    return train_datasets, test_datasets, client_info
