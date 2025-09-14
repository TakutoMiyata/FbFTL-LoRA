#!/usr/bin/env python3
"""
Single Client ViT Learning Script
For comparison with Federated Learning approach
Uses the same model, augmentations, and training settings as FL for fair comparison
"""

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import yaml
from pathlib import Path
import os
import sys
import numpy as np
import random
import json
import time
from datetime import datetime
from tqdm import tqdm

# Load environment variables from .env file
def load_env_file(env_path='.env'):
    """Load environment variables from .env file"""
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    value = value.strip().strip('"').strip("'")
                    os.environ[key.strip()] = value
        print(f"✅ Environment variables loaded from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")

# Load .env file at startup
load_env_file()

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import from src modules
from fedsa_ftl_model_vit import create_model_vit
from data_utils import load_cifar_data, MixupCutmixCollate
from notification_utils import SlackNotifier


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Single Client ViT Training')
    parser.add_argument('--config', type=str, default='configs/single_client_vit_augmented.yaml',
                       help='Path to configuration file')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Override number of epochs')
    parser.add_argument('--model', type=str, choices=['vit_tiny', 'vit_small', 'vit_base'], 
                       default=None, help='Override ViT model variant')
    return parser.parse_args()


def load_and_validate_config(args):
    """Load and validate configuration file"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Configuration file not found: {config_path}")
        print("Available configs:")
        configs_dir = Path("configs")
        if configs_dir.exists():
            for config_file in configs_dir.glob("*single_client*.yaml"):
                print(f"  {config_file}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Apply command line overrides
    if args.epochs:
        config['training']['epochs'] = args.epochs
    if args.model:
        config['model']['model_name'] = args.model
    
    # Ensure ViT-specific settings
    config['data']['model_type'] = 'vit'
    
    return config, config_path


def initialize_environment(config):
    """Initialize random seeds and device"""
    if 'seed' in config:
        seed = config['seed']
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Optional: for deterministic behavior
        if config.get('reproducibility', {}).get('deterministic', False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            torch.backends.cudnn.benchmark = True
    
    device = torch.device('cuda' if config.get('use_gpu', False) and torch.cuda.is_available() else 'cpu')
    return device


def create_experiment_directory(config):
    """Create experiment directory and generate naming suffix"""
    current_date = datetime.now().strftime('%m%d')
    base_dir = Path(config.get('experiment', {}).get('output_dir', 'experiments/single_client_vit'))
    experiment_dir = base_dir / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    date_suffix = f"{current_date}SingleViT"
    return experiment_dir, date_suffix


def create_dataloader(dataset, batch_size, shuffle=True, collate_fn=None, num_workers=0):
    """Create DataLoader with optional Mixup/CutMix"""
    return torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers
    )


def create_optimizer(model, config):
    """Create optimizer based on configuration"""
    # Get LoRA parameters (trainable parameters)
    lora_params = [p for p in model.parameters() if p.requires_grad]
    
    lr = float(config.get('learning_rate', 0.0005))
    weight_decay = float(config.get('weight_decay', 0.001))
    optimizer_type = config.get('optimizer', 'adamw').lower()
    
    if optimizer_type == 'adamw':
        betas = tuple(config.get('betas', [0.9, 0.999]))
        eps = float(config.get('eps', 1e-8))
        return optim.AdamW(lora_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    elif optimizer_type == 'adam':
        betas = tuple(config.get('betas', [0.9, 0.999]))
        eps = float(config.get('eps', 1e-8))
        return optim.Adam(lora_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    else:
        # Default to SGD if unknown optimizer specified
        print(f"Warning: Unknown optimizer '{optimizer_type}', using SGD")
        return optim.SGD(lora_params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, total_epochs):
    """Train for one epoch"""
    model.train()
    
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    
    with tqdm(train_loader, desc=f"Epoch {epoch+1}/{total_epochs}") as pbar:
        for batch_idx, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device)
            
            # Check if labels are one-hot (from Mixup/CutMix)
            if labels.dim() == 2 and labels.size(1) > 1:
                # Mixup/CutMix: labels are soft (one-hot or mixed)
                outputs = model(images)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                # KL divergence loss (equivalent to cross entropy for soft labels)
                loss = -(labels * log_probs).sum(dim=1).mean()
                
                # For accuracy, use the argmax of soft labels
                _, predicted = outputs.max(1)
                _, label_indices = labels.max(1)
                epoch_correct += predicted.eq(label_indices).sum().item()
            else:
                # Standard labels (integer format)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                epoch_correct += predicted.eq(labels).sum().item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_total += labels.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'loss': epoch_loss / (batch_idx + 1),
                'acc': 100. * epoch_correct / epoch_total
            })
    
    return epoch_loss / len(train_loader), 100. * epoch_correct / epoch_total


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test data"""
    model.eval()
    
    test_loss = 0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            
            # Check if labels are one-hot (shouldn't happen in evaluation, but handle gracefully)
            if labels.dim() == 2 and labels.size(1) > 1:
                outputs = model(images)
                log_probs = torch.nn.functional.log_softmax(outputs, dim=1)
                loss = -(labels * log_probs).sum(dim=1).mean()
                
                _, predicted = outputs.max(1)
                _, label_indices = labels.max(1)
                test_correct += predicted.eq(label_indices).sum().item()
                test_loss += loss.item() * labels.size(0)
            else:
                # Standard labels (normal case for evaluation)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                test_correct += predicted.eq(labels).sum().item()
            
            test_total += labels.size(0)
    
    return test_loss / test_total, 100. * test_correct / test_total


def save_results(results, experiment_dir, date_suffix):
    """Save results to JSON file"""
    results_file = experiment_dir / f'training_results_{date_suffix}.json'
    try:
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Warning: Failed to save results file: {e}")


def create_summary_csv(results, experiment_dir, date_suffix):
    """Create CSV summary for visualization"""
    try:
        import pandas as pd
        df_data = []
        for epoch_data in results['epochs']:
            df_data.append({
                'epoch': epoch_data['epoch'],
                'train_loss': epoch_data['train_loss'],
                'train_accuracy': epoch_data['train_accuracy'],
                'test_loss': epoch_data['test_loss'],
                'test_accuracy': epoch_data['test_accuracy'],
                'is_best': epoch_data['is_best_epoch']
            })
        
        if df_data:
            df = pd.DataFrame(df_data)
            csv_file = experiment_dir / f'results_summary_{date_suffix}.csv'
            df.to_csv(csv_file, index=False)
            print(f"Summary CSV saved to: {csv_file}")
            return csv_file
            
    except ImportError:
        print("Warning: pandas not available, skipping CSV export")
    except Exception as e:
        print(f"Warning: Failed to create CSV summary: {e}")
    
    return None


def main():
    """Main entry point for single client ViT training"""
    # Parse arguments and load configuration
    args = parse_arguments()
    config, config_path = load_and_validate_config(args)
    
    # Initialize environment
    device = initialize_environment(config)
    
    # Initialize Slack notifier (optional)
    slack_notifier = None
    webhook_url = os.environ.get('SLACK_WEBHOOK_URL')
    if webhook_url:
        slack_notifier = SlackNotifier(webhook_url)
        print("Slack notifications enabled")
    else:
        print("Slack notifications disabled")
    
    # Print configuration summary
    print("=" * 80)
    print("Single Client ViT Training (for FL comparison)")
    print("=" * 80)
    print(f"Configuration: {config_path}")
    print(f"Device: {device}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Dataset: {config['data']['dataset_name'].upper()}")
    print(f"Epochs: {config['training']['epochs']}")
    print("=" * 80)
    
    # Prepare data
    print("\nPreparing data...")
    dataset_name = config['data']['dataset_name']
    model_type = config['data']['model_type']
    augmentation_config = config['data'].get('augmentations', {})
    
    trainset, testset = load_cifar_data(
        dataset_name, 
        config['data'].get('data_dir', './data'), 
        model_type,
        augmentation_config
    )
    
    # Prepare Mixup/CutMix if enabled
    collate_fn = None
    mixup_config = augmentation_config.get('mixup', {})
    cutmix_config = augmentation_config.get('cutmix', {})
    num_classes = config['model'].get('num_classes', 100)
    
    if mixup_config.get('enabled', False) or cutmix_config.get('enabled', False):
        print("Enabling Mixup/CutMix augmentation...")
        collate_fn = MixupCutmixCollate(
            mixup_alpha=mixup_config.get('alpha', 0.2) if mixup_config.get('enabled', False) else 0,
            cutmix_alpha=cutmix_config.get('alpha', 1.0) if cutmix_config.get('enabled', False) else 0,
            mixup_prob=mixup_config.get('prob', 0.5) if mixup_config.get('enabled', False) else 0,
            cutmix_prob=cutmix_config.get('prob', 0.5) if cutmix_config.get('enabled', False) else 0,
            num_classes=num_classes
        )
        collate_fn.set_training(True)
    
    # Create data loaders
    train_loader = create_dataloader(
        trainset,
        config['data']['batch_size'],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=config['data'].get('num_workers', 0)
    )
    
    test_loader = create_dataloader(
        testset,
        config['data']['batch_size'],
        shuffle=False,
        collate_fn=None,  # No augmentation for test
        num_workers=config['data'].get('num_workers', 0)
    )
    
    # Create model
    print(f"\nCreating {config['model']['model_name']} model...")
    model = create_model_vit(config['model'])
    model.to(device)
    
    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Architecture: {config['model']['model_name']}")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  LoRA rank: {config['model']['lora_r']}")
    
    # Create optimizer and loss function
    optimizer = create_optimizer(model, config['training'])
    criterion = nn.CrossEntropyLoss()
    
    # Create experiment directory
    experiment_dir, date_suffix = create_experiment_directory(config)
    print(f"Results will be saved to: {experiment_dir}")
    
    # Initialize results tracking
    results = {
        'config': config,
        'start_time': datetime.now().isoformat(),
        'experiment_dir': str(experiment_dir),
        'date_suffix': date_suffix,
        'epochs': [],
        'summary': {}
    }
    
    # Training loop
    best_accuracy = 0
    best_epoch = 0
    start_time = time.time()
    num_epochs = config['training']['epochs']
    eval_freq = config.get('evaluation', {}).get('eval_freq', 5)
    
    print(f"\nStarting training for {num_epochs} epochs...")
    print("=" * 80)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, num_epochs
        )
        
        # Evaluate
        is_eval_epoch = (epoch + 1) % eval_freq == 0
        if is_eval_epoch:
            test_loss, test_acc = evaluate(model, test_loader, criterion, device)
            print(f"\nEpoch {epoch+1}/{num_epochs} Summary:")
            print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"  Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
            
            # Track best accuracy
            is_best = test_acc > best_accuracy
            if is_best:
                best_accuracy = test_acc
                best_epoch = epoch + 1
                print(f"  ** New best accuracy! **")
        else:
            test_loss = 0
            test_acc = 0
            is_best = False
        
        # Save epoch results
        epoch_result = {
            'epoch': epoch + 1,
            'timestamp': datetime.now().isoformat(),
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'is_best_epoch': is_best
        }
        results['epochs'].append(epoch_result)
        
        # Save results after each epoch
        save_results(results, experiment_dir, date_suffix)
    
    # Final results
    end_time = time.time()
    training_duration = end_time - start_time
    
    results['end_time'] = datetime.now().isoformat()
    results['training_duration_seconds'] = training_duration
    results['summary'] = {
        'best_test_accuracy': best_accuracy,
        'best_epoch': best_epoch,
        'total_epochs': num_epochs,
        'final_train_accuracy': train_acc,
        'final_test_accuracy': test_acc if is_eval_epoch else best_accuracy,
        'training_duration_hours': training_duration / 3600,
        'model_name': config['model']['model_name'],
        'dataset': config['data']['dataset_name']
    }
    
    # Save final results
    final_results_file = experiment_dir / f'final_results_{date_suffix}.json'
    try:
        with open(final_results_file, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Final results saved to: {final_results_file}")
    except Exception as e:
        print(f"Warning: Failed to save final results: {e}")
    
    # Create summary CSV
    csv_file = create_summary_csv(results, experiment_dir, date_suffix)
    
    # Print final summary
    print("\n" + "=" * 80)
    print("Single Client ViT Training Complete!")
    print("=" * 80)
    print(f"Configuration: {config_path.name}")
    print(f"Model: {config['model']['model_name']}")
    print(f"Best Test Accuracy: {best_accuracy:.2f}% (Epoch {best_epoch})")
    print(f"Total Epochs: {num_epochs}")
    print(f"Training Duration: {training_duration / 3600:.2f} hours")
    print(f"Results saved to: {experiment_dir}")
    print(f"  - Training results: training_results_{date_suffix}.json")
    print(f"  - Final results: final_results_{date_suffix}.json")
    if csv_file:
        print(f"  - Summary CSV: {csv_file.name}")
    print(f"  - File naming pattern: *_{date_suffix}.*")
    print("=" * 80)
    
    # Performance evaluation
    if best_accuracy > 60:  # Higher threshold for single client
        print("🎉 Single client ViT training successful!")
    elif best_accuracy > 40:
        print("⚠️ Training in progress - consider more epochs or hyperparameter tuning")
    else:
        print("❌ Training may need hyperparameter adjustment")
    
    # Send completion notification
    if slack_notifier:
        try:
            config_for_notification = {
                'experiment': {
                    'name': f'Single Client ViT ({config["model"]["model_name"]})',
                    'output_dir': str(experiment_dir)
                }
            }
            
            summary_for_notification = {
                'final_test_accuracy': best_accuracy,
                'total_epochs': num_epochs,
                'training_duration_hours': training_duration / 3600
            }
            
            # Send notification (simplified)
            slack_notifier.send_training_complete(
                config_for_notification,
                summary_for_notification,
                training_duration
            )
        except Exception as e:
            print(f"Warning: Failed to send Slack notification: {e}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()