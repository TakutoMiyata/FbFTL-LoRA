"""
FedSA-FTL Client Implementation
Handles local training with LoRA adaptation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from typing import Dict, Optional
from tqdm import tqdm


class FedSAFTLClient:
    """
    FedSA-FTL Client for local training
    """
    
    def __init__(self, client_id: int, model: nn.Module, device: torch.device = None):
        """
        Initialize FedSA-FTL client
        
        Args:
            client_id: Unique client identifier
            model: FedSAFTLModel instance
            device: Device to run training on
        """
        self.client_id = client_id
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store initial B matrices for personalization
        self.personalized_B_matrices = None
        
    def train(self, dataloader: DataLoader, config: Dict) -> Dict:
        """
        Local training with LoRA adaptation
        
        Args:
            dataloader: Local training data
            config: Training configuration
        
        Returns:
            Dictionary containing training metrics and parameters
        """
        # Training configuration
        num_epochs = config.get('local_epochs', 5)
        lr = config.get('learning_rate', 1e-3)
        weight_decay = config.get('weight_decay', 1e-4)
        
        # Initialize optimizer (only optimize LoRA parameters)
        optimizer = self._get_optimizer(lr, weight_decay)
        criterion = nn.CrossEntropyLoss()
        
        # Training metrics
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training loop
        self.model.train()
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0
            
            with tqdm(dataloader, desc=f"Client {self.client_id} Epoch {epoch+1}/{num_epochs}") as pbar:
                for batch_idx, (images, labels) in enumerate(pbar):
                    images, labels = images.to(self.device), labels.to(self.device)
                    
                    # Forward pass
                    outputs = self.model(images)
                    loss = criterion(outputs, labels)
                    
                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    
                    # Update metrics
                    epoch_loss += loss.item()
                    _, predicted = outputs.max(1)
                    epoch_total += labels.size(0)
                    epoch_correct += predicted.eq(labels).sum().item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': epoch_loss / (batch_idx + 1),
                        'acc': 100. * epoch_correct / epoch_total
                    })
            
            train_loss += epoch_loss
            train_correct += epoch_correct
            train_total += epoch_total
        
        # Calculate average metrics
        avg_loss = train_loss / (num_epochs * len(dataloader))
        avg_accuracy = 100. * train_correct / train_total
        
        # Get LoRA parameters to send to server (only A matrices)
        lora_A_params = self.model.get_lora_params(matrix_type='A')
        
        # Store B matrices locally for personalization
        self.personalized_B_matrices = self.model.get_lora_params(matrix_type='B')
        
        return {
            'client_id': self.client_id,
            'num_samples': len(dataloader.dataset),
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'lora_A_params': lora_A_params
        }
    
    def update_model(self, global_A_params: Dict):
        """
        Update model with global A matrices while keeping personalized B matrices
        
        Args:
            global_A_params: Global A matrices from server
        """
        # Update A matrices with global parameters
        self.model.set_lora_params(global_A_params, matrix_type='A')
        
        # Keep personalized B matrices
        if self.personalized_B_matrices is not None:
            self.model.set_lora_params(self.personalized_B_matrices, matrix_type='B')
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        """
        Evaluate model on local data
        
        Args:
            dataloader: Evaluation data
        
        Returns:
            Dictionary containing evaluation metrics
        """
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        return {
            'client_id': self.client_id,
            'loss': test_loss / len(dataloader),
            'accuracy': 100. * test_correct / test_total,
            'num_samples': test_total
        }
    
    def _get_optimizer(self, lr: float, weight_decay: float) -> optim.Optimizer:
        """
        Get optimizer for LoRA parameters only
        
        Args:
            lr: Learning rate
            weight_decay: Weight decay
        
        Returns:
            Optimizer instance
        """
        lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params.append(param)
        
        return optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    
    def get_model_size(self) -> Dict:
        """
        Calculate model size statistics
        
        Returns:
            Dictionary containing model size information
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        lora_A_params = sum(p.numel() for name, p in self.model.named_parameters() 
                           if 'lora_A' in name)
        lora_B_params = sum(p.numel() for name, p in self.model.named_parameters() 
                           if 'lora_B' in name)
        
        return {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'lora_A_params': lora_A_params,
            'lora_B_params': lora_B_params,
            'communication_params': lora_A_params,  # Only A matrices are communicated
            'compression_ratio': total_params / lora_A_params if lora_A_params > 0 else 0
        }