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
from privacy_utils import DifferentialPrivacy


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
            privacy_mechanism: Optional differential privacy mechanism
        """
        self.client_id = client_id
        self.model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Store initial B matrices for personalization
        self.personalized_B_matrices = None
        
        # Privacy mechanism
        self.privacy_mechanism = None

    def set_privacy_mechanism(self, privacy_mechanism):
        """学習直前にDPメカニズムを設定する"""
        self.privacy_mechanism = privacy_mechanism
        
    def train(self, dataloader: DataLoader, config: Dict) -> Dict:
        """
        Local training with LoRA adaptation and proper differential privacy
        
        Args:
            dataloader: Local training data
            config: Training configuration
        
        Returns:
            Dictionary containing training metrics and parameters
        """
        # Training configuration (ensure numeric types)
        num_epochs = int(config.get('local_epochs', 5))
        lr = float(config.get('learning_rate', 1e-3))
        weight_decay = float(config.get('weight_decay', 1e-4))
        
        # Check if differential privacy is enabled
        if self.privacy_mechanism is not None:
            # Use differential privacy training with per-sample gradient clipping
            return self._train_with_dp(dataloader, config, num_epochs)
        else:
            # Standard training without privacy
            return self._train_standard(dataloader, config, num_epochs, lr, weight_decay)
    
    def _train_standard(self, dataloader: DataLoader, config: Dict, num_epochs: int, 
                       lr: float, weight_decay: float) -> Dict:
        """Standard training without differential privacy"""
        # Initialize optimizer (only optimize LoRA parameters)
        optimizer = self._get_optimizer(config, lr, weight_decay)
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
                    
                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(
                        [p for p in self.model.parameters() if p.requires_grad], 
                        max_norm=1.0
                    )
                    
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
        
        # Store the NEWLY TRAINED B matrices for personalization in the next round
        self.personalized_B_matrices = self.model.get_lora_params(matrix_type='B')
        
        return {
            'client_id': self.client_id,
            'num_samples': len(dataloader.dataset),
            'loss': avg_loss,
            'accuracy': avg_accuracy,
            'lora_A_params': lora_A_params
        }
    
    def _train_with_dp(self, dataloader: DataLoader, config: Dict, num_epochs: int) -> Dict:
        """Training with differential privacy using Opacus"""
        print(f"    Training with differential privacy (Opacus)")

        self.model.train()
        
        # Enable Opacus mode for ViT models to handle dropout compatibility
        if hasattr(self.model, 'set_opacus_mode'):
            self.model.set_opacus_mode(True)
        
        # Use Opacus for efficient DP-SGD
        lr = float(config.get('learning_rate', 1e-3))
        weight_decay = float(config.get('weight_decay', 1e-4))
        
        # Create optimizer for Opacus
        optimizer = self._get_optimizer(config, lr, weight_decay)
        
        # Train with Opacus - model is updated in-place
        lora_A_params, sample_count = self.privacy_mechanism.train_with_opacus(
            self.model, dataloader, optimizer, num_epochs
        )
        
        # Disable Opacus mode after training
        if hasattr(self.model, 'set_opacus_mode'):
            self.model.set_opacus_mode(False)
        
        # Calculate training metrics (approximate, since we didn't do normal forward passes)
        self.model.eval()
        criterion = nn.CrossEntropyLoss()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item() * labels.size(0)
                _, predicted = outputs.max(1)
                total_samples += labels.size(0)
                total_correct += predicted.eq(labels).sum().item()
        
        avg_loss = total_loss / total_samples
        avg_accuracy = 100. * total_correct / total_samples
        
        # Get updated LoRA A parameters (already extracted by Opacus training)
        # lora_A_params was returned from train_with_opacus
        
        # Store the NEWLY TRAINED B matrices for personalization in the next round
        self.personalized_B_matrices = self.model.get_lora_params(matrix_type='B')
        
        # Log privacy budget
        epsilon_spent, delta = self.privacy_mechanism.get_privacy_spent()
        print(f"    Privacy budget spent: ε={epsilon_spent:.2f}, δ={delta:.2e}")
        
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
                
                test_loss += loss.item() * labels.size(0)  # Weight by batch size
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        return {
            'client_id': self.client_id,
            'loss': test_loss / test_total,  # Average over samples, not batches
            'accuracy': 100. * test_correct / test_total,
            'num_samples': test_total
        }
    
    def _get_optimizer(self, config: Dict, lr: float, weight_decay: float) -> optim.Optimizer:
        """
        Get optimizer for LoRA parameters only
        
        Args:
            config: Training configuration
            lr: Learning rate
            weight_decay: Weight decay
        
        Returns:
            Optimizer instance
        """
        # Collect LoRA parameters
        lora_params = []
        for name, param in self.model.named_parameters():
            if 'lora_' in name and param.requires_grad:
                lora_params.append(param)
        
        # Get optimizer type from config
        optimizer_type = config.get('optimizer', 'sgd').lower()
        
        if optimizer_type == 'sgd':
            # SGD with momentum for better convergence with frozen features
            momentum = float(config.get('momentum', 0.9))
            return optim.SGD(lora_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_type == 'adamw':
            # AdamW optimizer - often works well for fine-tuning
            betas = tuple(config.get('betas', [0.9, 0.999]))
            eps = float(config.get('eps', 1e-8))
            return optim.AdamW(lora_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        elif optimizer_type == 'adam':
            # Standard Adam optimizer
            betas = tuple(config.get('betas', [0.9, 0.999]))
            eps = float(config.get('eps', 1e-8))
            return optim.Adam(lora_params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        else:
            # Default to SGD if unknown optimizer specified
            print(f"Warning: Unknown optimizer '{optimizer_type}', using SGD")
            return optim.SGD(lora_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    
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