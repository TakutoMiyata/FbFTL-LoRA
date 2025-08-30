"""
FedSA-FTL Client Implementation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, List, Tuple, Optional
import copy
import logging
from fedsa_ftl_model import FedSAFTLModel

logger = logging.getLogger(__name__)


class FedSAFTLClient:
    """
    FedSA-FTL Client implementing the selective aggregation protocol
    """
    
    def __init__(self, 
                 client_id: int,
                 model: FedSAFTLModel,
                 train_loader: DataLoader,
                 val_loader: Optional[DataLoader] = None,
                 optimizer_config: Dict = None,
                 device: str = 'cpu'):
        
        self.client_id = client_id
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer configuration
        self.optimizer_config = optimizer_config or {
            'lr': 0.005,
            'weight_decay': 1e-4
        }
        
        # Initialize optimizer for LoRA parameters only
        self.optimizer = self._create_optimizer()
        
        # Training statistics
        self.train_stats = {
            'losses': [],
            'accuracies': [],
            'rounds': []
        }
        
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer for trainable LoRA parameters"""
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        return optim.AdamW(
            trainable_params, 
            lr=self.optimizer_config['lr'],
            weight_decay=self.optimizer_config.get('weight_decay', 1e-4)
        )
    
    def local_train(self, 
                   local_epochs: int = 1, 
                   criterion: nn.Module = None) -> Dict[str, float]:
        """
        Perform local training on client data
        
        Args:
            local_epochs: Number of local training epochs
            criterion: Loss function
            
        Returns:
            Dictionary containing training statistics
        """
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.train()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        for epoch in range(local_epochs):
            epoch_loss = 0.0
            epoch_samples = 0
            epoch_correct = 0
            
            for batch_idx, (data, targets) in enumerate(self.train_loader):
                data, targets = data.to(self.device), targets.to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                # Statistics
                batch_size = data.size(0)
                epoch_loss += loss.item() * batch_size
                epoch_samples += batch_size
                
                _, predicted = outputs.max(1)
                epoch_correct += predicted.eq(targets).sum().item()
                
            total_loss += epoch_loss
            total_samples += epoch_samples
            correct_predictions += epoch_correct
            
            logger.debug(f"Client {self.client_id}, Epoch {epoch + 1}/{local_epochs}, "
                        f"Loss: {epoch_loss / epoch_samples:.4f}, "
                        f"Acc: {100. * epoch_correct / epoch_samples:.2f}%")
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
        
        training_stats = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'num_samples': total_samples
        }
        
        # Update training history
        self.train_stats['losses'].append(avg_loss)
        self.train_stats['accuracies'].append(accuracy)
        
        return training_stats
    
    def evaluate(self, data_loader: Optional[DataLoader] = None, criterion: nn.Module = None) -> Dict[str, float]:
        """
        Evaluate the model on validation/test data
        
        Args:
            data_loader: DataLoader for evaluation (uses self.val_loader if None)
            criterion: Loss function
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if data_loader is None:
            data_loader = self.val_loader
            
        if data_loader is None:
            logger.warning(f"Client {self.client_id}: No validation data available")
            return {'val_loss': 0.0, 'val_accuracy': 0.0, 'val_samples': 0}
            
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
            
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        correct_predictions = 0
        
        with torch.no_grad():
            for data, targets in data_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                
                batch_size = data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
                _, predicted = outputs.max(1)
                correct_predictions += predicted.eq(targets).sum().item()
        
        avg_loss = total_loss / total_samples if total_samples > 0 else 0.0
        accuracy = 100. * correct_predictions / total_samples if total_samples > 0 else 0.0
        
        return {
            'val_loss': avg_loss,
            'val_accuracy': accuracy,
            'val_samples': total_samples
        }
    
    def get_lora_A_for_aggregation(self) -> Dict[str, torch.Tensor]:
        """
        Get LoRA A parameters for server aggregation
        
        Returns:
            Dictionary containing LoRA A parameters
        """
        return self.model.get_lora_A_parameters()
    
    def get_lora_B_for_personalization(self) -> Dict[str, torch.Tensor]:
        """
        Get LoRA B parameters (client-specific, not shared)
        
        Returns:
            Dictionary containing LoRA B parameters
        """
        return self.model.get_lora_B_parameters()
    
    def update_global_lora_A(self, global_lora_A: Dict[str, torch.Tensor]):
        """
        Update local LoRA A with global aggregated version
        
        Args:
            global_lora_A: Global LoRA A parameters from server
        """
        self.model.load_global_lora_A(global_lora_A)
        logger.debug(f"Client {self.client_id}: Updated global LoRA A parameters")
    
    def get_model_size_info(self) -> Dict[str, int]:
        """
        Get information about model parameter sizes for communication analysis
        
        Returns:
            Dictionary with parameter counts and sizes
        """
        lora_A_params = self.get_lora_A_for_aggregation()
        lora_B_params = self.get_lora_B_for_personalization()
        
        # Count parameters
        lora_A_count = sum(p.numel() for p in lora_A_params.values())
        lora_B_count = sum(p.numel() for p in lora_B_params.values())
        total_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        
        # Calculate sizes in bytes (assuming float32)
        bytes_per_param = 4  # float32
        
        return {
            'lora_A_params': lora_A_count,
            'lora_B_params': lora_B_count,
            'total_trainable_params': total_trainable,
            'total_params': total_params,
            'lora_A_size_bytes': lora_A_count * bytes_per_param,
            'lora_B_size_bytes': lora_B_count * bytes_per_param,
            'communication_size_bytes': lora_A_count * bytes_per_param,  # Only A is communicated
            'communication_reduction_ratio': lora_A_count / (lora_A_count + lora_B_count)
        }
    
    def save_checkpoint(self, path: str, round_num: int = 0):
        """Save client state"""
        checkpoint = {
            'client_id': self.client_id,
            'round': round_num,
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'train_stats': self.train_stats
        }
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, path: str) -> int:
        """Load client state and return round number"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.train_stats = checkpoint.get('train_stats', self.train_stats)
        return checkpoint.get('round', 0)


class FedSAFTLClientManager:
    """
    Manager for multiple FedSA-FTL clients
    """
    
    def __init__(self, clients: List[FedSAFTLClient]):
        self.clients = {client.client_id: client for client in clients}
        
    def train_all_clients(self, 
                         local_epochs: int = 1,
                         client_sampling_ratio: float = 1.0,
                         criterion: nn.Module = None) -> Dict[int, Dict[str, float]]:
        """
        Train all clients (or a sample of them)
        
        Args:
            local_epochs: Number of local training epochs
            client_sampling_ratio: Fraction of clients to train
            criterion: Loss function
            
        Returns:
            Dictionary mapping client_id to training statistics
        """
        # Sample clients
        import random
        num_clients_to_sample = max(1, int(len(self.clients) * client_sampling_ratio))
        sampled_client_ids = random.sample(list(self.clients.keys()), num_clients_to_sample)
        
        training_results = {}
        
        for client_id in sampled_client_ids:
            client = self.clients[client_id]
            stats = client.local_train(local_epochs=local_epochs, criterion=criterion)
            training_results[client_id] = stats
            
        return training_results
    
    def collect_lora_A_from_clients(self, client_ids: List[int] = None) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Collect LoRA A parameters from specified clients
        
        Args:
            client_ids: List of client IDs (all clients if None)
            
        Returns:
            Dictionary mapping client_id to LoRA A parameters
        """
        if client_ids is None:
            client_ids = list(self.clients.keys())
            
        lora_A_collection = {}
        for client_id in client_ids:
            if client_id in self.clients:
                lora_A_collection[client_id] = self.clients[client_id].get_lora_A_for_aggregation()
                
        return lora_A_collection
    
    def distribute_global_lora_A(self, global_lora_A: Dict[str, torch.Tensor], client_ids: List[int] = None):
        """
        Distribute global LoRA A to specified clients
        
        Args:
            global_lora_A: Global LoRA A parameters
            client_ids: List of client IDs (all clients if None)
        """
        if client_ids is None:
            client_ids = list(self.clients.keys())
            
        for client_id in client_ids:
            if client_id in self.clients:
                self.clients[client_id].update_global_lora_A(global_lora_A)
    
    def evaluate_all_clients(self, client_ids: List[int] = None) -> Dict[int, Dict[str, float]]:
        """
        Evaluate all specified clients
        
        Args:
            client_ids: List of client IDs (all clients if None)
            
        Returns:
            Dictionary mapping client_id to evaluation metrics
        """
        if client_ids is None:
            client_ids = list(self.clients.keys())
            
        evaluation_results = {}
        for client_id in client_ids:
            if client_id in self.clients:
                results = self.clients[client_id].evaluate()
                evaluation_results[client_id] = results
                
        return evaluation_results
    
    def get_aggregated_communication_stats(self) -> Dict[str, float]:
        """
        Get aggregated communication statistics across all clients
        
        Returns:
            Dictionary with aggregated communication statistics
        """
        if not self.clients:
            return {}
            
        # Get stats from first client (should be same for all)
        sample_client = list(self.clients.values())[0]
        size_info = sample_client.get_model_size_info()
        
        # Scale by number of clients
        num_clients = len(self.clients)
        
        return {
            'num_clients': num_clients,
            'lora_A_params_per_client': size_info['lora_A_params'],
            'lora_B_params_per_client': size_info['lora_B_params'],
            'total_communication_params_per_round': size_info['lora_A_params'] * num_clients,
            'total_communication_bytes_per_round': size_info['communication_size_bytes'] * num_clients,
            'communication_reduction_vs_full_lora': size_info['communication_reduction_ratio'],
            'total_personalized_params': size_info['lora_B_params'] * num_clients
        }
