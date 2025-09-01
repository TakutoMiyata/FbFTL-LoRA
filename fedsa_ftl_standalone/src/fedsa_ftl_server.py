"""
FedSA-FTL Server Implementation
Handles selective aggregation of A matrices only
"""

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np
import copy


class FedSAFTLServer:
    """
    FedSA-FTL Server for federated aggregation
    """
    
    def __init__(self, model: nn.Module, device: torch.device = None):
        """
        Initialize FedSA-FTL server
        
        Args:
            model: Global FedSAFTLModel instance
            device: Device to run aggregation on
        """
        self.global_model = model
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_model.to(self.device)
        
        # Training history
        self.history = {
            'round': [],
            'train_loss': [],
            'train_accuracy': [],
            'test_loss': [],
            'test_accuracy': [],
            'communication_cost': []
        }
        
        self.current_round = 0
    
    def aggregate_lora_A_matrices(self, client_updates: List[Dict], 
                                  aggregation_method: str = 'fedavg') -> Dict:
        """
        Aggregate LoRA A matrices from clients
        
        Args:
            client_updates: List of client updates containing A matrices
            aggregation_method: Aggregation method ('fedavg' or 'equal')
        
        Returns:
            Aggregated A matrices
        """
        if not client_updates:
            return {}
        
        # Calculate weights for aggregation
        if aggregation_method == 'fedavg':
            # Weighted average based on number of samples
            total_samples = sum(update['num_samples'] for update in client_updates)
            weights = [update['num_samples'] / total_samples for update in client_updates]
        else:
            # Equal weights
            weights = [1.0 / len(client_updates)] * len(client_updates)
        
        # Initialize aggregated parameters
        aggregated_A_params = {}
        
        # Get first client's A parameters as template
        first_A_params = client_updates[0]['lora_A_params']
        
        # Aggregate each A matrix
        for param_name in first_A_params.keys():
            # Initialize with zeros
            aggregated_param = torch.zeros_like(first_A_params[param_name])
            
            # Weighted sum of all clients' A matrices
            for client_update, weight in zip(client_updates, weights):
                if param_name in client_update['lora_A_params']:
                    aggregated_param += weight * client_update['lora_A_params'][param_name]
            
            aggregated_A_params[param_name] = aggregated_param
        
        return aggregated_A_params
    
    def update_global_model(self, aggregated_A_params: Dict):
        """
        Update global model with aggregated A matrices
        
        Args:
            aggregated_A_params: Aggregated A matrices
        """
        # Debug: Check if parameters are actually being updated
        if aggregated_A_params:
            before_params = self.global_model.get_lora_params(matrix_type='A')
            self.global_model.set_lora_params(aggregated_A_params, matrix_type='A')
            after_params = self.global_model.get_lora_params(matrix_type='A')
            
            # Verify update
            for key in aggregated_A_params.keys():
                if key in before_params and key in after_params:
                    diff = (after_params[key] - before_params[key]).abs().mean().item()
                    if diff < 1e-6:
                        print(f"WARNING: Parameter {key} was not updated!")
        else:
            self.global_model.set_lora_params(aggregated_A_params, matrix_type='A')
    
    def federated_round(self, client_updates: List[Dict], test_dataloader=None) -> Dict:
        """
        Execute one round of federated learning
        
        Args:
            client_updates: List of client updates
            test_dataloader: Optional test dataloader for evaluation
        
        Returns:
            Round statistics
        """
        self.current_round += 1
        
        # Calculate average training metrics
        avg_train_loss = np.mean([update['loss'] for update in client_updates])
        avg_train_accuracy = np.mean([update['accuracy'] for update in client_updates])
        
        # Aggregate A matrices
        aggregated_A_params = self.aggregate_lora_A_matrices(client_updates)
        
        # Update global model
        self.update_global_model(aggregated_A_params)
        
        # Calculate communication cost (only A matrices)
        communication_cost = sum(
            param.numel() * 4  # 4 bytes per float32
            for param in aggregated_A_params.values()
        )
        
        # Prepare round statistics
        round_stats = {
            'round': self.current_round,
            'num_clients': len(client_updates),
            'train_loss': avg_train_loss,
            'train_accuracy': avg_train_accuracy,
            'communication_cost_bytes': communication_cost,
            'communication_cost_mb': communication_cost / (1024 * 1024)
        }
        
        # Evaluate on test set if provided
        if test_dataloader is not None:
            test_metrics = self.evaluate(test_dataloader)
            round_stats.update({
                'test_loss': test_metrics['loss'],
                'test_accuracy': test_metrics['accuracy']
            })
        
        # Update history
        self.history['round'].append(self.current_round)
        self.history['train_loss'].append(avg_train_loss)
        self.history['train_accuracy'].append(avg_train_accuracy)
        if test_dataloader is not None:
            self.history['test_loss'].append(test_metrics['loss'])
            self.history['test_accuracy'].append(test_metrics['accuracy'])
        self.history['communication_cost'].append(communication_cost)
        
        return round_stats
    
    def evaluate(self, dataloader) -> Dict:
        """
        Evaluate global model
        
        Args:
            dataloader: Test dataloader
        
        Returns:
            Evaluation metrics
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()
        
        test_loss = 0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.global_model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = outputs.max(1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()
        
        return {
            'loss': test_loss / len(dataloader),
            'accuracy': 100. * test_correct / test_total
        }
    
    def get_global_A_params(self) -> Dict:
        """
        Get current global A matrices
        
        Returns:
            Global A matrices
        """
        return self.global_model.get_lora_params(matrix_type='A')
    
    def save_checkpoint(self, filepath: str):
        """
        Save server checkpoint
        
        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'round': self.current_round,
            'global_A_params': self.get_global_A_params(),
            'history': self.history,
            'model_config': self.global_model.lora_config
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        """
        Load server checkpoint
        
        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.current_round = checkpoint['round']
        self.update_global_model(checkpoint['global_A_params'])
        self.history = checkpoint['history']
        print(f"Checkpoint loaded from {filepath}")
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics of federated training
        
        Returns:
            Summary statistics
        """
        if not self.history['round']:
            return {}
        
        total_communication = sum(self.history['communication_cost'])
        
        return {
            'total_rounds': self.current_round,
            'best_train_accuracy': max(self.history['train_accuracy']) if self.history['train_accuracy'] else 0,
            'best_test_accuracy': max(self.history['test_accuracy']) if self.history['test_accuracy'] else 0,
            'final_train_accuracy': self.history['train_accuracy'][-1] if self.history['train_accuracy'] else 0,
            'final_test_accuracy': self.history['test_accuracy'][-1] if self.history['test_accuracy'] else 0,
            'total_communication_mb': total_communication / (1024 * 1024),
            'avg_communication_per_round_mb': (total_communication / self.current_round) / (1024 * 1024) if self.current_round > 0 else 0
        }