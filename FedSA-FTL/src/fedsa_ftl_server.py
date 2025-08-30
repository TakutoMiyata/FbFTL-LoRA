"""
FedSA-FTL Server Implementation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import copy
import logging
import numpy as np
from collections import OrderedDict
from .fedsa_ftl_model import FedSAFTLModel

logger = logging.getLogger(__name__)


class FedSAFTLServer:
    """
    FedSA-FTL Server implementing selective aggregation for LoRA A parameters
    """
    
    def __init__(self, 
                 model: FedSAFTLModel,
                 test_loader: Optional[DataLoader] = None,
                 aggregation_method: str = 'fedavg',
                 device: str = 'cpu'):
        
        self.model = model.to(device)
        self.test_loader = test_loader
        self.aggregation_method = aggregation_method
        self.device = device
        
        # Global LoRA A parameters
        self.global_lora_A = self.model.get_lora_A_parameters()
        
        # Server statistics
        self.round_stats = {
            'rounds': [],
            'test_accuracies': [],
            'test_losses': [],
            'aggregation_stats': [],
            'communication_stats': []
        }
        
        # Current round
        self.current_round = 0
        
    def aggregate_lora_A(self, 
                        client_lora_A_dict: Dict[int, Dict[str, torch.Tensor]],
                        client_weights: Optional[Dict[int, float]] = None) -> Dict[str, torch.Tensor]:
        """
        Aggregate LoRA A parameters from clients
        
        Args:
            client_lora_A_dict: Dictionary mapping client_id to LoRA A parameters
            client_weights: Dictionary mapping client_id to aggregation weights
            
        Returns:
            Aggregated global LoRA A parameters
        """
        if not client_lora_A_dict:
            logger.warning("No client LoRA A parameters received for aggregation")
            return self.global_lora_A
        
        # Set equal weights if not provided
        if client_weights is None:
            client_weights = {client_id: 1.0 for client_id in client_lora_A_dict.keys()}
            
        # Normalize weights
        total_weight = sum(client_weights.values())
        normalized_weights = {cid: w / total_weight for cid, w in client_weights.items()}
        
        if self.aggregation_method == 'fedavg':
            return self._fedavg_aggregation(client_lora_A_dict, normalized_weights)
        else:
            raise ValueError(f"Unsupported aggregation method: {self.aggregation_method}")
    
    def _fedavg_aggregation(self, 
                           client_lora_A_dict: Dict[int, Dict[str, torch.Tensor]],
                           client_weights: Dict[int, float]) -> Dict[str, torch.Tensor]:
        """
        FedAvg aggregation for LoRA A parameters
        
        Args:
            client_lora_A_dict: Client LoRA A parameters
            client_weights: Normalized client weights
            
        Returns:
            Aggregated LoRA A parameters
        """
        aggregated_params = {}
        
        # Get parameter names from first client
        param_names = list(next(iter(client_lora_A_dict.values())).keys())
        
        for param_name in param_names:
            # Weighted average of parameters
            weighted_sum = None
            
            for client_id, lora_A_params in client_lora_A_dict.items():
                if param_name in lora_A_params:
                    weight = client_weights[client_id]
                    param_tensor = lora_A_params[param_name].to(self.device)
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor
                    else:
                        weighted_sum += weight * param_tensor
            
            if weighted_sum is not None:
                aggregated_params[param_name] = weighted_sum
        
        return aggregated_params
    
    def update_global_model(self, aggregated_lora_A: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated LoRA A parameters
        
        Args:
            aggregated_lora_A: Aggregated LoRA A parameters
        """
        self.global_lora_A = aggregated_lora_A
        self.model.load_global_lora_A(aggregated_lora_A)
        logger.debug(f"Round {self.current_round}: Updated global LoRA A parameters")
    
    def evaluate(self, data_loader: Optional[DataLoader] = None, criterion: nn.Module = None) -> Dict[str, float]:
        """
        Evaluate the global model
        
        Args:
            data_loader: DataLoader for evaluation (uses self.test_loader if None)
            criterion: Loss function
            
        Returns:
            Dictionary containing evaluation metrics
        """
        if data_loader is None:
            data_loader = self.test_loader
            
        if data_loader is None:
            logger.warning("No test data available for server evaluation")
            return {'test_loss': 0.0, 'test_accuracy': 0.0, 'test_samples': 0}
            
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
        
        results = {
            'test_loss': avg_loss,
            'test_accuracy': accuracy,
            'test_samples': total_samples
        }
        
        logger.info(f"Round {self.current_round} - Global Test Accuracy: {accuracy:.2f}%, Loss: {avg_loss:.4f}")
        return results
    
    def federated_round(self, 
                       client_lora_A_dict: Dict[int, Dict[str, torch.Tensor]],
                       client_sample_counts: Optional[Dict[int, int]] = None) -> Dict[str, float]:
        """
        Execute one federated learning round
        
        Args:
            client_lora_A_dict: LoRA A parameters from clients
            client_sample_counts: Number of samples per client for weighted aggregation
            
        Returns:
            Round statistics
        """
        self.current_round += 1
        
        # Calculate aggregation weights based on sample counts
        client_weights = None
        if client_sample_counts is not None:
            client_weights = {cid: count for cid, count in client_sample_counts.items() 
                            if cid in client_lora_A_dict}
        
        # Aggregate LoRA A parameters
        aggregated_lora_A = self.aggregate_lora_A(client_lora_A_dict, client_weights)
        
        # Update global model
        self.update_global_model(aggregated_lora_A)
        
        # Evaluate global model
        eval_results = self.evaluate()
        
        # Calculate communication statistics
        comm_stats = self._calculate_communication_stats(client_lora_A_dict)
        
        # Store round statistics
        round_stats = {
            'round': self.current_round,
            'num_participants': len(client_lora_A_dict),
            **eval_results,
            **comm_stats
        }
        
        self._update_round_stats(round_stats)
        
        return round_stats
    
    def _calculate_communication_stats(self, client_lora_A_dict: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, float]:
        """
        Calculate communication statistics for the current round
        
        Args:
            client_lora_A_dict: LoRA A parameters from clients
            
        Returns:
            Communication statistics
        """
        if not client_lora_A_dict:
            return {}
        
        # Calculate parameter counts and sizes
        sample_params = next(iter(client_lora_A_dict.values()))
        lora_A_param_count = sum(p.numel() for p in sample_params.values())
        
        # Total communication for this round
        num_participants = len(client_lora_A_dict)
        total_uplink_params = lora_A_param_count * num_participants
        total_downlink_params = lora_A_param_count * num_participants  # Broadcast to all
        
        # Bytes (assuming float32)
        bytes_per_param = 4
        total_uplink_bytes = total_uplink_params * bytes_per_param
        total_downlink_bytes = total_downlink_params * bytes_per_param
        total_communication_bytes = total_uplink_bytes + total_downlink_bytes
        
        return {
            'lora_A_params_per_client': lora_A_param_count,
            'total_uplink_params': total_uplink_params,
            'total_downlink_params': total_downlink_params,
            'total_communication_params': total_uplink_params + total_downlink_params,
            'total_uplink_bytes': total_uplink_bytes,
            'total_downlink_bytes': total_downlink_bytes,
            'total_communication_bytes': total_communication_bytes,
            'communication_mb': total_communication_bytes / (1024 * 1024)
        }
    
    def _update_round_stats(self, round_stats: Dict[str, float]):
        """Update server statistics with round results"""
        self.round_stats['rounds'].append(round_stats['round'])
        self.round_stats['test_accuracies'].append(round_stats.get('test_accuracy', 0.0))
        self.round_stats['test_losses'].append(round_stats.get('test_loss', 0.0))
        
        # Store aggregation and communication stats
        agg_stats = {k: v for k, v in round_stats.items() 
                    if k.startswith(('num_participants', 'lora_A', 'total_'))}
        self.round_stats['aggregation_stats'].append(agg_stats)
    
    def get_global_lora_A(self) -> Dict[str, torch.Tensor]:
        """
        Get current global LoRA A parameters for distribution to clients
        
        Returns:
            Global LoRA A parameters
        """
        return copy.deepcopy(self.global_lora_A)
    
    def get_training_history(self) -> Dict[str, List]:
        """
        Get complete training history
        
        Returns:
            Dictionary containing training statistics over rounds
        """
        return copy.deepcopy(self.round_stats)
    
    def save_checkpoint(self, path: str):
        """Save server state"""
        checkpoint = {
            'current_round': self.current_round,
            'global_lora_A': self.global_lora_A,
            'model_state': self.model.state_dict(),
            'round_stats': self.round_stats,
            'aggregation_method': self.aggregation_method
        }
        torch.save(checkpoint, path)
        logger.info(f"Server checkpoint saved to {path}")
    
    def load_checkpoint(self, path: str) -> int:
        """Load server state and return current round"""
        checkpoint = torch.load(path, map_location=self.device)
        self.current_round = checkpoint['current_round']
        self.global_lora_A = checkpoint['global_lora_A']
        self.model.load_state_dict(checkpoint['model_state'])
        self.round_stats = checkpoint.get('round_stats', self.round_stats)
        self.aggregation_method = checkpoint.get('aggregation_method', self.aggregation_method)
        
        # Update model with loaded global LoRA A
        self.model.load_global_lora_A(self.global_lora_A)
        
        logger.info(f"Server checkpoint loaded from {path}, resuming from round {self.current_round}")
        return self.current_round
    
    def reset_round_stats(self):
        """Reset training statistics"""
        self.round_stats = {
            'rounds': [],
            'test_accuracies': [],
            'test_losses': [],
            'aggregation_stats': [],
            'communication_stats': []
        }
        self.current_round = 0
    
    def get_best_round_info(self) -> Dict[str, float]:
        """
        Get information about the best performing round
        
        Returns:
            Dictionary with best round statistics
        """
        if not self.round_stats['test_accuracies']:
            return {}
            
        best_accuracy = max(self.round_stats['test_accuracies'])
        best_round_idx = self.round_stats['test_accuracies'].index(best_accuracy)
        best_round = self.round_stats['rounds'][best_round_idx]
        best_loss = self.round_stats['test_losses'][best_round_idx]
        
        return {
            'best_round': best_round,
            'best_accuracy': best_accuracy,
            'best_loss': best_loss,
            'total_rounds': len(self.round_stats['rounds'])
        }
    
    def calculate_total_communication_cost(self) -> Dict[str, float]:
        """
        Calculate total communication cost across all rounds
        
        Returns:
            Dictionary with total communication statistics
        """
        if not self.round_stats['aggregation_stats']:
            return {}
        
        total_uplink_bytes = sum(stats.get('total_uplink_bytes', 0) 
                               for stats in self.round_stats['aggregation_stats'])
        total_downlink_bytes = sum(stats.get('total_downlink_bytes', 0) 
                                 for stats in self.round_stats['aggregation_stats'])
        total_communication_bytes = total_uplink_bytes + total_downlink_bytes
        
        return {
            'total_rounds': len(self.round_stats['aggregation_stats']),
            'total_uplink_mb': total_uplink_bytes / (1024 * 1024),
            'total_downlink_mb': total_downlink_bytes / (1024 * 1024),
            'total_communication_mb': total_communication_bytes / (1024 * 1024),
            'avg_communication_per_round_mb': (total_communication_bytes / len(self.round_stats['aggregation_stats'])) / (1024 * 1024)
        }
