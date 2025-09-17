#!/usr/bin/env python3
"""
Differential Privacy utilities for FedSA-LoRA
Implements DP-SGD for A matrix parameters only
"""

import torch
import numpy as np
from typing import List, Dict, Optional


class DPOptimizer:
    """DP-SGD optimizer for A matrix parameters in FedSA-LoRA"""
    
    def __init__(self, 
                 A_params: List[torch.nn.Parameter],
                 B_params: List[torch.nn.Parameter],
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0001,
                 max_grad_norm: float = 0.5,
                 noise_multiplier: float = 1.0,
                 eps: float = 8.0,
                 delta: float = 1e-5):
        """
        Initialize DP optimizer
        
        Args:
            A_params: List of A matrix parameters (DP applied)
            B_params: List of B matrix parameters (no DP)
            lr: Learning rate
            momentum: Momentum for SGD
            weight_decay: Weight decay
            max_grad_norm: Maximum gradient norm for clipping
            noise_multiplier: Noise multiplier for DP
            eps: Privacy epsilon
            delta: Privacy delta
        """
        self.A_params = A_params
        self.B_params = B_params
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = noise_multiplier
        self.eps = eps
        self.delta = delta
        
        # Create separate optimizers for A and B parameters
        self.A_optimizer = torch.optim.SGD(
            A_params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.B_optimizer = torch.optim.SGD(
            B_params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        
        # Track privacy budget
        self.privacy_spent = 0.0
        self.steps = 0
        
    def zero_grad(self):
        """Zero gradients for both optimizers"""
        self.A_optimizer.zero_grad()
        self.B_optimizer.zero_grad()
    
    def step(self):
        """
        Perform DP-SGD step for A parameters and regular SGD for B parameters
        """
        # Step 1: Apply DP-SGD to A parameters
        self._dp_step_A_params()
        
        # Step 2: Regular SGD step for B parameters
        self.B_optimizer.step()
        
        self.steps += 1
    
    def _dp_step_A_params(self):
        """Apply DP-SGD to A matrix parameters"""
        # Step 1: Clip gradients per sample
        total_norm = 0.0
        
        # Calculate total gradient norm for A parameters
        for param in self.A_params:
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        
        # Step 2: Clip gradients if necessary
        if total_norm > self.max_grad_norm:
            clip_coef = self.max_grad_norm / (total_norm + 1e-6)
            for param in self.A_params:
                if param.grad is not None:
                    param.grad.data.mul_(clip_coef)
        
        # Step 3: Add noise to gradients
        if self.noise_multiplier > 0:
            for param in self.A_params:
                if param.grad is not None:
                    noise = torch.normal(
                        mean=0,
                        std=self.noise_multiplier * self.max_grad_norm,
                        size=param.grad.shape,
                        device=param.grad.device,
                        dtype=param.grad.dtype
                    )
                    param.grad.data.add_(noise)
        
        # Step 4: Apply optimizer step to A parameters
        self.A_optimizer.step()
    
    def get_privacy_spent(self) -> float:
        """
        Calculate privacy spent using RDP accountant (simplified)
        """
        if self.steps == 0:
            return 0.0
        
        # Simplified privacy accounting (use proper accountant in production)
        # This is a rough approximation
        q = 1.0  # Assuming full batch participation
        sigma = self.noise_multiplier
        steps = self.steps
        
        # Basic composition bound (not tight)
        privacy_spent = np.sqrt(2 * steps * np.log(1/self.delta)) / sigma
        return min(privacy_spent, self.eps)
    
    def state_dict(self):
        """Get state dict for both optimizers"""
        return {
            'A_optimizer': self.A_optimizer.state_dict(),
            'B_optimizer': self.B_optimizer.state_dict(),
            'steps': self.steps,
            'privacy_spent': self.get_privacy_spent()
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict for both optimizers"""
        self.A_optimizer.load_state_dict(state_dict['A_optimizer'])
        self.B_optimizer.load_state_dict(state_dict['B_optimizer'])
        self.steps = state_dict.get('steps', 0)


class WeightedFedAvg:
    """Weighted FedAvg aggregation for A matrices"""
    
    @staticmethod
    def aggregate_A_matrices(client_A_params: List[Dict[str, torch.Tensor]], 
                           client_sample_counts: List[int]) -> Dict[str, torch.Tensor]:
        """
        Aggregate A matrix parameters using weighted average
        
        Args:
            client_A_params: List of A parameter dicts from clients
            client_sample_counts: Number of samples per client
            
        Returns:
            Aggregated A parameters
        """
        if not client_A_params:
            raise ValueError("No client parameters provided")
        
        # Calculate weights
        total_samples = sum(client_sample_counts)
        weights = [count / total_samples for count in client_sample_counts]
        
        # Initialize aggregated parameters
        aggregated_A = {}
        
        # Get parameter names from first client
        param_names = client_A_params[0].keys()
        
        for param_name in param_names:
            # Weighted average for each parameter
            weighted_sum = None
            
            for client_idx, client_params in enumerate(client_A_params):
                if param_name in client_params:
                    param_tensor = client_params[param_name]
                    weight = weights[client_idx]
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor.clone()
                    else:
                        weighted_sum += weight * param_tensor
            
            if weighted_sum is not None:
                aggregated_A[param_name] = weighted_sum
        
        return aggregated_A
    
    @staticmethod
    def log_aggregation_info(client_sample_counts: List[int], 
                           num_A_params: int) -> None:
        """Log aggregation information"""
        total_samples = sum(client_sample_counts)
        weights = [count / total_samples for count in client_sample_counts]
        
        print("=== FedSA A-Matrix Aggregation ===")
        print(f"Total clients: {len(client_sample_counts)}")
        print(f"Total samples: {total_samples}")
        print(f"A parameters aggregated: {num_A_params}")
        print("Client weights:")
        for i, (count, weight) in enumerate(zip(client_sample_counts, weights)):
            print(f"  Client {i}: {count} samples (weight: {weight:.4f})")
        print("=== Only A uploaded (B kept local) ===")


def create_dp_optimizer(model, config: Dict) -> DPOptimizer:
    """
    Factory function to create DP optimizer for ResNet model
    
    Args:
        model: ResNet model with LoRA
        config: Configuration dictionary
        
    Returns:
        DPOptimizer instance
    """
    # Get parameter groups
    A_params = model.get_A_parameter_groups()
    B_params = model.get_B_parameter_groups()
    
    # Get training config
    training_config = config.get('training', {})
    privacy_config = config.get('privacy', {})
    
    return DPOptimizer(
        A_params=A_params,
        B_params=B_params,
        lr=training_config.get('lr', 0.001),
        momentum=training_config.get('momentum', 0.9),
        weight_decay=training_config.get('weight_decay', 0.0001),
        max_grad_norm=privacy_config.get('max_grad_norm', 0.5),
        noise_multiplier=privacy_config.get('noise_multiplier', 1.0),
        eps=privacy_config.get('epsilon', 8.0),
        delta=privacy_config.get('delta', 1e-5)
    )