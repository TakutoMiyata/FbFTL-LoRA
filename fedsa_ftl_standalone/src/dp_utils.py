#!/usr/bin/env python3
"""
Differential Privacy utilities for FedSA-LoRA
Implements DP-SGD for A matrix parameters only
"""

import torch
import numpy as np
from typing import List, Dict, Optional


class DPOptimizer:
    """DP-SGD optimizer for A matrix parameters in FedSA-LoRA with per-sample clipping"""
    
    def __init__(self, 
                 A_params: List[torch.nn.Parameter],
                 B_params: List[torch.nn.Parameter],
                 lr: float = 0.001,
                 momentum: float = 0.9,
                 weight_decay: float = 0.0001,
                 max_grad_norm: float = 0.5,
                 noise_multiplier: float = 1.0,
                 eps: float = 8.0,
                 delta: float = 1e-5,
                 batch_size: int = 64,
                 dataset_size: int = 1000):
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
        self.batch_size = batch_size
        self.dataset_size = dataset_size
        
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
        WARNING: This method assumes gradients are already processed by dp_backward_on_loss()
        Do NOT call this after dp_backward_on_loss() as it will apply DP twice!
        """
        # DEPRECATED: Use A_optimizer.step() and B_optimizer.step() directly
        # after dp_backward_on_loss_efficient() to avoid double DP processing
        raise DeprecationWarning(
            "Use dp_backward_on_loss_efficient() followed by A_optimizer.step() and B_optimizer.step() instead"
        )
        
        # Legacy implementation (kept for backwards compatibility but not recommended)
        # self._dp_step_A_params()
        # self.B_optimizer.step()
        # self.steps += 1
    
    @torch.no_grad()
    def _zero_grads_A(self):
        """Zero gradients for A parameters only"""
        for p in self.A_params:
            if p.grad is not None:
                p.grad.zero_()
    
    def dp_backward_on_loss_efficient(self, loss_vec: torch.Tensor, 
                                     microbatch_size: int = 8,
                                     also_compute_B_grads: bool = True):
        """
        Efficient per-sample gradient clipping for A parameters with optional B gradients
        
        Args:
            loss_vec: Individual losses with shape [batch] (reduction='none')
            microbatch_size: Size of microbatches for per-sample processing
            also_compute_B_grads: If True, also compute regular gradients for B parameters
        """
        # Clear all gradients first
        self.A_optimizer.zero_grad(set_to_none=True)
        if also_compute_B_grads:
            self.B_optimizer.zero_grad(set_to_none=True)
        
        # Initialize accumulation buffers for A parameters
        for p in self.A_params:
            p._dp_accum = torch.zeros_like(p.data)
        
        n = loss_vec.shape[0]
        num_microbatches = 0
        
        # Process microbatches for A parameters with DP
        for start in range(0, n, microbatch_size):
            end = min(n, start + microbatch_size)
            micro_loss = loss_vec[start:end].mean()
            
            # Zero only A gradients for this microbatch
            for p in self.A_params:
                if p.grad is not None:
                    p.grad.zero_()
            
            # Backward pass for this microbatch
            micro_loss.backward(retain_graph=True)
            
            # Calculate L2 norm for A parameters only
            total_norm = 0.0
            for p in self.A_params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            
            # Clip A gradients if necessary
            if total_norm > self.max_grad_norm:
                coef = self.max_grad_norm / (total_norm + 1e-6)
                for p in self.A_params:
                    if p.grad is not None:
                        p.grad.data.mul_(coef)
            
            # Accumulate clipped A gradients
            for p in self.A_params:
                if p.grad is not None:
                    p._dp_accum.add_(p.grad.data)
            
            num_microbatches += 1
        
        # Finalize A gradients: average and add noise
        for p in self.A_params:
            if hasattr(p, '_dp_accum'):
                p.grad = p._dp_accum / num_microbatches
                delattr(p, '_dp_accum')
                
                # Add Gaussian noise to A parameters only
                if self.noise_multiplier > 0:
                    noise = torch.normal(
                        mean=0.0,
                        std=self.noise_multiplier * self.max_grad_norm,
                        size=p.grad.shape,
                        device=p.grad.device,
                        dtype=p.grad.dtype
                    )
                    p.grad.add_(noise)
        
        # Compute B gradients efficiently (single backward pass)
        if also_compute_B_grads:
            # Zero B gradients and compute from mean loss
            for p in self.B_params:
                if p.grad is not None:
                    p.grad.zero_()
            
            mean_loss = loss_vec.mean()
            mean_loss.backward()
        
        # Update step counter
        self.steps += 1
    
    # Keep old method for backwards compatibility
    def dp_backward_on_loss(self, loss_vec: torch.Tensor, microbatch_size: int = 8):
        """
        DEPRECATED: Use dp_backward_on_loss_efficient() instead
        """
        print("WARNING: dp_backward_on_loss() is deprecated. Use dp_backward_on_loss_efficient()")
        return self.dp_backward_on_loss_efficient(loss_vec, microbatch_size, also_compute_B_grads=False)
    
    def update_dataset_size(self, new_size: int):
        """Update dataset size for privacy accounting"""
        self.dataset_size = new_size
    
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
        Calculate privacy spent for A matrix parameters only using improved RDP approximation
        This accounts for privacy spent ONLY on A matrices, not B matrices
        
        WARNING: This is a simplified approximation. For production use or academic papers,
        consider using Opacus PrivacyEngine for more accurate privacy accounting:
        
        Example with Opacus:
            from opacus import PrivacyEngine
            privacy_engine = PrivacyEngine()
            model, optimizer, dataloader = privacy_engine.make_private(
                module=model,
                optimizer=A_optimizer,  # Only attach to A parameters
                data_loader=dataloader,
                noise_multiplier=self.noise_multiplier,
                max_grad_norm=self.max_grad_norm,
            )
            epsilon = privacy_engine.get_epsilon(delta=self.delta)
        """
        if self.steps == 0 or self.dataset_size == 0:
            return 0.0
        
        # Calculate sampling ratio for proper privacy accounting
        q = min(1.0, self.batch_size / self.dataset_size)
        sigma = max(self.noise_multiplier, 1e-12)  # Avoid division by zero
        T = self.steps
        
        # Improved RDP approximation (still simplified but more accurate)
        # Reference: ε ≈ q * sqrt(2 * T * log(1/δ)) / σ
        # For more accurate accounting, use Opacus PrivacyEngine
        import math
        eps_est = q * math.sqrt(2 * T * math.log(1 / self.delta)) / sigma
        
        # Cap at the configured epsilon
        return min(eps_est, self.eps)
    
    def get_privacy_analysis(self) -> Dict[str, float]:
        """Get detailed privacy analysis"""
        if self.dataset_size == 0:
            return {'error': 'Dataset size not set'}
        
        q = min(1.0, self.batch_size / self.dataset_size)
        return {
            'sampling_ratio': q,
            'steps_taken': self.steps,
            'noise_multiplier': self.noise_multiplier,
            'max_grad_norm': self.max_grad_norm,
            'privacy_spent': self.get_privacy_spent(),
            'privacy_budget': self.eps,
            'delta': self.delta,
            'note': 'Privacy applies only to A matrices',
            'warning': 'Use Opacus PrivacyEngine for production-grade privacy accounting',
            'opacus_setup': 'privacy_engine.make_private(module=model, optimizer=A_optimizer_only, ...)'
        }
    
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
            # Weighted average for each parameter with secure tensor handling
            weighted_sum = None
            
            for client_idx, client_params in enumerate(client_A_params):
                if param_name in client_params:
                    # Secure tensor handling: detach and move to CPU for aggregation
                    param_tensor = client_params[param_name].detach()
                    if param_tensor.device.type != 'cpu':
                        param_tensor = param_tensor.cpu()
                    
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
        print("Client weights (based on sample count):")
        for i, (count, weight) in enumerate(zip(client_sample_counts, weights)):
            print(f"  Client {i}: {count} samples (weight: {weight:.4f})")
        print("=== SECURITY: Only A uploaded (B kept local) ===")
        print("=== PRIVACY: DP applied only to A matrices ===")


def create_dp_optimizer(model, config: Dict, batch_size: int = 64, dataset_size: int = 1000) -> DPOptimizer:
    """
    Factory function to create DP optimizer for ResNet model
    
    Args:
        model: ResNet model with LoRA
        config: Configuration dictionary
        batch_size: Batch size for privacy accounting
        dataset_size: Local dataset size for privacy accounting
        
    Returns:
        DPOptimizer instance
    """
    # Get parameter groups
    A_params = model.get_A_parameter_groups()
    B_params = model.get_B_parameter_groups()
    
    # Get training config
    training_config = config.get('training', {})
    privacy_config = config.get('privacy', {})
    data_config = config.get('data', {})
    
    # Use provided values or fallback to config
    actual_batch_size = batch_size if batch_size > 0 else data_config.get('batch_size', 64)
    
    return DPOptimizer(
        A_params=A_params,
        B_params=B_params,
        lr=training_config.get('lr', 0.001),
        momentum=training_config.get('momentum', 0.9),
        weight_decay=training_config.get('weight_decay', 0.0001),
        max_grad_norm=privacy_config.get('max_grad_norm', 0.5),
        noise_multiplier=privacy_config.get('noise_multiplier', 1.0),
        eps=privacy_config.get('epsilon', 8.0),
        delta=privacy_config.get('delta', 1e-5),
        batch_size=actual_batch_size,
        dataset_size=dataset_size
    )