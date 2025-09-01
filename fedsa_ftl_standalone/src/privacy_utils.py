"""
Differential Privacy utilities for FedSA-FTL
Implements privacy-preserving mechanisms for A-matrix updates
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for FedSA-FTL
    Implements Gaussian mechanism for DP-SGD
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 max_grad_norm: float = 1.0, noise_multiplier: Optional[float] = None):
        """
        Initialize DP mechanism
        
        Args:
            epsilon: Privacy budget (smaller = more private)
            delta: Privacy parameter for (ε,δ)-differential privacy
            max_grad_norm: Maximum L2 norm for gradient clipping
            noise_multiplier: Noise scale (if None, computed from epsilon)
        """
        self.epsilon = float(epsilon)  # Ensure float type
        self.delta = float(delta)  # Ensure float type
        self.max_grad_norm = float(max_grad_norm)  # Ensure float type
        
        # Compute noise multiplier from privacy budget if not provided
        if noise_multiplier is None:
            # Approximation for Gaussian mechanism
            # σ ≈ sqrt(2 * log(1.25/δ)) / ε
            self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        else:
            self.noise_multiplier = float(noise_multiplier)
        
        # Track privacy budget consumption
        self.steps = 0
        self.consumed_epsilon = 0.0
    
    def add_noise_to_parameters(self, params: Dict[str, torch.Tensor], 
                                num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Add Gaussian noise to parameters for differential privacy
        
        Args:
            params: Dictionary of parameters (A matrices)
            num_samples: Number of samples in the dataset (for scaling)
        
        Returns:
            Noisy parameters
        """
        noisy_params = {}
        
        for name, param in params.items():
            # Compute sensitivity based on max_grad_norm and batch size
            sensitivity = self.max_grad_norm / num_samples
            
            # Add Gaussian noise
            noise_std = sensitivity * self.noise_multiplier
            noise = torch.randn_like(param) * noise_std
            
            noisy_params[name] = param + noise
        
        # Update privacy accounting
        self.steps += 1
        self._update_privacy_budget()
        
        return noisy_params
    
    def clip_parameters(self, params: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip parameters to bounded L2 norm
        
        Args:
            params: Dictionary of parameters to clip
        
        Returns:
            Clipped parameters
        """
        clipped_params = {}
        
        for name, param in params.items():
            # Compute L2 norm
            param_norm = torch.norm(param, p=2)
            
            # Clip if necessary
            if param_norm > self.max_grad_norm:
                clipped_params[name] = param * (self.max_grad_norm / param_norm)
            else:
                clipped_params[name] = param.clone()
        
        return clipped_params
    
    def apply_differential_privacy(self, params: Dict[str, torch.Tensor], 
                                   num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Apply full differential privacy mechanism: clip and add noise
        
        Args:
            params: Dictionary of parameters
            num_samples: Number of samples
        
        Returns:
            Private parameters
        """
        # First clip to bound sensitivity
        clipped_params = self.clip_parameters(params)
        
        # Then add noise
        private_params = self.add_noise_to_parameters(clipped_params, num_samples)
        
        return private_params
    
    def _update_privacy_budget(self):
        """Update consumed privacy budget using composition theorem"""
        # Using advanced composition for multiple steps
        # ε_total ≈ ε_step * sqrt(2 * steps * log(1/δ))
        self.consumed_epsilon = self.epsilon * np.sqrt(2 * self.steps * np.log(1 / self.delta))
    
    def get_privacy_spent(self) -> Tuple[float, float]:
        """
        Get total privacy budget spent
        
        Returns:
            (epsilon_spent, delta)
        """
        return self.consumed_epsilon, self.delta
    
    def reset_privacy_budget(self):
        """Reset privacy accounting"""
        self.steps = 0
        self.consumed_epsilon = 0.0


class SecureAggregation:
    """
    Secure Aggregation mechanism for additional privacy
    Implements additive masking for secure multi-party computation
    """
    
    def __init__(self, num_clients: int, seed: int = 42):
        """
        Initialize secure aggregation
        
        Args:
            num_clients: Number of clients
            seed: Random seed for reproducibility
        """
        self.num_clients = num_clients
        self.rng = np.random.RandomState(seed)
    
    def generate_masks(self, param_shape: Tuple, dtype: torch.dtype = torch.float32) -> Dict[int, torch.Tensor]:
        """
        Generate pairwise masks for secure aggregation
        
        Args:
            param_shape: Shape of parameter to mask
            dtype: Data type of masks
        
        Returns:
            Dictionary of masks for each client pair
        """
        masks = {}
        
        # Generate pairwise masks that sum to zero
        for i in range(self.num_clients):
            for j in range(i + 1, self.num_clients):
                # Generate random mask
                mask = torch.randn(param_shape, dtype=dtype) * 0.01
                
                # Store positive mask for client i
                if i not in masks:
                    masks[i] = torch.zeros(param_shape, dtype=dtype)
                masks[i] += mask
                
                # Store negative mask for client j
                if j not in masks:
                    masks[j] = torch.zeros(param_shape, dtype=dtype)
                masks[j] -= mask
        
        return masks
    
    def apply_masks(self, client_params: Dict[int, Dict[str, torch.Tensor]], 
                   masks: Dict[int, Dict[str, torch.Tensor]]) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Apply masks to client parameters
        
        Args:
            client_params: Dictionary of client parameters
            masks: Dictionary of masks for each client
        
        Returns:
            Masked parameters
        """
        masked_params = {}
        
        for client_id, params in client_params.items():
            masked_params[client_id] = {}
            for param_name, param in params.items():
                if client_id in masks and param_name in masks[client_id]:
                    masked_params[client_id][param_name] = param + masks[client_id][param_name]
                else:
                    masked_params[client_id][param_name] = param
        
        return masked_params
    
    def aggregate_with_secure_aggregation(self, client_updates: list) -> Dict[str, torch.Tensor]:
        """
        Perform secure aggregation with masking
        
        Args:
            client_updates: List of client parameter updates
        
        Returns:
            Securely aggregated parameters
        """
        # Generate masks for each parameter
        all_masks = {}
        for param_name in client_updates[0]['lora_A_params'].keys():
            param_shape = client_updates[0]['lora_A_params'][param_name].shape
            masks = self.generate_masks(param_shape)
            
            for client_id in range(len(client_updates)):
                if client_id not in all_masks:
                    all_masks[client_id] = {}
                if client_id in masks:
                    all_masks[client_id][param_name] = masks[client_id]
        
        # Apply masks to parameters
        masked_updates = []
        for i, update in enumerate(client_updates):
            masked_update = update.copy()
            if i in all_masks:
                for param_name in all_masks[i]:
                    if param_name in masked_update['lora_A_params']:
                        masked_update['lora_A_params'][param_name] = \
                            masked_update['lora_A_params'][param_name] + all_masks[i][param_name]
            masked_updates.append(masked_update)
        
        # Aggregate (masks cancel out due to pairwise structure)
        aggregated = {}
        first_params = masked_updates[0]['lora_A_params']
        
        for param_name in first_params.keys():
            aggregated[param_name] = torch.zeros_like(first_params[param_name])
            total_samples = sum(u['num_samples'] for u in masked_updates)
            
            for update in masked_updates:
                weight = update['num_samples'] / total_samples
                aggregated[param_name] += weight * update['lora_A_params'][param_name]
        
        return aggregated


def create_privacy_mechanism(config: Dict) -> Optional[DifferentialPrivacy]:
    """
    Create privacy mechanism from configuration
    
    Args:
        config: Privacy configuration dictionary
    
    Returns:
        DifferentialPrivacy instance or None if disabled
    """
    if not config.get('enable_privacy', False):
        return None
    
    # Ensure all parameters are properly typed
    epsilon = float(config.get('epsilon', 1.0))
    delta = float(config.get('delta', 1e-5))
    max_grad_norm = float(config.get('max_grad_norm', 1.0))
    noise_multiplier = config.get('noise_multiplier', None)
    if noise_multiplier is not None:
        noise_multiplier = float(noise_multiplier)
    
    return DifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier
    )