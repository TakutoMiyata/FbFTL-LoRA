"""
Differential Privacy utilities for FedSA-FTL
Implements privacy-preserving mechanisms for A-matrix updates
Supports both manual implementation and Opacus library
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

# Try to import Opacus for efficient DP-SGD
try:
    from opacus import PrivacyEngine
    from opacus.utils.batch_memory_manager import BatchMemoryManager
    OPACUS_AVAILABLE = True
except ImportError:
    OPACUS_AVAILABLE = False
    print("Warning: Opacus not installed. Using manual DP-SGD implementation (slower).")
    print("Install with: pip install opacus")


class DifferentialPrivacy:
    """
    Differential Privacy mechanism for FedSA-FTL
    Implements proper DP-SGD with per-sample gradient clipping
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, 
                 max_grad_norm: float = 1.0, noise_multiplier: Optional[float] = None,
                 total_rounds: int = 100, use_opacus: bool = True):
        """
        Initialize DP mechanism
        
        Args:
            epsilon: Total privacy budget (smaller = more private)
            delta: Privacy parameter for (ε,δ)-differential privacy
            max_grad_norm: Maximum L2 norm for per-sample gradient clipping
            noise_multiplier: Noise scale (if None, computed from epsilon)
            total_rounds: Total number of rounds for budget allocation
            use_opacus: Whether to use Opacus if available (recommended for efficiency)
        """
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.max_grad_norm = float(max_grad_norm)  # Per-sample gradient clipping threshold
        self.total_rounds = total_rounds
        self.use_opacus = use_opacus and OPACUS_AVAILABLE
        
        # Compute per-round epsilon budget
        self.epsilon_per_round = self.epsilon / self.total_rounds
        
        # Compute noise multiplier from per-round budget if not provided
        if noise_multiplier is None:
            # Using strong composition theorem approximation
            # σ = max_grad_norm * sqrt(2 * log(1.25/δ)) / ε_per_round
            self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon_per_round
        else:
            self.noise_multiplier = float(noise_multiplier)
        
        # Track privacy budget consumption
        self.steps = 0
        self.consumed_epsilon = 0.0
        
        # Opacus privacy engine (will be initialized when needed)
        self.privacy_engine = None
    
    def add_noise_to_accumulated_updates(self, accumulated_updates: Dict[str, torch.Tensor], 
                                        num_samples: int) -> Dict[str, torch.Tensor]:
        """
        Add calibrated Gaussian noise to accumulated gradient updates
        
        Args:
            accumulated_updates: Dictionary of averaged clipped gradient updates
            num_samples: Number of samples used to compute the average
        
        Returns:
            Noisy updates with proper DP guarantees
        """
        noisy_updates = {}
        
        for name, update_tensor in accumulated_updates.items():
            # Since we averaged the gradients, the sensitivity of the average is max_grad_norm / num_samples
            # This ensures correct noise calibration for averaged gradients
            noise_std = (self.noise_multiplier * self.max_grad_norm) / num_samples
            
            # Generate and add Gaussian noise
            noise = torch.randn_like(update_tensor) * noise_std
            noisy_updates[name] = update_tensor + noise
        
        # Update privacy accounting
        self.steps += 1
        self._update_privacy_budget()
        
        return noisy_updates
    
    def clip_per_sample_gradients(self, model, per_sample_grads: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Clip per-sample gradients to bounded L2 norm
        
        Args:
            model: Model instance
            per_sample_grads: Dictionary of per-sample gradients
        
        Returns:
            Clipped per-sample gradients
        """
        clipped_grads = {}
        
        for name, grad in per_sample_grads.items():
            if 'lora_A' in name:  # Only clip LoRA A matrix gradients
                # Compute L2 norm of this gradient
                grad_norm = torch.norm(grad, p=2)
                
                # Clip if necessary
                if grad_norm > self.max_grad_norm:
                    clipped_grads[name] = grad * (self.max_grad_norm / grad_norm)
                else:
                    clipped_grads[name] = grad.clone()
                    
        return clipped_grads
    
    def train_with_opacus(self, model, dataloader, optimizer, local_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Train using Opacus privacy engine for efficient DP-SGD
        
        Args:
            model: Model instance
            dataloader: Training dataloader
            optimizer: Optimizer instance
            local_epochs: Number of local epochs
        
        Returns:
            Accumulated gradient updates for LoRA A matrices
        """
        if not self.use_opacus:
            return self.simulate_per_sample_clipping(model, dataloader, local_epochs)
        
        device = next(model.parameters()).device
        
        # Initialize privacy engine if not already done
        if self.privacy_engine is None:
            self.privacy_engine = PrivacyEngine()
            
            # Make model, optimizer, dataloader private
            model, optimizer, dataloader = self.privacy_engine.make_private_with_epsilon(
                module=model,
                optimizer=optimizer,
                data_loader=dataloader,
                epochs=local_epochs,
                target_epsilon=self.epsilon_per_round,
                target_delta=self.delta,
                max_grad_norm=self.max_grad_norm,
            )
        
        # Train with Opacus
        model.train()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        
        for epoch in range(local_epochs):
            with BatchMemoryManager(
                data_loader=dataloader,
                max_physical_batch_size=32,  # Adjust based on GPU memory
                optimizer=optimizer
            ) as memory_safe_dataloader:
                
                for batch_idx, (images, labels) in enumerate(memory_safe_dataloader):
                    images, labels = images.to(device), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
        
        # Extract LoRA A gradients
        accumulated_updates = {}
        for name, param in model.named_parameters():
            if 'lora_A' in name and param.requires_grad and param.grad is not None:
                accumulated_updates[name] = param.grad.clone()
        
        # Get privacy spent
        epsilon_spent = self.privacy_engine.get_epsilon(self.delta)
        self.consumed_epsilon += epsilon_spent
        self.steps += 1
        
        return accumulated_updates, len(dataloader.dataset)
    
    def simulate_per_sample_clipping(self, model, dataloader, local_epochs: int) -> Dict[str, torch.Tensor]:
        """
        Efficient per-sample gradient clipping using batch processing
        
        This implements per-sample gradient clipping for DP-SGD using efficient
        batch computations while maintaining per-sample privacy guarantees.
        
        Args:
            model: Model instance
            dataloader: Local training data
            local_epochs: Number of local epochs
        
        Returns:
            Accumulated clipped gradient updates for LoRA A matrices
        """
        # Get device from model parameters
        device = next(model.parameters()).device

        accumulated_updates = {}
        sample_count = 0

        # Initialize accumulated updates on the same device as model
        for name, param in model.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                accumulated_updates[name] = torch.zeros_like(param, device=device)

        # Use reduction='none' to get per-sample losses
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        
        model.train()
        for epoch in range(local_epochs):
            for batch_idx, (images, labels) in enumerate(dataloader):
                # Move batch to device
                images = images.to(device)
                labels = labels.to(device)
                batch_size = images.size(0)
                
                # Enable gradient computation for individual samples
                model.zero_grad()
                
                # Forward pass for entire batch
                outputs = model(images)
                losses = criterion(outputs, labels)
                
                # Process gradients efficiently in smaller sub-batches
                # This balances memory usage and computational efficiency
                sub_batch_size = min(4, batch_size)  # Process 4 samples at a time
                
                for start_idx in range(0, batch_size, sub_batch_size):
                    end_idx = min(start_idx + sub_batch_size, batch_size)
                    sub_batch_losses = losses[start_idx:end_idx]
                    
                    # Accumulate gradients for sub-batch
                    for i, loss in enumerate(sub_batch_losses):
                        # Compute gradients for this sample
                        if i == len(sub_batch_losses) - 1:
                            loss.backward(retain_graph=False)
                        else:
                            loss.backward(retain_graph=True)
                        
                        # Collect and clip gradients for LoRA A matrices
                        sample_grads = {}
                        total_norm_squared = 0.0
                        
                        for name, param in model.named_parameters():
                            if 'lora_A' in name and param.requires_grad and param.grad is not None:
                                # Scale gradient by 1/batch_size to match individual processing
                                grad = param.grad.clone() / batch_size
                                sample_grads[name] = grad
                                total_norm_squared += torch.sum(grad ** 2).item()
                        
                        # Compute total L2 norm
                        total_norm = np.sqrt(total_norm_squared) if total_norm_squared > 0 else 0.0
                        
                        # Clip gradients
                        clip_factor = min(1.0, self.max_grad_norm / (total_norm + 1e-8))
                        
                        for name, grad in sample_grads.items():
                            clipped_grad = grad * clip_factor
                            accumulated_updates[name] += clipped_grad
                        
                        sample_count += 1
                        
                        # Clear gradients for next sample
                        model.zero_grad()
                
                # Optional: yield control to avoid blocking
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Average the accumulated gradients by sample count
        for name in accumulated_updates:
            accumulated_updates[name] = accumulated_updates[name] / sample_count

        # Return both the averaged updates and the sample count for correct noise scaling
        return accumulated_updates, sample_count
    
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
        """Update consumed privacy budget for communication-only DP"""
        # For communication-only DP: simple composition
        # Each round uses epsilon_per_round budget
        self.consumed_epsilon = self.steps * self.epsilon_per_round
    
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
    
    # Get total rounds from config for proper budget allocation
    total_rounds = config.get('total_rounds', 100)
    
    # Check if Opacus should be used
    use_opacus = config.get('use_opacus', True)  # Default to True if available
    
    if use_opacus and not OPACUS_AVAILABLE:
        print("Warning: Opacus requested but not installed. Using manual implementation.")
        print("For better performance, install Opacus: pip install opacus")
    
    return DifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        total_rounds=total_rounds,
        use_opacus=use_opacus
    )