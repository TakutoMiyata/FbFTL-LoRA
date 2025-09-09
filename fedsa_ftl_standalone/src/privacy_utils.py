"""
Differential Privacy utilities for FedSA-FTL
Implements privacy-preserving mechanisms using Opacus library
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple

from opacus import PrivacyEngine
import warnings

# OpacusのSecure RNGに関する警告を非表示にする
warnings.filterwarnings('ignore', message='Secure RNG turned off.*')
# Opacusのalphaに関する警告を非表示にする
warnings.filterwarnings('ignore', message='Optimal order is the largest alpha.*')
# PyTorchのbackward hookに関する警告を非表示にする
warnings.filterwarnings('ignore', message='Full backward hook is firing.*')

class DifferentialPrivacy:
    """
    Differential Privacy mechanism for FedSA-FTL using Opacus
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
            use_opacus: Whether to use Opacus (must be True)
        """
            
        self.epsilon = float(epsilon)
        self.delta = float(delta)
        self.max_grad_norm = float(max_grad_norm)
        self.total_rounds = total_rounds
        self.use_opacus = use_opacus  # Store use_opacus flag
        
        # Compute per-round epsilon budget
        self.epsilon_per_round = self.epsilon / self.total_rounds
        
        # Compute noise multiplier from per-round budget if not provided
        if noise_multiplier is None:
            self.noise_multiplier = np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon_per_round
        else:
            self.noise_multiplier = float(noise_multiplier)
        
        # Track privacy budget consumption
        self.steps = 0
        self.consumed_epsilon = 0.0
    
    def train_with_opacus(self, model, dataloader, optimizer, local_epochs: int) -> Tuple[Dict[str, torch.Tensor], int]:
        """
        Train using Opacus privacy engine for efficient DP-SGD
        
        This method uses Opacus to automatically handle per-sample gradient clipping
        and noise addition for differential privacy. The privacy engine tracks the
        privacy budget consumption throughout training.
        
        Args:
            model: Model instance
            dataloader: Training dataloader
            optimizer: Optimizer instance
            local_epochs: Number of local epochs
        
        Returns:
            Tuple of (updated LoRA A parameters, number of samples)
        """
        device = next(model.parameters()).device
        
        # To avoid hook conflicts, create a clean copy of the model for Opacus
        import copy
        
        # Create a deep copy to ensure no hook conflicts
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.to(device)
        
        # Ensure the copy has the exact same state
        model_copy.load_state_dict(model.state_dict())
        
        # Create new privacy engine for this training session
        privacy_engine = PrivacyEngine()
        
        # Create a new optimizer for the copied model
        import torch.optim as optim
        # Get learning rate and weight decay from the training config
        lr = 0.001  # Default learning rate for ViT
        weight_decay = 0.0001  # Default weight decay for ViT
        
        # Create fresh optimizer for the copied model
        trainable_params = [p for p in model_copy.parameters() if p.requires_grad]
        fresh_optimizer = optim.SGD(trainable_params, lr=lr, momentum=0.9, weight_decay=weight_decay)
        
        # Make model copy, fresh optimizer, dataloader private
        # Note: target_epsilon should be per-round budget, not multiplied by local_epochs
        # Opacus will automatically handle the composition across epochs
        private_model, private_optimizer, private_dataloader = privacy_engine.make_private_with_epsilon(
            module=model_copy,
            optimizer=fresh_optimizer,
            data_loader=dataloader,
            epochs=local_epochs,
            target_epsilon=self.epsilon_per_round,  # Per-round epsilon budget
            target_delta=self.delta,
            max_grad_norm=self.max_grad_norm,
        )
        
        # Train with Opacus
        private_model.train()
        criterion = torch.nn.CrossEntropyLoss().to(device)
        
        for epoch in range(local_epochs):
            for batch_idx, (images, labels) in enumerate(private_dataloader):
                images, labels = images.to(device), labels.to(device)
                
                private_optimizer.zero_grad()
                outputs = private_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                private_optimizer.step()
        
        # Extract updated LoRA A parameters
        lora_A_params = {}
        for name, param in private_model.named_parameters():
            if 'lora_A' in name and param.requires_grad:
                lora_A_params[name] = param.data.clone()
        
        # Get privacy spent
        # Note: epsilon_spent is the total privacy cost for all local_epochs
        epsilon_spent = privacy_engine.get_epsilon(self.delta)
        self.consumed_epsilon += epsilon_spent  # Don't divide by local_epochs
        self.steps += 1
        
        # Copy updated parameters back to original model
        # Only copy the LoRA parameters (A and B matrices) back to the original model
        with torch.no_grad():
            for (orig_name, orig_param), (priv_name, priv_param) in zip(
                model.named_parameters(), private_model.named_parameters()
            ):
                if orig_param.requires_grad and 'lora_' in orig_name:
                    # Only update LoRA parameters
                    orig_param.copy_(priv_param.data)
        
        return lora_A_params, len(dataloader.dataset)
    
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
    
    NOTE: This is a SIMULATION of secure aggregation for research/development purposes.
    It does not provide actual cryptographic security and should not be used in production.
    The masks simulate the effect of secure aggregation by canceling out during aggregation.
    """
    
    def __init__(self, num_clients: int, seed: int = 42, mask_scale: float = 0.01):
        """
        Initialize secure aggregation
        
        Args:
            num_clients: Number of clients
            seed: Random seed for reproducibility
            mask_scale: Scale factor for mask values (default: 0.01)
                       Adjust based on parameter scales in your model
        """
        self.num_clients = num_clients
        self.seed = seed
        self.mask_scale = mask_scale
        # Create a PyTorch random generator for reproducibility
        self.generator = torch.Generator().manual_seed(self.seed)
    
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
                # Generate random mask using the generator for reproducibility
                mask = torch.randn(param_shape, dtype=dtype, generator=self.generator) * self.mask_scale
                
                # Store positive mask for client i
                if i not in masks:
                    masks[i] = torch.zeros(param_shape, dtype=dtype)
                masks[i] += mask
                
                # Store negative mask for client j
                if j not in masks:
                    masks[j] = torch.zeros(param_shape, dtype=dtype)
                masks[j] -= mask
        
        return masks
    
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
    
    This factory function creates a DifferentialPrivacy instance based on the
    provided configuration. It ensures Opacus is available and properly configures
    the privacy parameters for federated learning.
    
    Args:
        config: Privacy configuration dictionary with the following keys:
            - enable_privacy: bool, whether to enable DP
            - epsilon: float, total privacy budget
            - delta: float, privacy parameter
            - max_grad_norm: float, gradient clipping bound
            - noise_multiplier: float (optional), custom noise scale
            - total_rounds: int, number of FL rounds for budget allocation
    
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
    
    return DifferentialPrivacy(
        epsilon=epsilon,
        delta=delta,
        max_grad_norm=max_grad_norm,
        noise_multiplier=noise_multiplier,
        total_rounds=total_rounds,
        use_opacus=True
    )