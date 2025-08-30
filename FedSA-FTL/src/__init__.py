"""
FedSA-FTL: Federated Share-A Transfer Learning

A hybrid architecture combining FbFTL and FedSA-LoRA principles for
personalized, communication-efficient federated transfer learning.
"""

from .fedsa_ftl_model import FedSAFTLModel, LoRALayer, FedSAFTLHead, create_vision_model, create_nlp_model
from .fedsa_ftl_client import FedSAFTLClient, FedSAFTLClientManager
from .fedsa_ftl_server import FedSAFTLServer
from .data_utils import (
    DirichletDataSplitter, 
    create_cifar10_dataloaders, 
    create_cifar100_dataloaders,
    get_dataset_info,
    analyze_data_distribution
)
from .experiment_controller import FedSAFTLExperiment, BaselineComparison

__version__ = "1.0.0"
__author__ = "FedSA-FTL Research Team"
__email__ = "contact@fedsa-ftl.ai"

__all__ = [
    # Core model components
    "FedSAFTLModel",
    "LoRALayer", 
    "FedSAFTLHead",
    "create_vision_model",
    "create_nlp_model",
    
    # Client-side components
    "FedSAFTLClient",
    "FedSAFTLClientManager",
    
    # Server-side components
    "FedSAFTLServer",
    
    # Data utilities
    "DirichletDataSplitter",
    "create_cifar10_dataloaders",
    "create_cifar100_dataloaders", 
    "get_dataset_info",
    "analyze_data_distribution",
    
    # Experiment management
    "FedSAFTLExperiment",
    "BaselineComparison"
]
