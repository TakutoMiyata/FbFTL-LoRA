"""
FedSA-FTL Experiment Controller
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
import time
import json
import os
from datetime import datetime

from fedsa_ftl_model import FedSAFTLModel, create_vision_model, create_nlp_model
from fedsa_ftl_client import FedSAFTLClient, FedSAFTLClientManager
from fedsa_ftl_server import FedSAFTLServer
from data_utils import create_cifar10_dataloaders, create_cifar100_dataloaders, get_dataset_info, analyze_data_distribution

logger = logging.getLogger(__name__)


class FedSAFTLExperiment:
    """
    Controller for running FedSA-FTL experiments
    """
    
    def __init__(self, config: Dict):
        """
        Initialize experiment with configuration
        
        Args:
            config: Experiment configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Set random seeds for reproducibility
        self._set_random_seeds(config.get('random_seed', 42))
        
        # Initialize components
        self.model = None
        self.server = None
        self.client_manager = None
        self.train_loaders = None
        self.val_loader = None
        self.test_loader = None
        
        # Experiment results
        self.results = {
            'config': config,
            'training_history': {},
            'final_metrics': {},
            'communication_stats': {},
            'time_stats': {}
        }
        
    def _set_random_seeds(self, seed: int):
        """Set random seeds for reproducibility"""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def setup_data(self):
        """Setup federated data loaders"""
        dataset_name = self.config['dataset']['name']
        num_clients = self.config['federated']['num_clients']
        alpha = self.config['dataset'].get('alpha', 0.5)
        batch_size = self.config['training'].get('batch_size', 32)
        
        logger.info(f"Setting up {dataset_name} dataset for {num_clients} clients (alpha={alpha})")
        
        if dataset_name == 'cifar10':
            self.train_loaders, self.val_loader, self.test_loader = create_cifar10_dataloaders(
                data_dir=self.config['dataset'].get('data_dir', './data'),
                num_clients=num_clients,
                alpha=alpha,
                batch_size=batch_size
            )
        elif dataset_name == 'cifar100':
            self.train_loaders, self.val_loader, self.test_loader = create_cifar100_dataloaders(
                data_dir=self.config['dataset'].get('data_dir', './data'),
                num_clients=num_clients,
                alpha=alpha,
                batch_size=batch_size
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")
        
        # Analyze data distribution
        dataset_info = get_dataset_info(dataset_name)
        distribution_stats = analyze_data_distribution(self.train_loaders, dataset_info['num_classes'])
        
        logger.info(f"Data distribution analysis:")
        logger.info(f"  - Average samples per client: {distribution_stats['avg_samples_per_client']:.1f}")
        logger.info(f"  - Average classes per client: {distribution_stats['avg_classes_per_client']:.1f}")
        logger.info(f"  - Class distribution variance: {distribution_stats['class_distribution_variance']:.4f}")
        
        self.results['data_distribution'] = distribution_stats
    
    def setup_model(self):
        """Setup FedSA-FTL model"""
        dataset_name = self.config['dataset']['name']
        dataset_info = get_dataset_info(dataset_name)
        
        model_config = self.config['model']
        model_type = model_config.get('type', 'vit_base')
        lora_rank = model_config.get('lora_rank', 8)
        
        logger.info(f"Setting up {model_type} model for {dataset_name}")
        
        if dataset_name in ['cifar10', 'cifar100']:
            self.model = create_vision_model(
                model_name=model_type,
                num_classes=dataset_info['num_classes'],
                lora_rank=lora_rank
            )
        else:
            # NLP models
            self.model = create_nlp_model(
                model_name=model_type,
                num_classes=dataset_info['num_classes'],
                lora_rank=lora_rank
            )
        
        # Log model information
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"Model setup complete:")
        logger.info(f"  - Total parameters: {total_params:,}")
        logger.info(f"  - Trainable parameters: {trainable_params:,}")
        logger.info(f"  - Trainable ratio: {100 * trainable_params / total_params:.2f}%")
        
        self.results['model_info'] = {
            'total_params': total_params,
            'trainable_params': trainable_params,
            'trainable_ratio': trainable_params / total_params,
            'lora_rank': lora_rank
        }
    
    def setup_clients_and_server(self):
        """Setup federated clients and server"""
        if self.model is None:
            raise ValueError("Model must be setup before clients and server")
        
        if self.train_loaders is None:
            raise ValueError("Data must be setup before clients and server")
        
        # Create clients
        clients = []
        optimizer_config = self.config['training'].get('optimizer', {'lr': 0.005})
        
        for client_id, train_loader in enumerate(self.train_loaders):
            # Create model copy for each client
            client_model = create_vision_model(
                model_name=self.config['model'].get('type', 'vit_base'),
                num_classes=get_dataset_info(self.config['dataset']['name'])['num_classes'],
                lora_rank=self.config['model'].get('lora_rank', 8)
            ) if self.config['dataset']['name'] in ['cifar10', 'cifar100'] else create_nlp_model(
                model_name=self.config['model'].get('type', 'roberta_base'),
                num_classes=get_dataset_info(self.config['dataset']['name'])['num_classes'],
                lora_rank=self.config['model'].get('lora_rank', 8)
            )
            
            # Copy global model parameters
            client_model.load_state_dict(self.model.state_dict())
            
            client = FedSAFTLClient(
                client_id=client_id,
                model=client_model,
                train_loader=train_loader,
                val_loader=None,  # Individual validation not used in this setup
                optimizer_config=optimizer_config,
                device=self.device
            )
            clients.append(client)
        
        self.client_manager = FedSAFTLClientManager(clients)
        
        # Create server
        self.server = FedSAFTLServer(
            model=self.model,
            test_loader=self.test_loader,
            aggregation_method='fedavg',
            device=self.device
        )
        
        logger.info(f"Setup complete: {len(clients)} clients and 1 server")
    
    def run_experiment(self):
        """Run the complete FedSA-FTL experiment"""
        if any(x is None for x in [self.model, self.server, self.client_manager]):
            raise ValueError("Must setup model, clients, and server before running experiment")
        
        num_rounds = self.config['federated']['num_rounds']
        local_epochs = self.config['training'].get('local_epochs', 1)
        client_sampling_ratio = self.config['federated'].get('client_sampling_ratio', 1.0)
        eval_frequency = self.config['training'].get('eval_frequency', 10)
        
        logger.info(f"Starting FedSA-FTL experiment:")
        logger.info(f"  - Total rounds: {num_rounds}")
        logger.info(f"  - Local epochs: {local_epochs}")
        logger.info(f"  - Client sampling ratio: {client_sampling_ratio}")
        
        start_time = time.time()
        
        # Training loop
        for round_num in range(num_rounds):
            round_start_time = time.time()
            
            logger.info(f"\n--- Round {round_num + 1}/{num_rounds} ---")
            
            # Client training
            training_results = self.client_manager.train_all_clients(
                local_epochs=local_epochs,
                client_sampling_ratio=client_sampling_ratio
            )
            
            # Collect LoRA A parameters
            participating_clients = list(training_results.keys())
            client_lora_A = self.client_manager.collect_lora_A_from_clients(participating_clients)
            
            # Calculate client weights based on sample counts
            client_sample_counts = {cid: stats['num_samples'] 
                                  for cid, stats in training_results.items()}
            
            # Server aggregation
            round_stats = self.server.federated_round(client_lora_A, client_sample_counts)
            
            # Distribute updated global LoRA A
            global_lora_A = self.server.get_global_lora_A()
            self.client_manager.distribute_global_lora_A(global_lora_A, participating_clients)
            
            round_time = time.time() - round_start_time
            
            # Logging
            avg_client_acc = np.mean([stats['accuracy'] for stats in training_results.values()])
            logger.info(f"Round {round_num + 1} completed in {round_time:.2f}s:")
            logger.info(f"  - Avg client accuracy: {avg_client_acc:.2f}%")
            logger.info(f"  - Global test accuracy: {round_stats['test_accuracy']:.2f}%")
            logger.info(f"  - Communication: {round_stats['communication_mb']:.2f} MB")
            
            # Detailed evaluation
            if (round_num + 1) % eval_frequency == 0 or round_num == num_rounds - 1:
                client_eval_results = self.client_manager.evaluate_all_clients()
                avg_client_val_acc = np.mean([r.get('val_accuracy', 0) for r in client_eval_results.values()])
                logger.info(f"  - Detailed eval - Avg client val accuracy: {avg_client_val_acc:.2f}%")
        
        total_time = time.time() - start_time
        
        # Store training history and final results
        self.results['training_history'] = self.server.get_training_history()
        self.results['time_stats'] = {
            'total_time_seconds': total_time,
            'total_time_minutes': total_time / 60,
            'avg_time_per_round': total_time / num_rounds
        }
        
        # Final evaluation
        final_eval = self.server.evaluate()
        best_round_info = self.server.get_best_round_info()
        communication_cost = self.server.calculate_total_communication_cost()
        comm_stats = self.client_manager.get_aggregated_communication_stats()
        
        self.results['final_metrics'] = {
            **final_eval,
            **best_round_info
        }
        self.results['communication_stats'] = {
            **communication_cost,
            **comm_stats
        }
        
        logger.info(f"\n=== Experiment Complete ===")
        logger.info(f"Final test accuracy: {final_eval['test_accuracy']:.2f}%")
        logger.info(f"Best test accuracy: {best_round_info['best_accuracy']:.2f}% (Round {best_round_info['best_round']})")
        logger.info(f"Total communication: {communication_cost['total_communication_mb']:.2f} MB")
        logger.info(f"Total time: {total_time / 60:.2f} minutes")
        
        return self.results
    
    def save_results(self, save_path: str):
        """Save experiment results to file"""
        # Add timestamp and create directory if needed
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Convert tensors to serializable format
        serializable_results = self._make_serializable(self.results)
        
        with open(save_path, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        
        logger.info(f"Results saved to {save_path}")
    
    def _make_serializable(self, obj):
        """Convert tensors and other non-serializable objects to serializable format"""
        if isinstance(obj, torch.Tensor):
            return obj.detach().cpu().numpy().tolist()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(v) for v in obj]
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        else:
            return obj
    
    @classmethod
    def from_config_file(cls, config_path: str):
        """Create experiment from configuration file"""
        with open(config_path, 'r') as f:
            config = json.load(f)
        return cls(config)


class BaselineComparison:
    """
    Compare FedSA-FTL with baseline methods
    """
    
    def __init__(self, base_config: Dict):
        self.base_config = base_config
        self.results = {}
    
    def run_fedavg_full_comparison(self):
        """Run FedAvg with full model finetuning"""
        logger.info("Running FedAvg-Full baseline...")
        
        # Modify config for full model training
        config = self.base_config.copy()
        config['model']['full_model_training'] = True
        config['experiment_name'] = 'FedAvg-Full'
        
        # Run experiment (would need separate implementation for full model training)
        # This is a placeholder - actual implementation would require modifying
        # the model to allow full training
        pass
    
    def run_fedavg_lora_comparison(self):
        """Run FedAvg with LoRA (both A and B transmitted)"""
        logger.info("Running FedAvg-LoRA baseline...")
        
        # This would require modifying the server to aggregate both A and B
        # and clients to share both matrices
        pass
    
    def run_ffa_lora_comparison(self):
        """Run FFA-LoRA (freeze A, only train and communicate B)"""
        logger.info("Running FFA-LoRA baseline...")
        
        # This would require modifying the model to freeze A matrix
        # and only train/communicate B
        pass
    
    def generate_comparison_report(self) -> Dict:
        """Generate comparison report across all methods"""
        return {
            'methods': list(self.results.keys()),
            'comparison_metrics': {
                'final_accuracy': {method: results['final_metrics']['test_accuracy'] 
                                 for method, results in self.results.items()},
                'communication_cost': {method: results['communication_stats']['total_communication_mb']
                                     for method, results in self.results.items()},
                'training_time': {method: results['time_stats']['total_time_minutes']
                                for method, results in self.results.items()}
            }
        }
