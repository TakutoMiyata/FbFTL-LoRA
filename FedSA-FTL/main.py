#!/usr/bin/env python3
"""
Main script for running FedSA-FTL experiments
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from experiment_controller import FedSAFTLExperiment


def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'fedsa_ftl_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )


def main():
    parser = argparse.ArgumentParser(description='Run FedSA-FTL experiments')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to configuration file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for results (auto-generated if not specified)')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting FedSA-FTL experiment")
    logger.info(f"Configuration file: {args.config}")
    
    try:
        # Load configuration
        with open(args.config, 'r') as f:
            config = json.load(f)
        
        logger.info(f"Loaded configuration for experiment: {config.get('experiment_name', 'Unnamed')}")
        
        # Create experiment
        experiment = FedSAFTLExperiment(config)
        
        # Setup experiment components
        logger.info("Setting up data...")
        experiment.setup_data()
        
        logger.info("Setting up model...")
        experiment.setup_model()
        
        logger.info("Setting up clients and server...")
        experiment.setup_clients_and_server()
        
        # Run experiment
        logger.info("Running experiment...")
        results = experiment.run_experiment()
        
        # Save results
        if args.output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            experiment_name = config.get('experiment_name', 'fedsa_ftl')
            args.output = f"./results/{experiment_name}_{timestamp}.json"
        
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        experiment.save_results(args.output)
        
        # Print summary
        final_metrics = results['final_metrics']
        communication_stats = results['communication_stats']
        time_stats = results['time_stats']
        
        print("\n" + "="*60)
        print("EXPERIMENT SUMMARY")
        print("="*60)
        print(f"Experiment: {config.get('experiment_name', 'Unnamed')}")
        print(f"Dataset: {config['dataset']['name']}")
        print(f"Model: {config['model']['type']}")
        print(f"Clients: {config['federated']['num_clients']}")
        print(f"Rounds: {config['federated']['num_rounds']}")
        print(f"Non-IID (alpha): {config['dataset']['alpha']}")
        print(f"LoRA rank: {config['model']['lora_rank']}")
        print("-"*60)
        print("RESULTS:")
        print(f"Final Test Accuracy: {final_metrics['test_accuracy']:.2f}%")
        print(f"Best Test Accuracy: {final_metrics['best_accuracy']:.2f}% (Round {final_metrics['best_round']})")
        print(f"Total Communication: {communication_stats['total_communication_mb']:.2f} MB")
        print(f"Avg Communication/Round: {communication_stats['avg_communication_per_round_mb']:.2f} MB")
        print(f"Total Training Time: {time_stats['total_time_minutes']:.1f} minutes")
        print(f"Communication Reduction: {communication_stats['communication_reduction_vs_full_lora']:.1%}")
        print("-"*60)
        print(f"Results saved to: {args.output}")
        print("="*60)
        
        logger.info("Experiment completed successfully")
        
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
