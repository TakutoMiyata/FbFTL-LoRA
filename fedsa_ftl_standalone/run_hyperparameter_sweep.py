#!/usr/bin/env python3
"""
Hyperparameter sweep script for FedSA-LoRA experiments
Automatically runs experiments with different combinations of:
- data.alpha: [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
- training.lr: [0.001, 0.00001, 0.000001]
- lora.dropout: [0.1, 0.2, 0.3]
"""

import os
import sys
import yaml
import subprocess
import itertools
from pathlib import Path
from datetime import datetime
import json
import time

# Hyperparameter grid
ALPHA_VALUES = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LR_VALUES = [0.001, 0.00001, 0.000001]
DROPOUT_VALUES = [0.1, 0.2, 0.3]

# Base configuration file
BASE_CONFIG = "configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml"

# Output directory for sweep results
SWEEP_DIR = "experiments/hyperparameter_sweep"


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, config_path):
    """Save YAML configuration file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_experiment_config(base_config, alpha, lr, dropout, output_dir):
    """
    Create a modified configuration for a specific hyperparameter combination
    
    Args:
        base_config: Base configuration dictionary
        alpha: Data distribution alpha value
        lr: Learning rate
        dropout: LoRA dropout rate
        output_dir: Output directory for this experiment
    
    Returns:
        Modified configuration dictionary
    """
    config = base_config.copy()
    
    # Update hyperparameters
    config['data']['alpha'] = float(alpha)
    config['training']['lr'] = float(lr)
    config['model']['lora']['dropout'] = float(dropout)
    
    # Update experiment name and output directory
    exp_name = f"FedSA_alpha{alpha}_lr{lr}_dropout{dropout}"
    config['experiment']['name'] = exp_name
    config['experiment']['output_dir'] = str(output_dir)
    
    return config


def run_experiment(config_path, rounds=None, log_file=None):
    """
    Run a single experiment with the given configuration
    
    Args:
        config_path: Path to configuration file
        rounds: Number of rounds (override config)
        log_file: Path to log file for this experiment
    
    Returns:
        Return code from subprocess
    """
    cmd = ["python", "quickstart_resnet.py", "--config", config_path]
    
    if rounds:
        cmd.extend(["--rounds", str(rounds)])
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run experiment and capture output
    if log_file:
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Stream output to both console and log file
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            process.wait()
            return process.returncode
    else:
        result = subprocess.run(cmd, capture_output=False)
        return result.returncode


def main():
    print("=" * 80)
    print("FedSA-LoRA Hyperparameter Sweep")
    print("=" * 80)
    
    # Load base configuration
    if not Path(BASE_CONFIG).exists():
        print(f"Error: Base configuration file not found: {BASE_CONFIG}")
        sys.exit(1)
    
    base_config = load_config(BASE_CONFIG)
    print(f"Loaded base configuration from: {BASE_CONFIG}")
    
    # Create sweep directory
    sweep_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    sweep_dir = Path(SWEEP_DIR) / sweep_timestamp
    sweep_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sweep results will be saved to: {sweep_dir}")
    
    # Generate all combinations
    combinations = list(itertools.product(ALPHA_VALUES, LR_VALUES, DROPOUT_VALUES))
    total_experiments = len(combinations)
    
    print(f"\nHyperparameter grid:")
    print(f"  Alpha values: {ALPHA_VALUES}")
    print(f"  LR values: {LR_VALUES}")
    print(f"  Dropout values: {DROPOUT_VALUES}")
    print(f"  Total experiments: {total_experiments}")
    print("=" * 80)
    
    # Create sweep summary file
    sweep_summary = {
        'start_time': datetime.now().isoformat(),
        'base_config': BASE_CONFIG,
        'hyperparameters': {
            'alpha': ALPHA_VALUES,
            'lr': LR_VALUES,
            'dropout': DROPOUT_VALUES
        },
        'total_experiments': total_experiments,
        'experiments': []
    }
    
    # Confirm before starting
    response = input(f"\nReady to run {total_experiments} experiments. Continue? (yes/no): ")
    if response.lower() not in ['yes', 'y']:
        print("Sweep cancelled.")
        sys.exit(0)
    
    # Run experiments
    start_time = time.time()
    successful = 0
    failed = 0
    
    for idx, (alpha, lr, dropout) in enumerate(combinations, 1):
        print("\n" + "=" * 80)
        print(f"Experiment {idx}/{total_experiments}")
        print(f"  Alpha: {alpha}, LR: {lr}, Dropout: {dropout}")
        print("=" * 80)
        
        # Create experiment-specific directory
        exp_name = f"alpha{alpha}_lr{lr}_dropout{dropout}"
        exp_dir = sweep_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create modified configuration
        exp_config = create_experiment_config(base_config, alpha, lr, dropout, exp_dir)
        
        # Save configuration
        config_path = exp_dir / "config.yaml"
        save_config(exp_config, config_path)
        print(f"Configuration saved to: {config_path}")
        
        # Create log file
        log_file = exp_dir / "training.log"
        
        # Run experiment
        exp_start_time = time.time()
        try:
            return_code = run_experiment(str(config_path), log_file=str(log_file))
            exp_duration = time.time() - exp_start_time
            
            if return_code == 0:
                status = "SUCCESS"
                successful += 1
                print(f"✅ Experiment completed successfully in {exp_duration:.1f}s")
            else:
                status = "FAILED"
                failed += 1
                print(f"❌ Experiment failed with return code {return_code}")
        except Exception as e:
            status = "ERROR"
            failed += 1
            exp_duration = time.time() - exp_start_time
            print(f"❌ Experiment error: {e}")
        
        # Record experiment result
        exp_result = {
            'experiment_id': idx,
            'name': exp_name,
            'hyperparameters': {
                'alpha': alpha,
                'lr': lr,
                'dropout': dropout
            },
            'config_path': str(config_path),
            'output_dir': str(exp_dir),
            'log_file': str(log_file),
            'status': status,
            'duration_seconds': exp_duration,
            'timestamp': datetime.now().isoformat()
        }
        sweep_summary['experiments'].append(exp_result)
        
        # Save intermediate summary
        summary_file = sweep_dir / "sweep_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(sweep_summary, f, indent=2)
        
        # Print progress
        elapsed = time.time() - start_time
        avg_time = elapsed / idx
        remaining = (total_experiments - idx) * avg_time
        print(f"\nProgress: {idx}/{total_experiments} ({100*idx/total_experiments:.1f}%)")
        print(f"  Successful: {successful}, Failed: {failed}")
        print(f"  Elapsed: {elapsed/3600:.1f}h, Estimated remaining: {remaining/3600:.1f}h")
    
    # Final summary
    total_duration = time.time() - start_time
    sweep_summary['end_time'] = datetime.now().isoformat()
    sweep_summary['total_duration_seconds'] = total_duration
    sweep_summary['successful_experiments'] = successful
    sweep_summary['failed_experiments'] = failed
    
    # Save final summary
    summary_file = sweep_dir / "sweep_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(sweep_summary, f, indent=2)
    
    print("\n" + "=" * 80)
    print("Hyperparameter Sweep Complete!")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_file}")
    print("=" * 80)
    
    # Create results CSV for easy analysis
    try:
        import pandas as pd
        
        # Extract results from experiments
        results_data = []
        for exp in sweep_summary['experiments']:
            # Try to load final results if available
            exp_dir = Path(exp['output_dir'])
            results_file = list(exp_dir.glob('**/final_results_*.json'))
            
            if results_file and exp['status'] == 'SUCCESS':
                try:
                    with open(results_file[0], 'r') as f:
                        exp_results = json.load(f)
                        best_acc = exp_results.get('summary', {}).get('best_test_accuracy', None)
                        final_acc = exp_results.get('summary', {}).get('final_avg_accuracy', None)
                except:
                    best_acc = None
                    final_acc = None
            else:
                best_acc = None
                final_acc = None
            
            results_data.append({
                'experiment_id': exp['experiment_id'],
                'alpha': exp['hyperparameters']['alpha'],
                'lr': exp['hyperparameters']['lr'],
                'dropout': exp['hyperparameters']['dropout'],
                'status': exp['status'],
                'duration_hours': exp['duration_seconds'] / 3600,
                'best_test_accuracy': best_acc,
                'final_avg_accuracy': final_acc
            })
        
        df = pd.DataFrame(results_data)
        csv_file = sweep_dir / "results_summary.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nResults CSV saved to: {csv_file}")
        
        # Print top 10 results
        if successful > 0:
            df_success = df[df['status'] == 'SUCCESS'].copy()
            df_success = df_success.dropna(subset=['best_test_accuracy'])
            
            if len(df_success) > 0:
                df_success = df_success.sort_values('best_test_accuracy', ascending=False)
                print("\n🏆 Top 10 Best Results:")
                print(df_success.head(10)[['alpha', 'lr', 'dropout', 'best_test_accuracy']].to_string(index=False))
        
    except ImportError:
        print("\nNote: Install pandas for CSV export: pip install pandas")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSweep interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
