#!/usr/bin/env python3
"""
Parallel hyperparameter sweep script for FedSA-LoRA experiments
Runs multiple experiments in parallel using available GPUs
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
import multiprocessing
from queue import Queue
import threading
import argparse

# Hyperparameter grid
ALPHA_VALUES = [0.1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
LR_VALUES = [0.001, 0.0001, 0.00001]
DROPOUT_VALUES = [0.1, 0.2, 0.3]

# Base configuration file
BASE_CONFIG = "configs/experiment_configs_non_iid/non-IID-FedSA-LoRA.yaml"

# Output directory for sweep results
SWEEP_DIR = "experiments/hyperparameter_sweep_parallel"

# Number of parallel jobs (set based on available GPUs)
# Set to None for automatic detection, or specify a number (e.g., 2, 3, 4)
NUM_PARALLEL_JOBS = None  # Auto-detect GPUs (or set manually like 2, 3, 4)


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_config(config, config_path):
    """Save YAML configuration file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def create_experiment_config(base_config, alpha, lr, dropout, output_dir):
    """Create a modified configuration for a specific hyperparameter combination"""
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


def run_experiment_worker(exp_info, gpu_id, result_queue):
    """
    Worker function to run a single experiment on a specific GPU
    
    Args:
        exp_info: Dictionary containing experiment information
        gpu_id: GPU device ID to use
        result_queue: Queue to put results
    """
    idx = exp_info['idx']
    alpha = exp_info['alpha']
    lr = exp_info['lr']
    dropout = exp_info['dropout']
    config_path = exp_info['config_path']
    log_file = exp_info['log_file']
    
    print(f"[GPU {gpu_id}] Starting experiment {idx}: alpha={alpha}, lr={lr}, dropout={dropout}")
    
    # Set GPU environment variable - THIS IS CRITICAL FOR GPU ASSIGNMENT
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    # Add GPU device argument to the command and force fresh Python process
    cmd = ["python", "-u", "quickstart_resnet.py", "--config", str(config_path), "--gpu_id", str(gpu_id)]
    
    exp_start_time = time.time()
    try:
        with open(log_file, 'w') as f:
            # Write GPU assignment info to log
            f.write(f"=== GPU Assignment Info ===\n")
            f.write(f"GPU ID: {gpu_id}\n")
            f.write(f"CUDA_VISIBLE_DEVICES: {gpu_id}\n")
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"=== Experiment Output ===\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=os.getcwd()  # Ensure correct working directory
            )
            process.wait()
            return_code = process.returncode
        
        exp_duration = time.time() - exp_start_time
        
        if return_code == 0:
            status = "SUCCESS"
            print(f"[GPU {gpu_id}] ✅ Experiment {idx} completed in {exp_duration:.1f}s")
        else:
            status = "FAILED"
            print(f"[GPU {gpu_id}] ❌ Experiment {idx} failed with return code {return_code}")
    except Exception as e:
        status = "ERROR"
        exp_duration = time.time() - exp_start_time
        print(f"[GPU {gpu_id}] ❌ Experiment {idx} error: {e}")
    
    result = {
        'idx': idx,
        'status': status,
        'duration': exp_duration,
        'gpu_id': gpu_id
    }
    result_queue.put(result)


class GPUPool:
    """Simple GPU pool manager"""
    
    def __init__(self, num_gpus):
        self.num_gpus = num_gpus
        self.available_gpus = Queue()
        for i in range(num_gpus):
            self.available_gpus.put(i)
    
    def acquire(self):
        """Acquire a GPU (blocking)"""
        return self.available_gpus.get()
    
    def release(self, gpu_id):
        """Release a GPU back to the pool"""
        self.available_gpus.put(gpu_id)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Sweep for FedSA-LoRA')
    parser.add_argument('--yes', '-y', action='store_true', 
                       help='Skip confirmation prompt (useful for nohup/background execution)')
    parser.add_argument('--rounds', type=int, default=None,
                       help='Override number of rounds per experiment')
    args = parser.parse_args()
    
    print("=" * 80)
    print("FedSA-LoRA Parallel Hyperparameter Sweep")
    print("=" * 80)
    
    # Check for GPUs
    try:
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        gpu_lines = result.stdout.strip().split('\n')
        available_gpus = len([line for line in gpu_lines if line.startswith('GPU')])
        print(f"Detected {available_gpus} GPUs")
        
        # Print GPU information
        for line in gpu_lines:
            if line.startswith('GPU'):
                print(f"  {line}")
    except:
        available_gpus = 1
        print("Could not detect GPUs, assuming 1 GPU")
    
    # Adjust parallel jobs based on available GPUs
    if NUM_PARALLEL_JOBS is None:
        # Auto-detect: use all available GPUs
        num_parallel = available_gpus
        print(f"Auto-detected: Running {num_parallel} parallel jobs (using all GPUs)")
    else:
        # Manual setting: use specified number, but cap at available GPUs
        num_parallel = min(NUM_PARALLEL_JOBS, available_gpus)
        if NUM_PARALLEL_JOBS > available_gpus:
            print(f"Warning: Requested {NUM_PARALLEL_JOBS} parallel jobs, but only {available_gpus} GPUs available")
        print(f"Running {num_parallel} parallel jobs (manually specified)")
    
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
    
    # Create sweep summary
    sweep_summary = {
        'start_time': datetime.now().isoformat(),
        'base_config': BASE_CONFIG,
        'num_parallel_jobs': num_parallel,
        'hyperparameters': {
            'alpha': ALPHA_VALUES,
            'lr': LR_VALUES,
            'dropout': DROPOUT_VALUES
        },
        'total_experiments': total_experiments,
        'experiments': []
    }
    
    # Confirm before starting (skip if --yes flag is provided)
    if not args.yes:
        try:
            response = input(f"\nReady to run {total_experiments} experiments with {num_parallel} parallel jobs. Continue? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Sweep cancelled.")
                sys.exit(0)
        except EOFError:
            # No input available (e.g., running in nohup), auto-confirm
            print("\nNo input available (nohup mode detected), auto-confirming...")
            print("Starting sweep automatically...")
    else:
        print("\n--yes flag provided, skipping confirmation...")
        print("Starting sweep automatically...")
    
    # Prepare all experiment configurations
    experiment_queue = []
    for idx, (alpha, lr, dropout) in enumerate(combinations, 1):
        exp_name = f"alpha{alpha}_lr{lr}_dropout{dropout}"
        exp_dir = sweep_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Create modified configuration
        exp_config = create_experiment_config(base_config, alpha, lr, dropout, exp_dir)
        
        # Save configuration
        config_path = exp_dir / "config.yaml"
        save_config(exp_config, config_path)
        
        # Create log file path
        log_file = exp_dir / "training.log"
        
        exp_info = {
            'idx': idx,
            'name': exp_name,
            'alpha': alpha,
            'lr': lr,
            'dropout': dropout,
            'config_path': config_path,
            'log_file': log_file,
            'output_dir': exp_dir
        }
        experiment_queue.append(exp_info)
    
    print(f"\nPrepared {len(experiment_queue)} experiment configurations")
    
    # Run experiments in parallel
    start_time = time.time()
    gpu_pool = GPUPool(num_parallel)
    result_queue = Queue()
    active_threads = []
    completed = 0
    successful = 0
    failed = 0
    
    def worker_wrapper(exp_info):
        """Wrapper to manage GPU allocation"""
        gpu_id = gpu_pool.acquire()
        try:
            run_experiment_worker(exp_info, gpu_id, result_queue)
        finally:
            gpu_pool.release(gpu_id)
    
    # Start experiments
    exp_iter = iter(experiment_queue)
    
    # Launch initial batch
    for _ in range(num_parallel):
        try:
            exp_info = next(exp_iter)
            thread = threading.Thread(target=worker_wrapper, args=(exp_info,))
            thread.start()
            active_threads.append((thread, exp_info))
        except StopIteration:
            break
    
    # Monitor and launch new experiments as old ones complete
    while active_threads or completed < total_experiments:
        # Check for completed threads
        time.sleep(1)
        
        new_active = []
        for thread, exp_info in active_threads:
            if thread.is_alive():
                new_active.append((thread, exp_info))
            else:
                thread.join()
                
                # Get result
                try:
                    result = result_queue.get_nowait()
                    completed += 1
                    
                    if result['status'] == 'SUCCESS':
                        successful += 1
                    else:
                        failed += 1
                    
                    # Update summary
                    exp_result = {
                        'experiment_id': result['idx'],
                        'name': exp_info['name'],
                        'hyperparameters': {
                            'alpha': exp_info['alpha'],
                            'lr': exp_info['lr'],
                            'dropout': exp_info['dropout']
                        },
                        'config_path': str(exp_info['config_path']),
                        'output_dir': str(exp_info['output_dir']),
                        'log_file': str(exp_info['log_file']),
                        'status': result['status'],
                        'duration_seconds': result['duration'],
                        'gpu_id': result['gpu_id'],
                        'timestamp': datetime.now().isoformat()
                    }
                    sweep_summary['experiments'].append(exp_result)
                    
                    # Save intermediate summary
                    summary_file = sweep_dir / "sweep_summary.json"
                    with open(summary_file, 'w') as f:
                        json.dump(sweep_summary, f, indent=2)
                    
                    # Print progress
                    elapsed = time.time() - start_time
                    avg_time = elapsed / completed if completed > 0 else 0
                    remaining = (total_experiments - completed) * avg_time / num_parallel
                    
                    print(f"\nProgress: {completed}/{total_experiments} ({100*completed/total_experiments:.1f}%)")
                    print(f"  Successful: {successful}, Failed: {failed}")
                    print(f"  Elapsed: {elapsed/3600:.1f}h, Estimated remaining: {remaining/3600:.1f}h")
                    
                    # Launch next experiment
                    try:
                        next_exp = next(exp_iter)
                        new_thread = threading.Thread(target=worker_wrapper, args=(next_exp,))
                        new_thread.start()
                        new_active.append((new_thread, next_exp))
                    except StopIteration:
                        pass
                        
                except:
                    pass
        
        active_threads = new_active
    
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
    print("Parallel Hyperparameter Sweep Complete!")
    print("=" * 80)
    print(f"Total experiments: {total_experiments}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total duration: {total_duration/3600:.2f} hours")
    print(f"Average speedup: {total_experiments / (total_duration/3600 * num_parallel):.1f}x")
    print(f"Results saved to: {sweep_dir}")
    print(f"Summary: {summary_file}")
    print("=" * 80)
    
    # Create results CSV
    try:
        import pandas as pd
        
        results_data = []
        for exp in sweep_summary['experiments']:
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
                'gpu_id': exp.get('gpu_id', -1),
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
