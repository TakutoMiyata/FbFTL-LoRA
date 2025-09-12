#!/usr/bin/env python3
"""
Visualization script for ViT federated learning results
Creates plots from saved training results for analysis and presentations.

Usage:
    python plot_results.py experiments/fedsa_ftl_cifar100_vit/20240912_123456/
"""

import argparse
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
from datetime import datetime


def load_results(results_dir):
    """Load training results from directory"""
    results_dir = Path(results_dir)
    
    # Try to find files with date+ViT pattern first
    final_results_files = list(results_dir.glob('final_results_*ViT.json'))
    if final_results_files:
        # Use the most recent file if multiple exist
        final_results_file = sorted(final_results_files)[-1]
        print(f"Loading results from: {final_results_file}")
        with open(final_results_file, 'r') as f:
            return json.load(f)
    
    # Try to find training results with date+ViT pattern
    training_results_files = list(results_dir.glob('training_results_*ViT.json'))
    if training_results_files:
        training_results_file = sorted(training_results_files)[-1]
        print(f"Loading results from: {training_results_file}")
        with open(training_results_file, 'r') as f:
            return json.load(f)
    
    # Fallback to old naming convention
    final_results_file = results_dir / 'final_results.json'
    if final_results_file.exists():
        print(f"Loading results from: {final_results_file}")
        with open(final_results_file, 'r') as f:
            return json.load(f)
    
    # Fallback to training results
    training_results_file = results_dir / 'training_results.json'
    if training_results_file.exists():
        print(f"Loading results from: {training_results_file}")
        with open(training_results_file, 'r') as f:
            return json.load(f)
    
    raise FileNotFoundError(f"No results files found in {results_dir}")
    # Available files: {list(results_dir.glob('*.json'))}")


def plot_accuracy_curves(results, save_dir):
    """Plot training and test accuracy curves"""
    rounds_data = results['rounds']
    
    # Extract data
    rounds = [r['round'] for r in rounds_data]
    train_accs = [r['avg_train_accuracy'] for r in rounds_data]
    test_accs = [r['avg_test_accuracy'] for r in rounds_data if r['avg_test_accuracy'] > 0]
    test_rounds = [r['round'] for r in rounds_data if r['avg_test_accuracy'] > 0]
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot accuracy curves
    plt.plot(rounds, train_accs, 'b-', label='Training Accuracy', linewidth=2, alpha=0.8)
    if test_accs:
        plt.plot(test_rounds, test_accs, 'r-', label='Test Accuracy', linewidth=2, alpha=0.8)
    
    # Mark best accuracy
    if test_accs:
        best_acc = max(test_accs)
        best_round = test_rounds[test_accs.index(best_acc)]
        plt.scatter([best_round], [best_acc], color='red', s=100, zorder=5, 
                   label=f'Best: {best_acc:.2f}% (Round {best_round})')
    
    plt.xlabel('Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title('ViT Federated Learning - Accuracy Over Time', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Generate filename with date+ViT suffix
    date_suffix = results.get('date_suffix', datetime.now().strftime('%m%d') + 'ViT')
    save_path = Path(save_dir) / f'accuracy_curves_{date_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Accuracy curves saved to: {save_path}")


def plot_communication_cost(results, save_dir):
    """Plot communication cost over rounds"""
    rounds_data = results['rounds']
    
    # Extract data
    rounds = [r['round'] for r in rounds_data]
    comm_costs = [r['communication_cost_mb'] for r in rounds_data]
    cumulative_costs = np.cumsum(comm_costs)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Per-round communication cost
    ax1.bar(rounds, comm_costs, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    ax1.set_xlabel('Round', fontsize=12)
    ax1.set_ylabel('Communication Cost (MB)', fontsize=12)
    ax1.set_title('Per-Round Communication Cost', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Cumulative communication cost
    ax2.plot(rounds, cumulative_costs, 'g-', linewidth=2, marker='o', markersize=4)
    ax2.set_xlabel('Round', fontsize=12)
    ax2.set_ylabel('Cumulative Cost (MB)', fontsize=12)
    ax2.set_title('Cumulative Communication Cost', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Add total cost annotation
    total_cost = cumulative_costs[-1] if cumulative_costs else 0
    ax2.annotate(f'Total: {total_cost:.2f} MB', 
                xy=(rounds[-1], total_cost), 
                xytext=(10, 10), 
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    
    # Generate filename with date+ViT suffix
    date_suffix = results.get('date_suffix', datetime.now().strftime('%m%d') + 'ViT')
    save_path = Path(save_dir) / f'communication_cost_{date_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Communication cost plots saved to: {save_path}")


def plot_client_performance(results, save_dir):
    """Plot individual client performance distribution"""
    rounds_data = results['rounds']
    
    # Collect client performance data
    client_train_data = []
    client_test_data = []
    
    for round_data in rounds_data:
        round_num = round_data['round']
        
        # Training accuracies
        for i, acc in enumerate(round_data['individual_train_accuracies']):
            client_train_data.append({
                'round': round_num,
                'client': f'Client {round_data["selected_clients"][i]}',
                'accuracy': acc,
                'type': 'Training'
            })
        
        # Test accuracies (if available)
        if round_data['individual_test_accuracies']:
            for i, acc in enumerate(round_data['individual_test_accuracies']):
                client_test_data.append({
                    'round': round_num,
                    'client': f'Client {round_data["selected_clients"][i]}',
                    'accuracy': acc,
                    'type': 'Test'
                })
    
    if not client_train_data:
        print("No client performance data available for plotting")
        return
    
    # Create DataFrame
    df_train = pd.DataFrame(client_train_data)
    
    # Create figure
    plt.figure(figsize=(15, 8))
    
    # Box plot of training accuracies by client
    if len(df_train) > 0:
        sns.boxplot(data=df_train, x='client', y='accuracy', palette='Set2')
        plt.title('Client Performance Distribution (Training Accuracy)', fontsize=14, fontweight='bold')
        plt.xlabel('Client', fontsize=12)
        plt.ylabel('Accuracy (%)', fontsize=12)
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Generate filename with date+ViT suffix
        date_suffix = results.get('date_suffix', datetime.now().strftime('%m%d') + 'ViT')
        save_path = Path(save_dir) / f'client_performance_{date_suffix}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Client performance plot saved to: {save_path}")


def plot_summary_dashboard(results, save_dir):
    """Create a comprehensive dashboard with multiple metrics"""
    rounds_data = results['rounds']
    config = results['config']
    summary = results.get('summary', {})
    
    # Extract data
    rounds = [r['round'] for r in rounds_data]
    train_accs = [r['avg_train_accuracy'] for r in rounds_data]
    test_accs = [r['avg_test_accuracy'] for r in rounds_data if r['avg_test_accuracy'] > 0]
    test_rounds = [r['round'] for r in rounds_data if r['avg_test_accuracy'] > 0]
    comm_costs = [r['communication_cost_mb'] for r in rounds_data]
    
    # Create dashboard
    fig = plt.figure(figsize=(20, 12))
    
    # Main accuracy plot (top)
    ax1 = plt.subplot(2, 3, (1, 2))
    ax1.plot(rounds, train_accs, 'b-', label='Training', linewidth=2, alpha=0.8)
    if test_accs:
        ax1.plot(test_rounds, test_accs, 'r-', label='Test', linewidth=2, alpha=0.8)
        best_acc = max(test_accs)
        best_round = test_rounds[test_accs.index(best_acc)]
        ax1.scatter([best_round], [best_acc], color='red', s=100, zorder=5)
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Learning Curves', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Configuration info (top right)
    ax2 = plt.subplot(2, 3, 3)
    ax2.axis('off')
    config_text = f"""Configuration:
    Model: {config['model']['model_name']}
    Dataset: {config['data']['dataset_name'].upper()}
    Clients: {config['federated']['num_clients']}
    Rounds: {config['federated']['num_rounds']}
    LoRA rank: {config['model']['lora_r']}
    Privacy: {'Yes' if config.get('privacy', {}).get('enable_privacy') else 'No'}
    
    Results:
    Best Accuracy: {summary.get('best_test_accuracy', 0):.2f}%
    Best Round: {summary.get('best_round', 'N/A')}
    Total Comm: {summary.get('total_communication_mb', 0):.1f} MB
    Duration: {summary.get('training_duration_hours', 0):.1f}h"""
    
    ax2.text(0.05, 0.95, config_text, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    # Communication cost (bottom left)
    ax3 = plt.subplot(2, 3, 4)
    ax3.bar(rounds, comm_costs, alpha=0.7, color='skyblue', edgecolor='navy', linewidth=0.5)
    ax3.set_xlabel('Round')
    ax3.set_ylabel('Cost (MB)')
    ax3.set_title('Communication Cost per Round', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Accuracy improvement (bottom middle)
    if len(test_accs) > 1:
        ax4 = plt.subplot(2, 3, 5)
        improvements = [test_accs[i] - test_accs[i-1] for i in range(1, len(test_accs))]
        colors = ['green' if x > 0 else 'red' for x in improvements]
        ax4.bar(test_rounds[1:], improvements, color=colors, alpha=0.7)
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Round')
        ax4.set_ylabel('Accuracy Change (%)')
        ax4.set_title('Round-to-Round Improvement', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
    
    # Performance summary (bottom right)
    if test_accs:
        ax5 = plt.subplot(2, 3, 6)
        ax5.hist(test_accs, bins=min(10, len(test_accs)//2), alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(np.mean(test_accs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(test_accs):.2f}%')
        ax5.set_xlabel('Test Accuracy (%)')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Accuracy Distribution', fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
    
    plt.suptitle(f'ViT Federated Learning Dashboard - {config["model"]["model_name"]}', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    
    # Generate filename with date+ViT suffix
    date_suffix = results.get('date_suffix', datetime.now().strftime('%m%d') + 'ViT')
    save_path = Path(save_dir) / f'training_dashboard_{date_suffix}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Training dashboard saved to: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot ViT federated learning results')
    parser.add_argument('results_dir', help='Directory containing training results')
    parser.add_argument('--output-dir', default=None, help='Output directory for plots (default: same as results_dir)')
    parser.add_argument('--plots', nargs='+', choices=['accuracy', 'communication', 'clients', 'dashboard', 'all'],
                       default=['all'], help='Which plots to generate')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    try:
        results = load_results(args.results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Set output directory
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.results_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set style
    plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
    
    plots_to_generate = args.plots if 'all' not in args.plots else ['accuracy', 'communication', 'clients', 'dashboard']
    
    print(f"Generating plots: {plots_to_generate}")
    print(f"Output directory: {output_dir}")
    
    # Generate plots
    if 'accuracy' in plots_to_generate:
        plot_accuracy_curves(results, output_dir)
    
    if 'communication' in plots_to_generate:
        plot_communication_cost(results, output_dir)
    
    if 'clients' in plots_to_generate:
        plot_client_performance(results, output_dir)
    
    if 'dashboard' in plots_to_generate:
        plot_summary_dashboard(results, output_dir)
    
    print(f"\n✅ All plots generated successfully!")
    print(f"📁 Plots saved to: {output_dir}")
    
    # Print summary
    summary = results.get('summary', {})
    if summary:
        print(f"\n📊 Training Summary:")
        print(f"  Best Accuracy: {summary.get('best_test_accuracy', 0):.2f}%")
        print(f"  Total Rounds: {summary.get('total_rounds', 0)}")
        print(f"  Training Time: {summary.get('training_duration_hours', 0):.2f} hours")
        print(f"  Model: {summary.get('model_name', 'N/A')}")


if __name__ == "__main__":
    main()
