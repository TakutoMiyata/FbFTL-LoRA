"""
Analysis and visualization tools for FedSA-FTL results
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from typing import Dict, List, Optional
import os


class FedSAFTLAnalyzer:
    """
    Analyzer for FedSA-FTL experiment results
    """
    
    def __init__(self, results_path: str):
        """
        Initialize analyzer with results file
        
        Args:
            results_path: Path to experiment results JSON file
        """
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        self.config = self.results['config']
        self.training_history = self.results['training_history']
        self.final_metrics = self.results['final_metrics']
        self.communication_stats = self.results['communication_stats']
        
    def plot_training_curves(self, save_path: Optional[str] = None):
        """
        Plot training curves (accuracy and loss over rounds)
        """
        rounds = self.training_history['rounds']
        accuracies = self.training_history['test_accuracies']
        losses = self.training_history['test_losses']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Accuracy plot
        ax1.plot(rounds, accuracies, 'b-', linewidth=2, label='Test Accuracy')
        ax1.set_xlabel('Communication Round')
        ax1.set_ylabel('Test Accuracy (%)')
        ax1.set_title('Test Accuracy over Rounds')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Loss plot
        ax2.plot(rounds, losses, 'r-', linewidth=2, label='Test Loss')
        ax2.set_xlabel('Communication Round')
        ax2.set_ylabel('Test Loss')
        ax2.set_title('Test Loss over Rounds')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training curves saved to {save_path}")
        
        plt.show()
    
    def plot_communication_analysis(self, save_path: Optional[str] = None):
        """
        Plot communication efficiency analysis
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
        
        # Communication per round
        rounds = self.training_history['rounds']
        if 'aggregation_stats' in self.training_history:
            comm_per_round = [stats.get('communication_mb', 0) 
                            for stats in self.training_history['aggregation_stats']]
            
            ax1.plot(rounds, comm_per_round, 'g-', linewidth=2)
            ax1.set_xlabel('Round')
            ax1.set_ylabel('Communication (MB)')
            ax1.set_title('Communication per Round')
            ax1.grid(True, alpha=0.3)
        
        # Cumulative communication
        cumulative_comm = np.cumsum(comm_per_round) if 'aggregation_stats' in self.training_history else []
        if cumulative_comm:
            ax2.plot(rounds, cumulative_comm, 'orange', linewidth=2)
            ax2.set_xlabel('Round')
            ax2.set_ylabel('Cumulative Communication (MB)')
            ax2.set_title('Cumulative Communication Cost')
            ax2.grid(True, alpha=0.3)
        
        # Communication breakdown (pie chart)
        if 'lora_A_params_per_client' in self.communication_stats:
            lora_A_params = self.communication_stats['lora_A_params_per_client']
            lora_B_params = self.communication_stats['lora_B_params_per_client']
            
            labels = ['LoRA A (Communicated)', 'LoRA B (Local Only)']
            sizes = [lora_A_params, lora_B_params]
            colors = ['skyblue', 'lightcoral']
            
            ax3.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax3.set_title('Parameter Distribution')
        
        # Accuracy vs Communication efficiency
        if len(rounds) > 0 and len(cumulative_comm) > 0:
            accuracies = self.training_history['test_accuracies']
            ax4.plot(cumulative_comm, accuracies, 'purple', linewidth=2, marker='o', markersize=3)
            ax4.set_xlabel('Cumulative Communication (MB)')
            ax4.set_ylabel('Test Accuracy (%)')
            ax4.set_title('Accuracy vs Communication Trade-off')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Communication analysis saved to {save_path}")
        
        plt.show()
    
    def generate_summary_table(self) -> pd.DataFrame:
        """
        Generate summary table of key metrics
        """
        summary_data = {
            'Metric': [
                'Final Test Accuracy (%)',
                'Best Test Accuracy (%)',
                'Best Round',
                'Total Rounds',
                'Total Communication (MB)',
                'Avg Communication per Round (MB)',
                'Total Training Time (min)',
                'Communication Reduction vs Full LoRA (%)',
                'LoRA Rank',
                'Number of Clients',
                'Non-IID Level (α)',
                'Dataset'
            ],
            'Value': [
                f"{self.final_metrics.get('test_accuracy', 0):.2f}",
                f"{self.final_metrics.get('best_accuracy', 0):.2f}",
                f"{self.final_metrics.get('best_round', 0)}",
                f"{self.final_metrics.get('total_rounds', 0)}",
                f"{self.communication_stats.get('total_communication_mb', 0):.2f}",
                f"{self.communication_stats.get('avg_communication_per_round_mb', 0):.2f}",
                f"{self.results['time_stats'].get('total_time_minutes', 0):.1f}",
                f"{self.communication_stats.get('communication_reduction_vs_full_lora', 0)*100:.1f}",
                f"{self.config['model']['lora_rank']}",
                f"{self.config['federated']['num_clients']}",
                f"{self.config['dataset']['alpha']}",
                f"{self.config['dataset']['name'].upper()}"
            ]
        }
        
        return pd.DataFrame(summary_data)
    
    def analyze_convergence(self) -> Dict[str, float]:
        """
        Analyze convergence properties
        """
        accuracies = np.array(self.training_history['test_accuracies'])
        rounds = np.array(self.training_history['rounds'])
        
        if len(accuracies) == 0:
            return {}
        
        # Find convergence point (when accuracy stops improving significantly)
        improvement_threshold = 0.1  # 0.1% improvement
        convergence_round = len(accuracies)
        
        for i in range(10, len(accuracies)):  # Start checking after round 10
            recent_max = np.max(accuracies[max(0, i-10):i])
            current_max = np.max(accuracies[max(0, i-5):i+1])
            
            if current_max - recent_max < improvement_threshold:
                convergence_round = rounds[i]
                break
        
        # Calculate final convergence metrics
        final_10_rounds = accuracies[-10:] if len(accuracies) >= 10 else accuracies
        convergence_stability = np.std(final_10_rounds)
        
        return {
            'convergence_round': convergence_round,
            'convergence_accuracy': float(np.max(accuracies)),
            'final_stability': float(convergence_stability),
            'improvement_rate': float((accuracies[-1] - accuracies[0]) / len(accuracies)) if len(accuracies) > 1 else 0.0
        }
    
    def compare_with_baselines(self, baseline_results: List[str], save_path: Optional[str] = None):
        """
        Compare FedSA-FTL with baseline methods
        
        Args:
            baseline_results: List of paths to baseline result files
            save_path: Path to save comparison plot
        """
        # Load baseline results
        all_results = {'FedSA-FTL': self.results}
        
        for baseline_path in baseline_results:
            with open(baseline_path, 'r') as f:
                baseline_data = json.load(f)
                method_name = baseline_data['config'].get('experiment_name', os.path.basename(baseline_path))
                all_results[method_name] = baseline_data
        
        # Create comparison DataFrame
        comparison_data = []
        for method_name, results in all_results.items():
            comparison_data.append({
                'Method': method_name,
                'Final Accuracy (%)': results['final_metrics'].get('test_accuracy', 0),
                'Best Accuracy (%)': results['final_metrics'].get('best_accuracy', 0),
                'Total Communication (MB)': results['communication_stats'].get('total_communication_mb', 0),
                'Training Time (min)': results['time_stats'].get('total_time_minutes', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        x_pos = np.arange(len(df))
        ax1.bar(x_pos, df['Final Accuracy (%)'], color='skyblue', alpha=0.7)
        ax1.set_xlabel('Method')
        ax1.set_ylabel('Final Test Accuracy (%)')
        ax1.set_title('Final Test Accuracy Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(df['Method'], rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Communication comparison
        ax2.bar(x_pos, df['Total Communication (MB)'], color='lightcoral', alpha=0.7)
        ax2.set_xlabel('Method')
        ax2.set_ylabel('Total Communication (MB)')
        ax2.set_title('Communication Cost Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(df['Method'], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Training time comparison
        ax3.bar(x_pos, df['Training Time (min)'], color='lightgreen', alpha=0.7)
        ax3.set_xlabel('Method')
        ax3.set_ylabel('Training Time (min)')
        ax3.set_title('Training Time Comparison')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(df['Method'], rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # Efficiency scatter plot (accuracy vs communication)
        ax4.scatter(df['Total Communication (MB)'], df['Final Accuracy (%)'], 
                   s=100, alpha=0.7, c=range(len(df)), cmap='viridis')
        for i, method in enumerate(df['Method']):
            ax4.annotate(method, (df.iloc[i]['Total Communication (MB)'], 
                               df.iloc[i]['Final Accuracy (%)']), 
                        xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax4.set_xlabel('Total Communication (MB)')
        ax4.set_ylabel('Final Test Accuracy (%)')
        ax4.set_title('Accuracy vs Communication Efficiency')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Comparison plot saved to {save_path}")
        
        plt.show()
        
        return df
    
    def export_results_csv(self, save_path: str):
        """Export detailed results to CSV"""
        # Create detailed results DataFrame
        rounds = self.training_history['rounds']
        data = {
            'Round': rounds,
            'Test_Accuracy': self.training_history['test_accuracies'],
            'Test_Loss': self.training_history['test_losses']
        }
        
        # Add communication data if available
        if 'aggregation_stats' in self.training_history:
            comm_data = [stats.get('communication_mb', 0) 
                        for stats in self.training_history['aggregation_stats']]
            data['Communication_MB'] = comm_data
            data['Cumulative_Communication_MB'] = np.cumsum(comm_data).tolist()
        
        df = pd.DataFrame(data)
        df.to_csv(save_path, index=False)
        print(f"Detailed results exported to {save_path}")


def analyze_multiple_experiments(results_dir: str, pattern: str = "*.json"):
    """
    Analyze multiple experiment results in a directory
    
    Args:
        results_dir: Directory containing result files
        pattern: File pattern to match
    """
    import glob
    
    result_files = glob.glob(os.path.join(results_dir, pattern))
    
    if not result_files:
        print(f"No result files found in {results_dir} with pattern {pattern}")
        return
    
    print(f"Found {len(result_files)} result files")
    
    # Aggregate results
    all_results = []
    for file_path in result_files:
        try:
            analyzer = FedSAFTLAnalyzer(file_path)
            summary = analyzer.generate_summary_table()
            
            # Extract key info
            experiment_info = {
                'File': os.path.basename(file_path),
                'Dataset': analyzer.config['dataset']['name'],
                'Alpha': analyzer.config['dataset']['alpha'],
                'Clients': analyzer.config['federated']['num_clients'],
                'LoRA_Rank': analyzer.config['model']['lora_rank'],
                'Final_Accuracy': analyzer.final_metrics.get('test_accuracy', 0),
                'Best_Accuracy': analyzer.final_metrics.get('best_accuracy', 0),
                'Total_Communication_MB': analyzer.communication_stats.get('total_communication_mb', 0),
                'Training_Time_min': analyzer.results['time_stats'].get('total_time_minutes', 0)
            }
            all_results.append(experiment_info)
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    if all_results:
        # Create summary DataFrame
        df = pd.DataFrame(all_results)
        print("\nExperiment Summary:")
        print(df.to_string(index=False))
        
        # Save summary
        summary_path = os.path.join(results_dir, "experiment_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\nSummary saved to {summary_path}")
        
        return df
    
    return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze FedSA-FTL results')
    parser.add_argument('--results', type=str, required=True,
                       help='Path to results file or directory')
    parser.add_argument('--mode', type=str, choices=['single', 'multiple'], default='single',
                       help='Analysis mode')
    parser.add_argument('--output-dir', type=str, default='./analysis_output',
                       help='Output directory for plots and analysis')
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == 'single':
        analyzer = FedSAFTLAnalyzer(args.results)
        
        # Generate plots
        analyzer.plot_training_curves(os.path.join(args.output_dir, 'training_curves.png'))
        analyzer.plot_communication_analysis(os.path.join(args.output_dir, 'communication_analysis.png'))
        
        # Generate summary
        summary = analyzer.generate_summary_table()
        print("Experiment Summary:")
        print(summary.to_string(index=False))
        
        # Export detailed results
        analyzer.export_results_csv(os.path.join(args.output_dir, 'detailed_results.csv'))
        
        # Convergence analysis
        convergence = analyzer.analyze_convergence()
        print("\nConvergence Analysis:")
        for key, value in convergence.items():
            print(f"  {key}: {value}")
    
    elif args.mode == 'multiple':
        df = analyze_multiple_experiments(args.results)
        
        if df is not None:
            # Create aggregate plots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Accuracy vs Alpha
            if 'Alpha' in df.columns:
                ax1.scatter(df['Alpha'], df['Final_Accuracy'], alpha=0.7)
                ax1.set_xlabel('Non-IID Level (α)')
                ax1.set_ylabel('Final Accuracy (%)')
                ax1.set_title('Accuracy vs Non-IID Level')
                ax1.grid(True, alpha=0.3)
            
            # Communication vs Clients
            if 'Clients' in df.columns:
                ax2.scatter(df['Clients'], df['Total_Communication_MB'], alpha=0.7)
                ax2.set_xlabel('Number of Clients')
                ax2.set_ylabel('Total Communication (MB)')
                ax2.set_title('Communication vs Number of Clients')
                ax2.grid(True, alpha=0.3)
            
            # LoRA Rank impact
            if 'LoRA_Rank' in df.columns:
                ax3.scatter(df['LoRA_Rank'], df['Final_Accuracy'], alpha=0.7)
                ax3.set_xlabel('LoRA Rank')
                ax3.set_ylabel('Final Accuracy (%)')
                ax3.set_title('Accuracy vs LoRA Rank')
                ax3.grid(True, alpha=0.3)
            
            # Training time distribution
            ax4.hist(df['Training_Time_min'], bins=10, alpha=0.7, edgecolor='black')
            ax4.set_xlabel('Training Time (min)')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Training Time Distribution')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, 'aggregate_analysis.png'), 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"Aggregate analysis saved to {args.output_dir}")
