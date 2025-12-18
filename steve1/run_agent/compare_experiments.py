import os
import json
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from steve1.utils.plots import plot_comparison_histories

def load_stats(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compare results from multiple runs (run_chain or run_llm_chain)")
    parser.add_argument('--inputs', nargs='+', required=True, help="List of stats.json files or directories containing them")
    parser.add_argument('--labels', nargs='+', help="List of labels for the inputs (optional, defaults to filename/dirname)")
    parser.add_argument('--output_dir', type=str, default='comparison_results', help="Directory to save comparison plots")
    
    args = parser.parse_args()
    
    experiments_data = {}
    
    for i, input_path in enumerate(args.inputs):
        label = args.labels[i] if args.labels and i < len(args.labels) else None
        
        if os.path.isfile(input_path):
            # It's a file, assume it's a stats.json
            stats = load_stats(input_path)
            if stats:
                if not label:
                    # Use parent dir name as label if file is stats.json
                    if os.path.basename(input_path) == 'stats.json':
                        label = os.path.basename(os.path.dirname(input_path))
                    else:
                        label = os.path.basename(input_path)
                experiments_data[label] = stats
                
        elif os.path.isdir(input_path):
            # It's a directory, look for stats.json inside
            # It might be a run directory or a parent directory containing multiple runs
            
            # Check if stats.json exists directly
            stats_path = os.path.join(input_path, 'stats.json')
            if os.path.exists(stats_path):
                stats = load_stats(stats_path)
                if stats:
                    if not label:
                        label = os.path.basename(input_path)
                    experiments_data[label] = stats
            else:
                # Look for stats_*.json files (from run_llm_chain)
                found = False
                for filename in os.listdir(input_path):
                    if filename.startswith('stats_') and filename.endswith('.json'):
                        stats_path = os.path.join(input_path, filename)
                        stats = load_stats(stats_path)
                        if stats:
                            run_label = label if label else filename.replace('stats_', '').replace('.json', '')
                            experiments_data[run_label] = stats
                            found = True
                
                if not found:
                    print(f"No stats found in {input_path}")
        else:
            print(f"Input not found: {input_path}")

    if not experiments_data:
        print("No valid data found to compare.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Comparing {len(experiments_data)} experiments...")
    plot_comparison_histories(experiments_data, args.output_dir)


def compare_csv_experiments(llm_csv, manual_csv, save_dirpath):
    """
    Compare LLM chain and manual chain experiments from CSV files.
    
    Args:
        llm_csv: Path to LLM chain experiment_results.csv
        manual_csv: Path to manual chain experiment_results.csv
        save_dirpath: Directory to save comparison plots
    """
    # Load data
    llm_df = pd.read_csv(llm_csv)
    manual_df = pd.read_csv(manual_csv)
    
    # Get experiment names
    llm_name = 'Ender' #llm_df.iloc[0]['Task Name'] if 'Task Name' in llm_df.columns else llm_df.iloc[0]['Chain Name']
    manual_name = 'Human Generated' #manual_df.iloc[0]['Chain Name'] if 'Chain Name' in manual_df.columns else manual_df.iloc[0]['Task Name']
    
    # Extract delta_y and dirt data
    llm_delta_y = llm_df['Delta Y'].values
    llm_dirt = llm_df['Final Dirt'].values
    manual_delta_y = manual_df['Delta Y'].values
    manual_dirt = manual_df['Final Dirt'].values
    
    # Calculate statistics
    stats = {
        llm_name: {
            'delta_y_mean': np.mean(llm_delta_y),
            'delta_y_std': np.std(llm_delta_y),
            'dirt_mean': np.mean(llm_dirt),
            'dirt_std': np.std(llm_dirt)
        },
        manual_name: {
            'delta_y_mean': np.mean(manual_delta_y),
            'delta_y_std': np.std(manual_delta_y),
            'dirt_mean': np.mean(manual_dirt),
            'dirt_std': np.std(manual_dirt)
        }
    }
    
    # Create comparison plots
    os.makedirs(save_dirpath, exist_ok=True)
    
    # Plot 1: Combined Side-by-Side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    names = [llm_name, manual_name]
    
    # Delta Y subplot
    delta_y_means = [stats[llm_name]['delta_y_mean'], stats[manual_name]['delta_y_mean']]
    delta_y_stds = [stats[llm_name]['delta_y_std'], stats[manual_name]['delta_y_std']]
    colors_dy = ['skyblue' if m >= 0 else 'salmon' for m in delta_y_means]
    
    bars1 = ax1.bar(names, delta_y_means, yerr=delta_y_stds, capsize=5, color=colors_dy, alpha=0.7, edgecolor='black')
    ax1.set_title('Net Height Change', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Delta Y (Blocks)', fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    for bar, mean, std in zip(bars1, delta_y_means, delta_y_stds):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom' if height >= 0 else 'top', fontsize=9)
    
    # Dirt subplot
    dirt_means = [stats[llm_name]['dirt_mean'], stats[manual_name]['dirt_mean']]
    dirt_stds = [stats[llm_name]['dirt_std'], stats[manual_name]['dirt_std']]
    
    bars2 = ax2.bar(names, dirt_means, yerr=dirt_stds, capsize=5, color='brown', alpha=0.7, edgecolor='black')
    ax2.set_title('Final Dirt Count', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Dirt Count', fontsize=11)
    ax2.grid(axis='y', alpha=0.3)
    
    for bar, mean, std in zip(bars2, dirt_means, dirt_stds):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{mean:.1f}±{std:.1f}',
                ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Experiment Comparison: LLM Chain vs Manual Chain', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    combined_path = os.path.join(save_dirpath, 'comparison_combined.png')
    plt.savefig(combined_path, dpi=150)
    plt.close()
    print(f"Saved combined comparison to {combined_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("EXPERIMENT STATISTICS")
    print("="*60)
    for name in [llm_name, manual_name]:
        print(f"\n{name}:")
        print(f"  Delta Y: {stats[name]['delta_y_mean']:.2f} ± {stats[name]['delta_y_std']:.2f} blocks")
        print(f"  Final Dirt: {stats[name]['dirt_mean']:.2f} ± {stats[name]['dirt_std']:.2f} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare results from multiple runs")
    parser.add_argument('--mode', type=str, choices=['json', 'csv'], default='json',
                        help='Comparison mode: json (individual stats files) or csv (aggregate CSVs)')
    parser.add_argument('--inputs', nargs='+', help="List of stats.json files or directories (for json mode)")
    parser.add_argument('--labels', nargs='+', help="List of labels for the inputs (optional)")
    parser.add_argument('--llm_csv', type=str, 
                        default='data/generated_videos/llm_chain_experiments/experiment_results.csv',
                        help='Path to LLM chain CSV (for csv mode)')
    parser.add_argument('--manual_csv', type=str,
                        default='data/generated_videos/multi_chain_test/experiment_results.csv',
                        help='Path to manual chain CSV (for csv mode)')
    parser.add_argument('--output_dir', type=str, default='comparison_results',
                        help="Directory to save comparison plots")
    
    args = parser.parse_args()
    
    if args.mode == 'csv':
        compare_csv_experiments(args.llm_csv, args.manual_csv, args.output_dir)
    else:
        main()
