import os
import json
import argparse
import sys

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

if __name__ == "__main__":
    main()
