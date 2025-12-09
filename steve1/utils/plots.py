import os
import numpy as np
import matplotlib.pyplot as plt

def plot_combined_delta_y(results, save_dirpath):
    """
    results: dict {chain_name: [delta_run1, delta_run2, ...]}
    """
    names = []
    means = []
    stds = []
    colors = []

    for name, deltas in results.items():
        if not deltas:
            continue
        
        mean_val = np.mean(deltas)
        std_val = np.std(deltas)
        
        names.append(name[:20] + "..." if len(name) > 20 else name) # Truncate long names
        means.append(mean_val)
        stds.append(std_val)
        colors.append('skyblue' if mean_val >= 0 else 'salmon')

    plt.figure(figsize=(10, 6))
    # Plot bars with error bars
    bars = plt.bar(names, means, yerr=stds, capsize=5, color=colors)
    plt.title("Net Height Change Comparison (Mean ± Std)")
    plt.ylabel("Delta Y (Blocks)")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom' if height >= 0 else 'top')

    plt.tight_layout()
    save_path = os.path.join(save_dirpath, 'combined_delta_y_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined comparison plot to {save_path}")

def plot_combined_dirt_stats(results, save_dirpath):
    """
    results: dict {chain_name: [dirt_run1, dirt_run2, ...]}
    """
    names = []
    means = []
    stds = []
    colors = []

    for name, dirts in results.items():
        if not dirts:
            continue
        
        mean_val = np.mean(dirts)
        std_val = np.std(dirts)
        
        names.append(name[:20] + "..." if len(name) > 20 else name) # Truncate long names
        means.append(mean_val)
        stds.append(std_val)
        colors.append('brown')

    plt.figure(figsize=(10, 6))
    # Plot bars with error bars
    bars = plt.bar(names, means, yerr=stds, capsize=5, color=colors)
    plt.title("Final Dirt Count Comparison (Mean ± Std)")
    plt.ylabel("Dirt Count")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}',
                ha='center', va='bottom')

    plt.tight_layout()
    save_path = os.path.join(save_dirpath, 'combined_dirt_comparison.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined dirt comparison plot to {save_path}")

def plot_ypos_history(ypos_history, save_path):
    """Save a plot of Y-position over time."""
    if not ypos_history:
        print("No Y-position history to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(ypos_history)
    plt.title("Agent Y-Position Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Y Position (Height)")
    plt.grid(True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Y-position plot to {save_path}")

def plot_delta_y(delta_y, save_path):
    """Save a bar plot of the net change in Y-position."""
    plt.figure(figsize=(6, 6))
    plt.bar(['Net Height Change'], [delta_y], color='skyblue' if delta_y >= 0 else 'salmon')
    plt.title(f"Delta Y (End - Start): {delta_y:.2f}")
    plt.ylabel("Blocks")
    plt.grid(axis='y')
    
    # Add text label on the bar
    plt.text(0, delta_y, f"{delta_y:.2f}", ha='center', va='bottom' if delta_y >= 0 else 'top')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Delta Y plot to {save_path}")

def plot_dirt_history(dirt_history, save_path):
    """Save a plot of Dirt count over time."""
    if not dirt_history:
        print("No Dirt history to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    plt.plot(dirt_history, color='brown')
    plt.title("Agent Dirt Inventory Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Dirt Count")
    plt.grid(True)
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Dirt plot to {save_path}")

def plot_combined_ypos_and_dirt(ypos_history, dirt_history, save_path):
    """Save a plot of Y-position and Dirt count over time on the same graph."""
    if not ypos_history or not dirt_history:
        print("Missing history for combined plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Time Step')
    ax1.set_ylabel('Y Position (Height)', color=color)
    ax1.plot(ypos_history, color=color, label='Y Position')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:brown'
    ax2.set_ylabel('Dirt Count', color=color)  # we already handled the x-label with ax1
    ax2.plot(dirt_history, color=color, linestyle='--', label='Dirt Count')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Agent Y-Position and Dirt Inventory Over Time")
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    # Ensure directory exists
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
    print(f"Saved combined plot to {save_path}")

def plot_comparison_histories(experiments_data, save_dirpath):
    """
    Plot comparison of histories from multiple experiments.
    experiments_data: dict {experiment_name: {'ypos_history': [], 'dirt_history': []}}
    """
    if not experiments_data:
        print("No data to plot.")
        return

    # Plot Y-Position Comparison
    plt.figure(figsize=(12, 8))
    for name, data in experiments_data.items():
        if 'ypos_history' in data and data['ypos_history']:
            plt.plot(data['ypos_history'], label=name)
    
    plt.title("Agent Y-Position Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Y Position (Height)")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dirpath, 'comparison_ypos.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Y-pos comparison to {save_path}")

    # Plot Dirt Count Comparison
    plt.figure(figsize=(12, 8))
    for name, data in experiments_data.items():
        if 'dirt_history' in data and data['dirt_history']:
            plt.plot(data['dirt_history'], label=name)
            
    plt.title("Agent Dirt Inventory Comparison")
    plt.xlabel("Time Step")
    plt.ylabel("Dirt Count")
    plt.legend()
    plt.grid(True)
    
    save_path = os.path.join(save_dirpath, 'comparison_dirt.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved Dirt comparison to {save_path}")
