import numpy as np
import os
import json
from steve1.utils.plots import plot_ypos_history, plot_delta_y, plot_dirt_history, plot_combined_ypos_and_dirt

class ProgrammaticEvaluator:
    """Class for keeping track of: travel distance, seed count, dirt count, and log count."""
    def __init__(self, initial_obs) -> None:
        # Store the max inventory counts for each block type and travel distance (these are lower bound measures)
        self.prog_values = {}
        self.initial_obs = initial_obs
        self.ypos_history = []
        self.dirt_history = []

    def update(self, obs):
        """Update the programmatic evaluation metrics."""
        self.prog_values = compute_programmatic_rewards(self.initial_obs, obs, self.prog_values)
        if 'location_stats' in obs:
            self.ypos_history.append(obs['location_stats']['ypos'])
        
        # Track dirt count
        dirt_count = 0
        if 'inventory' in obs:
            for key, val in obs['inventory'].items():
                if 'dirt' in key:
                    dirt_count += val
        self.dirt_history.append(dirt_count)

    def print_results(self):
        """Print the results of the programmatic evaluation."""
        print("Programmatic Evaluation Results:")
        for prog_task in self.prog_values.keys():
            print(f"{prog_task}: {self.prog_values[prog_task]}")
        print()

    def save_ypos_plot(self, save_path):
        """Save a plot of Y-position over time."""
        plot_ypos_history(self.ypos_history, save_path)

    def save_delta_y_plot(self, save_path):
        """Save a bar plot of the net change in Y-position."""
        if not self.ypos_history:
            print("No Y-position history to plot.")
            return

        initial_y = self.ypos_history[0]
        final_y = self.ypos_history[-1]
        delta_y = final_y - initial_y
        
        plot_delta_y(delta_y, save_path)

    def save_dirt_plot(self, save_path):
        """Save a plot of Dirt count over time."""
        plot_dirt_history(self.dirt_history, save_path)

    def save_combined_plot(self, save_path):
        """Save a combined plot of Y-position and Dirt count over time."""
        plot_combined_ypos_and_dirt(self.ypos_history, self.dirt_history, save_path)

    def save_stats(self, save_path):
        """Save evaluation statistics to a JSON file."""
        stats = {}
        # Convert numpy types to python types for JSON serialization
        for k, v in self.prog_values.items():
            if isinstance(v, (np.integer, np.int32, np.int64)):
                stats[k] = int(v)
            elif isinstance(v, (np.floating, np.float32, np.float64)):
                stats[k] = float(v)
            else:
                stats[k] = v

        if self.ypos_history:
            stats['delta_y'] = float(self.ypos_history[-1] - self.ypos_history[0])
            stats['final_y'] = float(self.ypos_history[-1])
            stats['initial_y'] = float(self.ypos_history[0])
        else:
            stats['delta_y'] = 0.0
            
        if self.dirt_history:
            stats['final_dirt'] = float(self.dirt_history[-1])
            stats['max_dirt'] = float(max(self.dirt_history))
        else:
            stats['final_dirt'] = 0.0
            stats['max_dirt'] = 0.0
            
        # Save full history for replotting
        stats['ypos_history'] = [float(x) for x in self.ypos_history]
        stats['dirt_history'] = [float(x) for x in self.dirt_history]
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(stats, f, indent=4)
        print(f"Saved stats to {save_path}")

def update_max_inventory_counts(current_inventory, inventory_counts, block_type, block_key):
    """ Update the inventory counts for the block type

    Args:
        current_inventory (dict): Dictionary containing the agent's current inventory counts for each block type
        inventory_counts (dict): Dictionary containing the max inventory counts for each block type
        block_type (str): The string filter for the block type to update the inventory count for
        block_key (str): The key for the block type in the inventory dictionary
    """
    block_names = [x for x in current_inventory.keys() if block_type in x]
    block_count = 0
    for block_name in block_names:
        block_count += current_inventory.get(block_name, 0)

    # Update the dirt count in inventory_counts
    if block_count > inventory_counts.get(block_key, 0):
        print(f"Updating count for {block_key} from {inventory_counts.get(block_key, 0)} to {block_count}")
        inventory_counts[block_key] = block_count

    return inventory_counts


def compute_programmatic_rewards(obs_init, obs_current, prog_values):
    """Compute the inventory count across various types of blocks."""
    # Handle missing inventory key gracefully
    if 'inventory' not in obs_current:
        return prog_values

    current_inventory = obs_current['inventory']

    block_filter_types = ["_log", "dirt", "seed"]
    block_names = ["log", "dirt", "seed"]

    # Update the inventory counts for the block types
    for block_name in block_names:
        if block_name not in prog_values:
            prog_values[block_name] = 0

    for block_filter_type, block_name in zip(block_filter_types, block_names):
        prog_values = update_max_inventory_counts(current_inventory, prog_values, block_filter_type,
                                                  block_name)

    # Keep track of the travel distance. The travel distance is the Euclidean distance from the spawn point to the
    # farthest point the agent reached during the episode on the horizontal (x-z) plane
    if 'location_stats' in obs_current and 'location_stats' in obs_init:
        curr_x, curr_z = obs_current['location_stats']['xpos'], obs_current['location_stats']['zpos']

        # Compute the Euclidean distance from the spawn point to the current location
        dist = np.sqrt(
            (curr_x - obs_init['location_stats']['xpos']) ** 2 + (curr_z - obs_init['location_stats']['zpos']) ** 2)

        if dist > prog_values.get("travel_dist", 0):
            prog_values["travel_dist"] = dist

    return prog_values
