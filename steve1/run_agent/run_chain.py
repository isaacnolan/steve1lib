import os
import sys
import cv2
import torch
from tqdm import tqdm
import argparse
import numpy as np
import json
import csv

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video
from steve1.utils.plots import plot_combined_delta_y, plot_combined_dirt_stats
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator

FPS = 30


def save_results_to_csv(delta_results, dirt_results, save_dirpath):
    """
    Save all run results to a CSV file.
    """
    csv_path = os.path.join(save_dirpath, 'experiment_results.csv')
    
    # Collect all data rows
    rows = []
    for chain_name in delta_results.keys():
        deltas = delta_results.get(chain_name, [])
        dirts = dirt_results.get(chain_name, [])
        
        # Assuming lists are same length (num_runs)
        for i in range(len(deltas)):
            rows.append({
                'Chain Name': chain_name,
                'Run Index': i + 1,
                'Delta Y': deltas[i],
                'Final Dirt': dirts[i] if i < len(dirts) else 0
            })
            
    if not rows:
        print("No results to save to CSV.")
        return

    keys = ['Chain Name', 'Run Index', 'Delta Y', 'Final Dirt']
    
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
        
    print(f"Saved results CSV to {csv_path}")

def run_agent_chain(prompt_chain, save_video_filepath, in_model, in_weights, seed, cond_scale):
    # prompt_chain is a list of (prompt_text, duration, prompt_embed)
    
    # Calculate total steps
    total_steps = sum([duration for _, duration, _ in prompt_chain])
    
    print(f"Starting agent chain with {len(prompt_chain)} prompts, total steps: {total_steps}")

    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)

    obs = env.reset()
    if seed is not None:
        env.seed(seed)

    gameplay_frames = []
    prog_evaluator = ProgrammaticEvaluator(obs)

    current_step = 0
    
    for prompt_text, duration, prompt_embed in prompt_chain:
        print(f"\nSwitching to prompt: '{prompt_text}' for {duration} steps")
        
        for _ in tqdm(range(duration), desc=f"Running '{prompt_text}'"):
            with torch.cuda.amp.autocast():
                minerl_action = agent.get_action(obs, prompt_embed)

            obs, _, _, _ = env.step(minerl_action)
            frame = obs['pov']
            # frame = cv2.resize(frame, (128, 128)) # Keep original resolution (640x360)
            
            # Add text overlay for current prompt
            # Draw black background for text
            cv2.rectangle(frame, (0, 0), (640, 30), (0, 0, 0), -1)
            cv2.putText(frame, prompt_text[:50], (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            gameplay_frames.append(frame)
            prog_evaluator.update(obs)
            current_step += 1

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)

    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()
    
    # Save Y-pos plot
    plot_path = os.path.join(os.path.dirname(save_video_filepath), 'ypos_plot.png')
    prog_evaluator.save_ypos_plot(plot_path)

    # Save Delta Y plot
    delta_plot_path = os.path.join(os.path.dirname(save_video_filepath), 'delta_y_plot.png')
    prog_evaluator.save_delta_y_plot(delta_plot_path)

    # Save Dirt plot
    dirt_plot_path = os.path.join(os.path.dirname(save_video_filepath), 'dirt_plot.png')
    prog_evaluator.save_dirt_plot(dirt_plot_path)

    # Save Combined plot
    combined_plot_path = os.path.join(os.path.dirname(save_video_filepath), 'combined_plot.png')
    prog_evaluator.save_combined_plot(combined_plot_path)
    
    # Save stats
    stats_path = os.path.join(os.path.dirname(save_video_filepath), 'stats.json')
    prog_evaluator.save_stats(stats_path)
    
    return prog_evaluator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--text_cond_scale', type=float, default=6.0)
    parser.add_argument('--save_dirpath', type=str, default='data/generated_videos/')
    parser.add_argument('--prompt_chain', type=str, nargs='+', required=True, 
                        help='List of prompt chains. Each chain is "prompt:duration,prompt:duration".')
    parser.add_argument('--num_runs', type=int, default=1, help='Number of times to run each chain.')
    args = parser.parse_args()

    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(PRIOR_INFO)

    experiment_results = {} # Map chain_name to list of deltas
    dirt_results = {} # Map chain_name to list of final dirt counts

    for i, chain_arg in enumerate(args.prompt_chain):
        print(f"\n--- Processing Chain {i+1}/{len(args.prompt_chain)}: {chain_arg} ---")
        
        # Parse name if present (Format: "Name|prompt:duration,...")
        if '|' in chain_arg:
            chain_name, chain_content = chain_arg.split('|', 1)
            chain_name = chain_name.strip()
            chain_content = chain_content.strip()
        else:
            chain_name = f"Chain {i+1}"
            chain_content = chain_arg

        # Parse chain
        chain_parts = chain_content.split(',')
        parsed_chain = []
        for part in chain_parts:
            part = part.strip()
            if ':' in part:
                prompt, duration = part.rsplit(':', 1)
                try:
                    duration = int(duration)
                except ValueError:
                    print(f"Invalid duration in '{part}', using default 100")
                    duration = 100
            else:
                prompt = part
                duration = 100 # Default
            
            print(f"Encoding prompt: '{prompt}'")
            embed = get_prior_embed(prompt, mineclip, prior, DEVICE)
            parsed_chain.append((prompt, duration, embed))

        deltas_for_chain = []
        dirts_for_chain = []
        
        for run_idx in range(args.num_runs):
            print(f"\n  > Run {run_idx+1}/{args.num_runs} for '{chain_name}'")
            
            # Create a unique subdirectory for this chain's results
            chain_name_safe = "".join([c if c.isalnum() else '_' for c in chain_name])[:50]
            chain_save_dir = os.path.join(args.save_dirpath, f"chain_{i}_{chain_name_safe}")
            save_filepath = os.path.join(chain_save_dir, f'video_run_{run_idx}.mp4')
            stats_path = os.path.join(chain_save_dir, f'stats_run_{run_idx}.json')
            
            # Check if video and stats already exist
            if os.path.exists(save_filepath) and os.path.exists(stats_path):
                print(f"Video and stats already exist, loading from {stats_path}...")
                try:
                    with open(stats_path, 'r') as f:
                        stats = json.load(f)
                        deltas_for_chain.append(stats.get('delta_y', 0))
                        dirts_for_chain.append(stats.get('final_dirt', 0))
                    continue
                except Exception as e:
                    print(f"    Error loading stats: {e}, re-running...")

            evaluator = run_agent_chain(parsed_chain, save_filepath, args.in_model, args.in_weights, None, args.text_cond_scale)
            
            # Rename the stats file created by run_agent_chain (it saves as 'stats.json' in the dir)
            # run_agent_chain saves to os.path.dirname(save_filepath)/stats.json
            default_stats_path = os.path.join(os.path.dirname(save_filepath), 'stats.json')
            if os.path.exists(default_stats_path):
                os.rename(default_stats_path, stats_path)
            
            if evaluator.ypos_history:
                delta = evaluator.ypos_history[-1] - evaluator.ypos_history[0]
            else:
                delta = 0
            deltas_for_chain.append(delta)
            
            if evaluator.dirt_history:
                final_dirt = evaluator.dirt_history[-1]
            else:
                final_dirt = 0
            dirts_for_chain.append(final_dirt)
            
        experiment_results[chain_name] = deltas_for_chain
        dirt_results[chain_name] = dirts_for_chain

    # Generate combined plot
    plot_combined_delta_y(experiment_results, args.save_dirpath)
    plot_combined_dirt_stats(dirt_results, args.save_dirpath)
    
    # Save results to CSV
    save_results_to_csv(experiment_results, dirt_results, args.save_dirpath)
