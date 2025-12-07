import os
import sys
import cv2
import torch
from tqdm import tqdm
import argparse
import numpy as np

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator

FPS = 30

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
            frame = cv2.resize(frame, (128, 128))
            
            # Add text overlay for current prompt
            # Draw black background for text
            # cv2.rectangle(frame, (0, 0), (128, 20), (0, 0, 0), -1)
            # cv2.putText(frame, prompt_text[:20], (2, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
            
            gameplay_frames.append(frame)
            #prog_evaluator.update(obs)
            current_step += 1

    # Make the eval episode dir and save it
    os.makedirs(os.path.dirname(save_video_filepath), exist_ok=True)
    save_frames_as_video(gameplay_frames, save_video_filepath, FPS, to_bgr=True)

    # Print the programmatic eval task results at the end of the gameplay
    prog_evaluator.print_results()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_model', type=str, default='data/weights/vpt/2x.model')
    parser.add_argument('--in_weights', type=str, default='data/weights/steve1/steve1.weights')
    parser.add_argument('--prior_weights', type=str, default='data/weights/steve1/steve1_prior.pt')
    parser.add_argument('--text_cond_scale', type=float, default=6.0)
    parser.add_argument('--save_dirpath', type=str, default='data/generated_videos/')
    parser.add_argument('--prompt_chain', type=str, required=True, 
                        help='Comma separated list of "prompt:duration". E.g. "chop wood:200,craft planks:100"')
    args = parser.parse_args()

    mineclip = load_mineclip_wconfig()
    prior = load_vae_model(PRIOR_INFO)

    # Parse chain
    chain_parts = args.prompt_chain.split(';')
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

    save_filepath = os.path.join(args.save_dirpath, 'chain_video.mp4')
    run_agent_chain(parsed_chain, save_filepath, args.in_model, args.in_weights, None, args.text_cond_scale)
