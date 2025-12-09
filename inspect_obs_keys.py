
import sys
import os
import gym
import numpy as np

# Add the current directory to sys.path so we can import steve1
sys.path.append(os.getcwd())

from steve1.utils.mineclip_agent_env_utils import make_env

def inspect_obs_keys():
    print("Creating environment to inspect observation keys...")
    try:
        env = make_env(seed=42)
        print("Resetting environment...")
        obs = env.reset()
        
        print("\n--- Observation Keys ---")
        for key in obs.keys():
            print(f"- {key}")
            if key == 'location_stats':
                print(f"  -> Content: {obs[key]}")
        
        env.close()
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    inspect_obs_keys()
