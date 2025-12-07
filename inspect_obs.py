
import sys
import os

# Add the current directory to sys.path so we can import steve1
sys.path.append(os.getcwd())

from steve1.utils.mineclip_agent_env_utils import make_env

def inspect_env():
    print("Creating environment...")
    # We don't need the agent or mineclip for this test, just the env
    env = make_env(seed=42)
    
    print("Resetting environment...")
    obs = env.reset()
    
    print("\nObservation keys available:")
    for key in obs.keys():
        print(f"- {key}")
        
    # Check if inventory is present
    if 'inventory' in obs:
        print("\nInventory details:")
        print(obs['inventory'])
    else:
        print("\n'inventory' key NOT found in observation.")

    env.close()

if __name__ == "__main__":
    inspect_env()
