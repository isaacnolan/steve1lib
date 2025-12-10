import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steve1.utils.llm_client import list_available_models

if __name__ == "__main__":
    print("Fetching available models...")
    models = list_available_models()
    print("Available models:")
    for model in models:
        print(f"- {model}")
