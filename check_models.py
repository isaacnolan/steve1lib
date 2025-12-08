#!/usr/bin/env python3
"""
Check available models on OSU AI proxy and try free options.
Run this to discover which free models are available.

Usage:
    python check_models.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from steve1.utils.llm_client import list_available_models

print("=" * 70)
print("OSU AI Proxy - Available Models")
print("=" * 70)

models = list_available_models()

if models:
    print(f"\nFound {len(models)} available model(s):\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")

    print("\n" + "=" * 70)
    print("RECOMMENDATION:")
    print("=" * 70)

    # Suggest which model to use
    free_models = []
    for model in models:
        if any(free in model.lower() for free in ['ollama', 'llama', 'mistral', 'zephyr', 'neural']):
            free_models.append(model)

    if free_models:
        print(f"\nFree/Open-source models available:")
        for model in free_models:
            print(f"  - {model}")
        print(f"\nTry using one of these with:")
        print(f"  python steve1/run_agent/run_llm_chain.py --model '{free_models[0]}'")
    else:
        print(f"\nNo obvious free models detected, but you could try:")
        print(f"  python steve1/run_agent/run_llm_chain.py --model '{models[0]}'")

else:
    print("\nERROR: Could not retrieve model list.")
    print("\nMake sure:")
    print("  1. OSU_AI_API_KEY is set in .env file or environment")
    print("  2. You have internet connectivity")
    print("  3. The OSU AI proxy is accessible at: https://litellmproxy.osu-ai.org/")
    print("\nTo set up your API key:")
    print("  1. Create/edit .env file in project root")
    print("  2. Add: OSU_AI_API_KEY=your_actual_key_here")
    sys.exit(1)

print("\n" + "=" * 70)
