# Experiment Comparison Guide

This guide walks you through the complete process of setting up, running experiments, and comparing results using the `compare_experiments.py` script.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Download Weights](#download-weights)
4. [Running Experiments](#running-experiments)
5. [Comparing Results](#comparing-results)
6. [Understanding the Output](#understanding-the-output)

---

## Prerequisites

### System Requirements
- Linux OS (recommended)
- GPU with CUDA support
- At least 16GB RAM
- ~10GB disk space for model weights and data

### Required Software
- Conda/Miniconda
- Python 3.10
- CUDA 11.7 or compatible version

---

## Environment Setup

### Step 1: Create and Activate Conda Environment

```bash
conda create -n steve1 python=3.10
conda activate steve1
```

### Step 2: Install Dependencies

Install packages in the following order:

```bash
# 1. Install PyTorch 2.0 with CUDA support
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia

# 2. Install MineDojo and MineCLIP
pip install minedojo git+https://github.com/MineDojo/MineCLIP

# 3. Install MineRL
pip install git+https://github.com/minerllabs/minerl@v1.0.1

# 4. Install VPT requirements (this will downgrade gym to 0.19 which is required)
pip install gym==0.19 gym3 attrs opencv-python

# 5. Install additional requirements
pip install gdown tqdm accelerate==0.18.0 wandb pandas matplotlib

# 6. Install python-dotenv for API key management
pip install python-dotenv

# 7. Install steve1 package locally
pip install -e .
```

### Step 3: Setup for Headless Server (Optional)

If running on a server without display:

```bash
# Install xvfb
sudo apt-get install xvfb

# All python commands should be prefixed with: xvfb-run -a
```

### Step 4: Configure API Keys (for LLM Chain)

Create a `.env` file in the project root:

```bash
# Create .env file
cat > .env << EOF
OSU_AI_API_KEY=your_api_key_here
EOF
```

**Note:** Get your API key from https://litellmproxy.osu-ai.org/

---

## Download Weights

Download pre-trained model weights:

```bash
chmod +x download_weights.sh
./download_weights.sh
```

This will download:
- VPT model weights (~2GB)
- STEVE-1 fine-tuned weights
- MineCLIP weights
- Prior CVAE model

Verify the download:
```bash
ls -lh data/weights/vpt/
ls -lh data/weights/steve1/
ls -lh data/weights/mineclip/
```

---

## Running Experiments

### Experiment Type 1: Manual Prompt Chains

Run pre-defined sequences of text prompts:

```bash
# Make the script executable
chmod +x run_agent/4_run_chain.sh

# Run the experiment
./run_agent/4_run_chain.sh
```

**What this does:**
- Executes a series of text prompts (e.g., "Look at the Sky", "Dig", "Turn Right")
- Each prompt runs for a specified duration (e.g., 500 steps)
- Generates videos and stats for each chain
- Saves results to `data/generated_videos/chain_test/`

**Output structure:**
```
data/generated_videos/chain_test/
├── chain_0_Look_at_the_Sky/
│   ├── video_run_0.mp4
│   ├── stats_run_0.json
│   ├── ypos_plot.png
│   ├── delta_y_plot.png
│   ├── dirt_plot.png
│   └── combined_plot.png
├── chain_1_Dig/
│   └── ...
├── combined_delta_y_comparison.png
├── combined_dirt_comparison.png
└── experiment_results.csv
```

**Customization:**
Edit `run_agent/4_run_chain.sh` to modify:
- `--prompt_chain`: Add/remove chains in format `"Name|prompt text:duration"`
- `--num_runs`: Number of runs per chain (default: 1)
- `--save_dirpath`: Output directory

### Experiment Type 2: LLM-Driven Planning Chain

Run experiments with autonomous LLM planning:

```bash
# Make the script executable
chmod +x run_agent/5_run_llm_chain.sh

# Run the experiment
./run_agent/5_run_llm_chain.sh
```

**What this does:**
- Uses a Vision Language Model (VLM) to generate adaptive plans
- Replans every N steps based on current observations
- Executes plans using STEVE-1 agent
- Saves videos and statistics

**Output structure:**
```
outputs/videos/
├── llm_chain_Build_a_tower_20251217_143025.mp4
├── stats_20251217_143025.json
├── ypos_plot_20251217_143025.png
├── delta_y_plot_20251217_143025.png
├── dirt_plot_20251217_143025.png
└── combined_plot_20251217_143025.png
```

**Customization:**
Edit `run_agent/5_run_llm_chain.sh` to modify:
- `--task`: High-level task description (e.g., "Build a tower")
- `--replan-interval`: How often to replan (in steps)
- `--max-steps`: Maximum episode length
- `--model`: LLM model to use (see available models with `python steve1/run_agent/check_models.py`)
- `--save-video-dirpath`: Output directory

**Available Models:**
- `gpt-4o-mini-2024-07-18` (default)
- `gpt-4o-2024-11-20`
- `claude-3-5-sonnet-20240620`
- `claude-3-5-haiku-20241022`
- And more (run `python steve1/run_agent/check_models.py` to see all)

### Experiment Type 3: Multi-Chain Comparison

Run multiple chains with multiple runs each:

```bash
# Make the script executable
chmod +x run_agent/6_run_multi_chain.sh

# Run the experiment
./run_agent/6_run_multi_chain.sh
```

**What this does:**
- Runs multiple prompt chains
- Executes each chain N times (e.g., 30 runs)
- Aggregates statistics (mean ± std)
- Generates comparison plots

---

## Comparing Results

### Basic Comparison

Compare two or more experiments:

```bash
python steve1/run_agent/compare_experiments.py \
  --inputs data/generated_videos/chain_test/chain_0_Look_at_the_Sky \
           data/generated_videos/chain_test/chain_1_Dig \
  --labels "Look at Sky" "Dig Down" \
  --output_dir comparison_results
```

### Compare Manual vs LLM Chain

```bash
python steve1/run_agent/compare_experiments.py \
  --inputs data/generated_videos/chain_test/chain_0_Look_at_the_Sky \
           outputs/videos \
  --labels "Manual Chain" "LLM Chain" \
  --output_dir manual_vs_llm_comparison
```

### Compare Multiple Runs

```bash
# Compare all chains from a multi-chain experiment
python steve1/run_agent/compare_experiments.py \
  --inputs data/generated_videos/multi_chain_test/chain_0_Simple_Prompt \
           data/generated_videos/multi_chain_test/chain_1_Make_a_tower \
  --labels "Get Dirt Task" "Build Tower Task" \
  --output_dir task_comparison
```

### Advanced: Compare Specific Stats Files

```bash
# Compare specific JSON files directly
python steve1/run_agent/compare_experiments.py \
  --inputs data/generated_videos/chain_test/chain_0_Look_at_the_Sky/stats_run_0.json \
           outputs/videos/stats_20251217_143025.json \
  --labels "Manual Prompt" "LLM Generated" \
  --output_dir detailed_comparison
```

---

## Understanding the Output

### Statistics JSON Format

Each `stats.json` file contains:

```json
{
  "delta_y": 5.2,              // Net height change (blocks)
  "final_y": 68.5,             // Final Y position
  "initial_y": 63.3,           // Starting Y position
  "final_dirt": 12,            // Dirt count at end
  "max_dirt": 15,              // Maximum dirt held
  "log": 3,                    // Log count
  "seed": 2,                   // Seed count
  "travel_dist": 45.6,         // Horizontal distance traveled
  "ypos_history": [63.3, ...], // Full Y position timeline
  "dirt_history": [0, 2, ...]  // Full dirt count timeline
}
```

### Comparison Plots

The comparison script generates two main plots:

1. **`comparison_ypos.png`**
   - Shows Y-position (height) over time for each experiment
   - Useful for comparing exploration or building behaviors
   - Each line represents a different experiment

2. **`comparison_dirt.png`**
   - Shows dirt inventory over time for each experiment
   - Useful for comparing resource gathering behaviors
   - Each line represents a different experiment

### CSV Format (from run_chain.py)

```csv
Chain Name,Run Index,Delta Y,Final Dirt
Look at the Sky,1,2.5,0
Look at the Sky,2,3.1,0
Dig,1,-8.2,15
Dig,2,-7.9,18
```

---

## Troubleshooting

### Common Issues

**1. MineRL crashes unexpectedly**
- This is normal! The scripts automatically retry
- Check logs for actual errors vs. transient MineRL issues

**2. GPU out of memory**
```bash
# Reduce batch processing or use CPU
export CUDA_VISIBLE_DEVICES=""
```

**3. Missing stats.json files**
```bash
# Check if the experiment completed successfully
ls -la data/generated_videos/chain_test/chain_0_*/
```

**4. API Key errors for LLM chain**
```bash
# Verify your .env file exists and has the correct key
cat .env

# Test API connection
python steve1/run_agent/check_models.py
```

**5. Import errors**
```bash
# Make sure steve1 is installed
pip install -e .

# Verify PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.
```

---

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Setup (one-time)
conda create -n steve1 python=3.10
conda activate steve1
# ... install all dependencies as shown above
./download_weights.sh

# 2. Run a manual chain experiment
./run_agent/4_run_chain.sh

# 3. Run an LLM chain experiment
./run_agent/5_run_llm_chain.sh

# 4. Compare results
python steve1/run_agent/compare_experiments.py \
  --inputs data/generated_videos/chain_test/chain_0_Look_at_the_Sky \
           outputs/videos \
  --labels "Manual: Look at Sky" "LLM: Build Tower" \
  --output_dir my_comparison

# 5. View results
ls -lh my_comparison/
# Open comparison_ypos.png and comparison_dirt.png to view results
```

---

## Tips for Best Results

1. **Consistent Duration**: When comparing chains, use the same `--max-steps` or duration
2. **Multiple Runs**: Use `--num_runs 10` or higher for statistical significance
3. **Clear Labels**: Use descriptive `--labels` for easy interpretation
4. **Organized Outputs**: Create separate output directories for different experiments
5. **Monitor Resources**: Watch GPU memory and disk space during long experiments
6. **Save Logs**: Redirect output to log files for debugging:
   ```bash
   ./run_agent/4_run_chain.sh 2>&1 | tee experiment_log.txt
   ```

---

## Next Steps

- Modify prompt chains in `run_agent/4_run_chain.sh`
- Experiment with different LLM models and replan intervals
- Create custom task prompts in `config/task_prompts.json`
- Analyze CSV files with your own scripts using pandas
- Extend `compare_experiments.py` for custom metrics

For more details, see the main [README.md](README.md).
