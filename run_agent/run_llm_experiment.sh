#!/bin/bash
#SBATCH --job-name=steve1_llm_chain
#SBATCH --output=logs/steve1_llm_%j.log
#SBATCH --error=logs/steve1_llm_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --account=pas3150

# Load necessary modules or activate environment
cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs
source /users/PAS2926/inolan/miniconda/etc/profile.d/conda.sh
conda activate steve1

# Define the command to run
# We use xvfb-run for headless display
COMMAND="xvfb-run -a python steve1/run_agent/run_llm_chain.py \
    --task 'Build a tower' \
    --replan-interval 10 \
    --max-steps 400 \
    --in-model data/weights/vpt/2x.model \
    --in-weights data/weights/steve1/steve1.weights \
    --text-cond-scale 6.0 \
    --save-video-dirpath data/generated_videos/llm_chain_experiments \
    --model gpt-4o-mini"

# Run the command and get its exit status
echo "Starting LLM Chain Experiment..."
eval $COMMAND
EXIT_STATUS=$?

# Keep running the command until the exit status is 0 (generates all videos)
# Note: For LLM chain, we might not want to loop infinitely if it's an API error, 
# but for MineRL crashes this is useful.
RETRY_COUNT=0
MAX_RETRIES=5

while [ $EXIT_STATUS -ne 0 ] && [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo
    echo "Encountered an error (likely internal MineRL error), restarting... (Attempt $((RETRY_COUNT+1))/$MAX_RETRIES)"
    sleep 10
    eval $COMMAND
    EXIT_STATUS=$?
    RETRY_COUNT=$((RETRY_COUNT+1))
done

if [ $EXIT_STATUS -eq 0 ]; then
    echo "Finished LLM chain experiment successfully."
else
    echo "Failed to complete LLM chain experiment after $MAX_RETRIES retries."
fi
