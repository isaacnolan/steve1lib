#!/bin/bash
#SBATCH --job-name=steve1_plots
#SBATCH --output=logs/steve1_plots_%j.log
#SBATCH --error=logs/steve1_plots_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --mem=32G
#SBATCH --account=pas3150

# Load necessary modules or activate environment
# module load python/3.9
# source activate steve1

cd $SLURM_SUBMIT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs
source /users/PAS2926/inolan/miniconda/etc/profile.d/conda.sh
conda activate steve1
# Command from 6_run_multi_chain.sh
COMMAND="xvfb-run -a python steve1/run_agent/run_chain.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --prior_weights data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --save_dirpath data/generated_videos/multi_chain_test \
    --num_runs 30 \
    --prompt_chain \
        \"Simple Prompt|get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt:1500;Make a tower:1500\""

# Run the command and get its exit status
eval $COMMAND
EXIT_STATUS=$?

# Keep running the command until the exit status is 0 (generates all videos)
while [ $EXIT_STATUS -ne 0 ]; do
    echo
    echo "Encountered an error (likely internal MineRL error), restarting..."
    sleep 10
    eval $COMMAND
    EXIT_STATUS=$?
done
echo "Finished generating all videos and comparison plot."
