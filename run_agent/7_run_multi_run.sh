# Example usage of the multi-chain script with multiple runs
# You can pass multiple chains as separate arguments.
# Each chain is enclosed in quotes.

COMMAND="xvfb-run -a python steve1/run_agent/run_chain.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --prior_weights data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --save_dirpath data/generated_videos/multi_run_test \
    --num_runs 3 \
    --prompt_chain \
        \"Look and Chop|look around:100, chop tree:200\" \
        \"Digging Task|dig hole:100, collect dirt:200\" \
        \"Tower Building|build tower:300\""

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
