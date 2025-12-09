# Example usage of the multi-chain script
# You can pass multiple chains as separate arguments.
# Each chain is enclosed in quotes.

COMMAND="xvfb-run -a python steve1/run_agent/run_chain.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --prior_weights data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --save_dirpath data/generated_videos/multi_chain_test \
    --num_runs 3
    --prompt_chain \
        \"High Level|Go as high as possible from the starting position:3000\" \
        \"Fine Grained|Explore the area:100; Find the nearest hill and keep looking at it:100; Climb the hill:500; Dig Dirt:500; Build a tower of dirt:1800\" \
        \"Verbose|Explore, go as high as possible, climb the nearest hill, don't go down':1000; 'stay at a high place, get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt':1000; 'build a tower':1000\""

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
