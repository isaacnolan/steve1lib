# For some reason, MineRL kills the program unpredictably when we instantiate a couple of environments.
# A simple solution is to run the run_agent in an infinite loop and have the python script only generate videos
# that are not already present in the output directory. Then, whenever this error happens, the python script will
# exit with a non-zero exit code, which will cause the bash script to restart the python script.
# When it finishes all videos, it should exit with a zero exit code, which will cause the bash script to exit.

# Example usage of the prompt chain script
# Format: "prompt1:duration1,prompt2:duration2,..."
# Duration is in steps (20 steps = 1 second approx)
#prompt_chain="Explore, go as high as possible, climb the nearest hill':1000; 'get dirt, dig hole, dig dirt, gather a ton of dirt, collect dirt':1000; 'build a tower':1000"

COMMAND="xvfb-run -a python steve1/run_agent/run_chain.py \
    --in_model data/weights/vpt/2x.model \
    --in_weights data/weights/steve1/steve1.weights \
    --prior_weights data/weights/steve1/steve1_prior.pt \
    --text_cond_scale 6.0 \
    --save_dirpath data/generated_videos/chain_test \
     --prompt_chain \
        \"Look at the Sky|Look at the Sky:500\" \
        \"Dig|Dig as far as possible:500\" \
        \"Turn Right|turn right:500\" \
        \"Stay Still|stay absolutely still, don't move the character at all:500\" \
        \"Jump Up and Down|jump up and down repeatedly:500\" \
        \"Move in a Circle|move in a circle:500\" \
        \"Look Down and Spin|look down at the ground and spin around:500\""

# Run the command and get its exit status
eval $COMMAND
EXIT_STATUS=$?

# Keep running the command until the exit status is 0 (generates all videos)
while [ $EXIT_STATUS -ne 0 ]; do
    echo
    echo "Encountered an error (likely internal MineRL error), restarting (will skip existing videos)..."
    echo "NOTE: If not MineRL error, then there might be a bug or the parameters might be wrong."
    sleep 10
    eval $COMMAND
    EXIT_STATUS=$?
done
echo "Finished generating all videos."