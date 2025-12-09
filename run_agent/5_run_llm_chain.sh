# Run the LLM-based planning chain for Minecraft tasks.
# Uses a vision language model to generate high-level plans that are executed step by step.
# The system regenerates plans every N steps with updated context from observations.
#
# For some reason, MineRL kills the program unpredictably when we instantiate a couple of environments.
# A simple solution is to run the run_llm_chain in an infinite loop and have the python script only generate outputs
# that are not already present in the output directory. Then, whenever this error happens, the python script will
# exit with a non-zero exit code, which will cause the bash script to restart the python script.
# When it finishes, it should exit with a zero exit code, which will cause the bash script to exit.

# Example usage:
# ./5_run_llm_chain.sh

COMMAND="xvfb-run -a python steve1/run_agent/run_llm_chain.py \
    --task 'Build a tower' \
    --replan-interval 120 \
    --max-steps 2000 \
    --temperature 0.7 \
    --max-tokens 512 \
    --max-history 5 \
    --model claude-3-5-sonnet-20240620 \
    --in-model data/weights/vpt/2x.model \
    --in-weights data/weights/steve1/steve1.weights \
    --text-cond-scale 6.0 \
    --save-video-dirpath outputs/videos"

# Run the command and get its exit status
eval $COMMAND
EXIT_STATUS=$?

# Keep running the command until the exit status is 0 (completes successfully)
while [ $EXIT_STATUS -ne 0 ]; do
    echo
    echo "Encountered an error (likely internal MineRL error), restarting..."
    echo "NOTE: If not MineRL error, then there might be a bug or the parameters might be wrong."
    sleep 10
    eval $COMMAND
    EXIT_STATUS=$?
done
echo "Finished LLM chain execution."
