"""
Main orchestration for the LLM-based planning chain.

This module coordinates the full pipeline:
1. Initialize agent and environment
2. Generate initial plan using VLM (vision language model)
3. Execute plan step by step with agent
4. Every N steps, regenerate plan with updated context
"""

import logging
import sys
import os
import base64
import cv2
import torch
from typing import List, Tuple, Optional
from io import BytesIO
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steve1.config import PRIOR_INFO, DEVICE
from steve1.data.text_alignment.vae import load_vae_model
from steve1.utils.embed_utils import get_prior_embed
from steve1.utils.mineclip_agent_env_utils import load_mineclip_agent_env, load_mineclip_wconfig
from steve1.utils.video_utils import save_frames_as_video
from steve1.run_agent.programmatic_eval import ProgrammaticEvaluator
from steve1.ender.prompt_builder import PromptBuilder
from steve1.ender.parsers import VLMResponseParser
from steve1.ender.image_processor import ImageProcessor
from steve1.common.models import ImageData, StateActionPair
from steve1.utils.llm_client import query_model

logger = logging.getLogger(__name__)

FPS = 30


def run_llm_chain(
    task_name: str = "Build a tower",
    replan_interval: int = 10,
    max_steps: int = 200,
    temperature: float = 0.7,
    max_tokens: int = 512,
    max_history_length: int = 5,
    llm_model: str = "gpt-4o-mini-2024-07-18",
    in_model: str = "data/weights/vpt/2x.model",
    in_weights: str = "data/weights/steve1/steve1.weights",
    cond_scale: float = 6.0,
    seed: Optional[int] = None,
    save_video_dirpath: Optional[str] = None
) -> None:
    """
    Main orchestration function for the LLM chain.

    Coordinates the full pipeline:
    1. Initialize agent and environment
    2. Generate initial plan using VLM
    3. Execute plan step by step with agent
    4. Every replan_interval steps, regenerate plan

    Args:
        task_name: Name of the task (e.g., "Build a tower")
        replan_interval: Generate new plan every N steps (default: 10)
        max_steps: Maximum total steps to execute (default: 200)
        temperature: LLM sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in LLM response (default: 512)
        max_history_length: Max historical frames to include (default: 5)
        llm_model: LLM model to use (default: gpt-4o-mini)
        in_model: Path to VPT model weights
        in_weights: Path to Steve1 model weights
        cond_scale: Conditioning scale for the model
        seed: Random seed for reproducibility
        save_video_dirpath: Directory to save gameplay video (optional)

    Raises:
        ValueError: If initial plan generation or parsing fails
        Exception: If LLM query or agent/env initialization fails
    """
    logger.info(f"Starting LLM chain for task: {task_name}")
    logger.info(
        f"Config: replan_interval={replan_interval}, max_steps={max_steps}, "
        f"temperature={temperature}, max_history_length={max_history_length}"
    )

    # Initialize agent and environment
    logger.info("Loading agent and environment...")
    agent, mineclip, env = load_mineclip_agent_env(in_model, in_weights, seed, cond_scale)
    obs = env.reset()
    if seed is not None:
        env.seed(seed)
    logger.info("Agent and environment loaded successfully")

    # Initialize components
    prompt_builder = PromptBuilder(config_path="config/task_prompts.json")
    parser = VLMResponseParser()

    # Initialize state tracking
    current_step = 0
    history: List[StateActionPair] = []
    current_image = obs['pov']  # Initialize with first observation
    # Queue of (action, steps, embed) tuples
    action_queue: List[Tuple[str, int, Optional[object]]] = []

    # Initialize frame collection for video saving
    gameplay_frames = []
    current_action_name = "waiting"
    
    # Initialize evaluator
    prog_evaluator = ProgrammaticEvaluator(obs)

    # Models will be loaded on first plan generation
    prior = None

    try:
        while current_step < max_steps:
            # Generate new plan every replan_interval steps
            if current_step % replan_interval == 0:
                logger.info(f"Step {current_step}: Generating new plan...")

                # Process history for context
                if history:
                    images, history_text = ImageProcessor.process_history(
                        history, max_history_length, current_step
                    )
                else:
                    images = []
                    history_text = None

                # Add current image if available
                if current_image is not None:
                    images.append(current_image)

                # Generate plan via VLM
                if not images:
                    logger.warning(
                        "No images available for planning. "
                        "Cannot proceed without observations."
                    )
                    break

                try:
                    messages = prompt_builder.build_tower_prompt(
                        step=current_step,
                        images=images,
                        history_text=history_text,
                        n_steps=replan_interval
                    )

                    # Query the LLM
                    response_text = query_model(
                        messages=messages,
                        model=llm_model,
                        temperature=temperature,
                        max_tokens=max_tokens
                    )

                    logger.info(f"LLM Response:\n{response_text}")

                    # Parse response
                    action_strings, total_steps = parser.parse_single_step(response_text)

                    # Initialize embedding models on first plan generation
                    if prior is None:
                        from steve1.data.text_alignment.vae import load_vae_model as load_vae
                        logger.info("Loading VAE model...")
                        prior = load_vae(PRIOR_INFO)
                        logger.info("VAE model loaded successfully")

                    # Convert to action queue format with embeddings
                    # Each action_string is in format "action_name: steps"
                    action_queue = []
                    for action_str in action_strings:
                        parts = action_str.split(':')
                        action_name = parts[0].strip()
                        steps = int(parts[1].strip())

                        # Create embedding for this action
                        try:
                            embed = get_prior_embed(action_name, mineclip, prior, DEVICE)
                            action_queue.append((action_name, steps, embed))
                            logger.debug(f"Created embedding for action: {action_name}")
                        except Exception as e:
                            logger.error(f"Failed to create embedding for action '{action_name}': {e}")
                            raise

                    logger.info(
                        f"Generated plan with {len(action_queue)} actions, "
                        f"total {total_steps} steps"
                    )

                except ValueError as e:
                    logger.error(f"Failed to parse LLM response: {e}")
                    logger.error(f"Raw response was:\n{response_text}")
                    raise
                except Exception as e:
                    logger.error(f"Error during planning: {e}")
                    raise

            # Execute one action from the queue
            if action_queue:
                action_name, steps, action_embed = action_queue.pop(0)
                logger.info(f"Step {current_step}: Executing '{action_name}' for {steps} steps")
                current_action_name = action_name

                # Execute action steps with the agent
                for step_in_action in range(steps):
                    # Get action from agent using the embedding
                    with torch.cuda.amp.autocast():
                        minerl_action = agent.get_action(obs, action_embed)

                    # Step the environment
                    obs, _, done, _ = env.step(minerl_action)
                    
                    # Update evaluator
                    prog_evaluator.update(obs)

                    # Update current image from observation
                    current_image = obs['pov']

                    # Collect frame with action label for video
                    if save_video_dirpath is not None:
                        try:
                            # Make a copy to ensure the array is contiguous and writable for OpenCV
                            frame = current_image.copy()
                            # frame = cv2.resize(frame, (128, 128)) # Keep original resolution (640x360)

                            # Add action label at bottom of frame
                            # Draw black background for text
                            cv2.rectangle(frame, (0, 330), (640, 360), (0, 0, 0), -1)
                            # Draw action text
                            cv2.putText(frame, action_name[:50], (10, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                            gameplay_frames.append(frame)
                        except Exception as e:
                            logger.warning(f"Failed to collect frame for video: {e}")

                    # Encode image to base64 and add to history
                    try:
                        from PIL import Image as PILImage
                        buffer = BytesIO()
                        PILImage.fromarray(current_image).save(buffer, format='PNG')
                        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
                        history.append(StateActionPair(
                            image=ImageData(data=img_base64),
                            action=action_name
                        ))
                    except Exception as e:
                        logger.warning(f"Failed to add frame to history: {e}")

                    current_step += 1

                    # Check if episode is done
                    if done:
                        logger.info("Episode finished")
                        break
            else:
                # No more actions in queue, need to wait for next plan
                logger.debug(f"Action queue empty at step {current_step}")
                current_step += 1

    except Exception as e:
        logger.error(f"LLM chain failed: {e}")
        raise
    finally:
        # Save video if frames were collected
        if save_video_dirpath is not None and gameplay_frames:
            try:
                os.makedirs(save_video_dirpath, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                video_filename = f"llm_chain_{task_name.replace(' ', '_')}_{timestamp}.mp4"
                save_filepath = os.path.join(save_video_dirpath, video_filename)
                save_frames_as_video(gameplay_frames, save_filepath, FPS, to_bgr=True)
                logger.info(f"Saved gameplay video to {save_filepath}")
                
                # Save evaluation plots and stats
                plot_dir = os.path.dirname(save_filepath)
                
                # Save Y-pos plot
                plot_path = os.path.join(plot_dir, f'ypos_plot_{timestamp}.png')
                prog_evaluator.save_ypos_plot(plot_path)

                # Save Delta Y plot
                delta_plot_path = os.path.join(plot_dir, f'delta_y_plot_{timestamp}.png')
                prog_evaluator.save_delta_y_plot(delta_plot_path)

                # Save Dirt plot
                dirt_plot_path = os.path.join(plot_dir, f'dirt_plot_{timestamp}.png')
                prog_evaluator.save_dirt_plot(dirt_plot_path)

                # Save Combined plot
                combined_plot_path = os.path.join(plot_dir, f'combined_plot_{timestamp}.png')
                prog_evaluator.save_combined_plot(combined_plot_path)
                
                # Save stats
                stats_path = os.path.join(plot_dir, f'stats_{timestamp}.json')
                prog_evaluator.save_stats(stats_path)
                
            except Exception as e:
                logger.error(f"Failed to save video or stats: {e}")

    logger.info(f"LLM chain completed at step {current_step}/{max_steps}")


if __name__ == '__main__':
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(
        description="Run the LLM-based planning chain for Minecraft tasks"
    )
    # LLM chain parameters
    parser.add_argument(
        '--task',
        type=str,
        default='Build a tower',
        help='Name of the task to execute'
    )
    parser.add_argument(
        '--replan-interval',
        type=int,
        default=10,
        help='Generate new plan every N steps'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=200,
        help='Maximum total steps to execute'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.7,
        help='LLM sampling temperature'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=512,
        help='Maximum tokens in LLM response'
    )
    parser.add_argument(
        '--max-history',
        type=int,
        default=5,
        help='Maximum historical frames to include'
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt-4o-mini-2024-07-18',
        help='LLM model to use (e.g., gpt-4o-mini-2024-07-18, gpt-4o-2024-11-20, claude-3-5-sonnet-20240620)'
    )

    # Agent and environment parameters
    parser.add_argument(
        '--in-model',
        type=str,
        default='data/weights/vpt/2x.model',
        help='Path to VPT model weights'
    )
    parser.add_argument(
        '--in-weights',
        type=str,
        default='data/weights/steve1/steve1.weights',
        help='Path to Steve1 model weights'
    )
    parser.add_argument(
        '--text-cond-scale',
        type=float,
        default=6.0,
        help='Conditioning scale for the model'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--save-video-dirpath',
        type=str,
        default=None,
        help='Directory to save gameplay video (optional)'
    )

    args = parser.parse_args()

    run_llm_chain(
        task_name=args.task,
        replan_interval=args.replan_interval,
        max_steps=args.max_steps,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_history_length=args.max_history,
        llm_model=args.model,
        in_model=args.in_model,
        in_weights=args.in_weights,
        cond_scale=args.text_cond_scale,
        seed=args.seed,
        save_video_dirpath=args.save_video_dirpath
    )
