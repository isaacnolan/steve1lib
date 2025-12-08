"""Build prompts for the VLM model."""
import logging
import json
import base64
from typing import List, Dict, Any, Optional
from io import BytesIO
from PIL import Image

logger = logging.getLogger(__name__)


class PromptBuilder:
    """Handles prompt construction for VLM queries."""

    @staticmethod
    def _encode_image_to_base64(img) -> str:
        """
        Convert PIL Image or numpy array to base64-encoded PNG string for OpenAI API.

        Converts PIL Image → PNG bytes (via BytesIO buffer) → base64 string.
        Or numpy array → PIL Image → PNG bytes → base64 string.
        This is efficient because:
        - PIL's C-level libraries handle image encoding optimally
        - BytesIO avoids disk I/O (stays in RAM)
        - PNG compression reduces size by ~70-80% vs raw pixels
        - Base64 format is required by OpenAI API for JSON serialization

        Args:
            img: PIL Image object or numpy array

        Returns:
            Base64-encoded PNG string formatted for OpenAI API: 'data:image/png;base64,...'
        """
        # Convert numpy array to PIL Image if needed
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        buffer = BytesIO()
        img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return f"data:image/png;base64,{img_base64}"

    def __init__(self, config_path: str = "config/task_prompts.json"):
        """
        Initialize the prompt builder.

        Args:
            config_path: Path to task prompts configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)

    def get_task_config(self, task_name: str) -> Dict[str, str]:
        """
        Get configuration for a specific task.

        Args:
            task_name: Name of the task

        Returns:
            Dictionary with system, task_description, action_prompt_template, and velocity_guidance
        """
        return self.config.get(task_name, self.config.get("MineRLBasaltFindCave-v0"))

    def build_text_prompt(self, task_name: str, step: int,
                          history_text: Optional[str] = None,
                          n_steps: int = 60) -> str:
        """
        Build the text prompt for VLM.

        Args:
            task_name: Name of the task
            step: Current step number
            history_text: Optional history summary
            n_steps: Number of steps the agent should plan for (default: 60)

        Returns:
            Formatted text prompt with action template substituted with n value
        """
        task_config = self.get_task_config(task_name)
        action_template = task_config.get('action_prompt_template', '')
        velocity_guidance = task_config.get('velocity_guidance', '')

        # Substitute {n} placeholder in action template
        action_template = action_template.format(n=n_steps)

        if history_text:
            text_prompt = f"""Step {step}

{task_config['task_description']}

Recent History:{history_text}

Current observation is shown in the most recent image.

{velocity_guidance}

{action_template}"""
        else:
            text_prompt = f"""Step {step}

{task_config['task_description']}

{action_template}"""

        return text_prompt

    def build_tower_prompt(self, step: int, images: List[Image.Image],
                          history_text: Optional[str] = None,
                          n_steps: int = 60) -> List[Dict[str, Any]]:
        """
        Build the full message structure for the "Build a tower" task.

        This method includes structured examples so the VLM understands
        the expected output format.

        Args:
            step: Current step number
            images: List of PIL Images
            history_text: Optional history summary
            n_steps: Number of steps the agent should plan for (default: 60)

        Returns:
            Messages list in OpenAI API format with system and user roles
        """
        task_name = "Build a tower"
        task_config = self.get_task_config(task_name)

        # Build text prompt
        text_prompt = self.build_text_prompt(task_name, step, history_text, n_steps)

        # Build content with all images
        content = []

        # Add all images to content with labels
        for idx, img in enumerate(images):
            if len(images) > 1:
                # Label images in history
                if idx < len(images) - 1:
                    content.append({
                        "type": "text",
                        "text": f"[Historical Frame {idx + 1}]"
                    })
                else:
                    content.append({
                        "type": "text",
                        "text": "[Current Frame]"
                    })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image_to_base64(img)
                }
            })

        # Add the main text prompt at the end
        content.append({
            "type": "text",
            "text": text_prompt
        })

        messages = [
            {
                "role": "system",
                "content": task_config['system']
            },
            {
                "role": "user",
                "content": content
            }
        ]

        return messages

    def build_messages(self, task_name: str, images: List[Image.Image],
                      text_prompt: str, n_steps: int = 60) -> List[Dict[str, Any]]:
        """
        Build the full message structure for VLM.

        Args:
            task_name: Name of the task
            images: List of PIL Images
            text_prompt: Text prompt string
            n_steps: Number of steps the agent should plan for (default: 60)

        Returns:
            Messages list in format expected by model
        """
        task_config = self.get_task_config(task_name)

        # Build content with all images
        content = []

        # Add all images to content
        for idx, img in enumerate(images):
            if len(images) > 1:
                # Label images in history
                if idx < len(images) - 1:
                    content.append({
                        "type": "text",
                        "text": f"[Historical Frame {idx + 1}]"
                    })
                else:
                    content.append({
                        "type": "text",
                        "text": "[Current Frame]"
                    })
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": self._encode_image_to_base64(img)
                }
            })

        # Add the main text prompt at the end
        content.append({
            "type": "text",
            "text": text_prompt
        })

        messages = [
            {
                "role": "system",
                "content": task_config['system']
            },
            {
                "role": "user",
                "content": content
            }
        ]

        return messages
