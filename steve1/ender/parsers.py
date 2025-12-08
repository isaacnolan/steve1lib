"""Parse VLM responses into actions."""
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VLMResponseParser:
    """Handles parsing of VLM responses."""

    @staticmethod
    def parse_single_step(response_text: str) -> Tuple[List[str], int]:
        """
        Extract action list and total step count from VLM response.

        The expected format from the VLM is:
        action_name: number_of_steps
        action_name: number_of_steps
        ...

        Args:
            response_text: Raw text response from the VLM

        Returns:
            Tuple of (action_strings: List[str], total_steps: int)
            where action_strings are the action lines in order
            and total_steps is the sum of all step counts

        Raises:
            ValueError: If response format is invalid or cannot be parsed
        """
        if not response_text or not isinstance(response_text, str):
            raise ValueError("Response text must be a non-empty string")

        # Split response into lines
        lines = response_text.strip().split('\n')
        if not lines:
            raise ValueError("Response text is empty")

        action_strings = []
        total_steps = 0

        for line in lines:
            line = line.strip()

            # Skip empty lines
            if not line:
                continue

            # Try to parse as "action_name: steps"
            if ':' in line:
                parts = line.split(':', 1)  # Split on first colon only
                action_name = parts[0].strip()
                steps_str = parts[1].strip()

                # Try to extract the integer from the steps part
                try:
                    steps = int(steps_str)
                    # Only add if action name is not empty and steps is positive
                    if action_name:
                        action_strings.append(f"{action_name}: {steps}")
                        total_steps += steps
                except ValueError:
                    # Not a valid number, skip this line
                    continue

        if not action_strings:
            raise ValueError(
                f"No valid actions found in response. Expected format: 'action_name: steps'\n"
                f"Got: {response_text}"
            )

        logger.info(f"Parsed {len(action_strings)} actions, total steps: {total_steps}")
        return action_strings, total_steps
