import os
import openai
import logging
from typing import List, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file in project root
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    logger.debug("python-dotenv not installed. Install with: pip install python-dotenv")

# Initialize the client once here
api_key = os.getenv("OSU_AI_API_KEY")
if not api_key:
    logger.warning("OSU_AI_API_KEY not found in environment or .env file")

client = openai.OpenAI(
    api_key=api_key,
    base_url="https://litellmproxy.osu-ai.org/"
)

def list_available_models():
    """
    List available models from the OSU AI proxy.

    Returns:
        List of model IDs available for your API key
    """
    try:
        models = client.models.list()
        model_names = [model.id for model in models.data]
        return model_names
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return []


def get_completion(messages):
    """
    Wrapper to ensure all calls use the correct model and client.
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini-2024-07-18",  # Updated to a valid model
        messages=messages
    )
    return response

def query_model(messages: List[Dict[str, Any]],
                model: str = "gpt-4o-mini-2024-07-18",
                temperature: float = 0.7,
                max_tokens: int = 512) -> str:
    """
    Query the LLM with messages and return response text.

    Args:
        messages: List of message dictionaries in OpenAI format
        model: Model name to use (default: gpt-4o-mini-2024-07-18)
        temperature: Sampling temperature (default: 0.7)
        max_tokens: Maximum tokens in response (default: 512)

    Returns:
        Response text from the model

    Raises:
        Exception: If API call fails
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        logger.error(f"LLM query failed: {e}")
        raise