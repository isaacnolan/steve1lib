# LLM Chain Setup Guide

## Installation

### 1. Install Dependencies

Install the required packages for the LLM chain:

```bash
# Install all dependencies including LLM requirements
pip install -e .

# Or install just LLM extras
pip install -e ".[llm]"

# Or manually install key packages
pip install python-dotenv openai pydantic pillow
```

### 2. Configure API Key

Create a `.env` file in the project root directory with your OSU AI API key:

```bash
# Copy the example file
cp .env.example .env

# Edit .env and add your API key
# OSU_AI_API_KEY=your_actual_api_key_here
```

Alternatively, set the environment variable directly:

```bash
export OSU_AI_API_KEY=your_api_key_here
```

## Running the LLM Chain

### Option 1: Using the shell script

```bash
./run_agent/5_run_llm_chain.sh
```

### Option 2: Direct Python execution

```bash
python steve1/run_agent/run_llm_chain.py \
  --task "Build a tower" \
  --replan-interval 10 \
  --max-steps 200 \
  --temperature 0.7 \
  --max-tokens 512 \
  --max-history 5
```

### Option 3: Get help on arguments

```bash
python steve1/run_agent/run_llm_chain.py --help
```

## Configuration

### Task Prompts

The task prompts are configured in `config/task_prompts.json`. This file contains:

- **System role**: Instructions for the VLM
- **Task description**: What the agent should do
- **Action prompt template**: Expected output format with examples
- **Velocity guidance**: How to execute actions smoothly

### Environment Variables

- `OSU_AI_API_KEY`: Your API key for the OSU AI proxy
- Loaded automatically from `.env` file if present

## Troubleshooting

### "OSU_AI_API_KEY not found"

Make sure:
1. `.env` file exists in project root
2. It contains `OSU_AI_API_KEY=your_key`
3. Or set environment variable: `export OSU_AI_API_KEY=your_key`

### "ModuleNotFoundError: No module named 'dotenv'"

Install python-dotenv:

```bash
pip install python-dotenv
```

Or install the LLM extras:

```bash
pip install -e ".[llm]"
```

### Agent/Environment Integration

The main execution loop has integration hooks (lines 173-196 in `run_llm_chain.py`). To enable actual Minecraft execution:

1. Initialize agent and environment
2. Uncomment the agent execution loop
3. Connect to MineRL environment
4. Handle observation capture and history tracking

See comments in `run_llm_chain.py` for exact integration points.

## File Structure

```
STEVE-1/
├── .env                          # API configuration (create from .env.example)
├── .env.example                  # Example .env template
├── setup.py                      # Package installation config
├── config/
│   └── task_prompts.json         # Task-specific prompt configurations
├── steve1/
│   ├── common/
│   │   ├── __init__.py
│   │   └── models.py             # Pydantic data models
│   ├── ender/
│   │   ├── __init__.py
│   │   ├── image_processor.py    # Image encoding/decoding
│   │   ├── parsers.py            # VLM response parsing
│   │   └── prompt_builder.py     # Prompt construction
│   ├── run_agent/
│   │   ├── __init__.py
│   │   ├── run_llm_chain.py      # Main orchestration
│   │   └── 5_run_llm_chain.sh    # Executable shell script
│   └── utils/
│       └── llm_client.py         # LLM API client
```

## Key Components

- **PromptBuilder**: Creates task-specific prompts with examples
- **VLMResponseParser**: Parses LLM output into structured actions
- **ImageProcessor**: Handles image encoding/decoding
- **query_model()**: Makes API calls to the LLM
- **run_llm_chain()**: Main orchestration function

## Next Steps

1. Set up `.env` with API key
2. Install dependencies: `pip install -e ".[llm]"`
3. Test basic functionality: `python steve1/run_agent/run_llm_chain.py --help`
4. Integrate with MineRL agent and environment
5. Run full pipeline: `./run_agent/5_run_llm_chain.sh`
