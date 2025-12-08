# LLM-Based Planning Chain Implementation

## Overview
Implemented a complete orchestration system for the "Build a tower" task using Vision Language Models (VLM) to generate high-level plans that are executed step-by-step by a Minecraft agent.

## Files Created/Modified

### 1. **config/task_prompts.json** (NEW)
- Configuration file defining task-specific prompts
- Contains "Build a tower" task with:
  - System role: Agent instructions
  - Task description: Clear workflow (collect materials → position → place blocks)
  - Action prompt template: Output format with examples
  - Velocity guidance: Execution instructions

### 2. **steve1/utils/llm_client.py** (ENHANCED)
- Added `query_model()` function
- Wraps OpenAI client with configurable parameters
- Parameters: messages, model, temperature, max_tokens
- Returns raw response text from the LLM

### 3. **steve1/ender/prompt_builder.py** (REFACTORED)
- Added `build_tower_prompt()` method specialized for tower building
- Includes structured examples showing expected action output format
- Integrates task configuration from task_prompts.json
- Returns OpenAI API-compliant message structure

### 4. **steve1/ender/parsers.py** (COMPLETED)
- Implemented `parse_single_step()` method
- Parses VLM response into action list and total step count
- Returns tuple: `(List[str], int)` format: `(action_strings, total_steps)`
- Strict validation with fail-fast on malformed output
- Regex-based parsing of "action_name: steps" format

### 5. **steve1/run_agent/run_llm_chain.py** (NEW)
- Main orchestration function: `run_llm_chain()`
- Parameters:
  - `task_name`: Task to execute
  - `replan_interval`: Generate new plan every N steps
  - `max_steps`: Maximum execution steps
  - `temperature`: LLM sampling temperature
  - `max_tokens`: Max tokens in response
  - `max_history_length`: Historical frames for context
- Core workflow:
  1. Every `replan_interval` steps, generate new plan via VLM
  2. Parse plan and create action embeddings via `get_prior_embed()`
  3. Execute actions one at a time with step-by-step agent integration
  4. Update history with observations for next planning cycle
- Action queue structure: `(action_name, steps, action_embed)`
- Includes commented integration hooks for agent/env execution

### 6. **steve1/ender/image_processor.py** (VERIFIED)
- No changes required
- Already compatible with Pydantic models
- Properly handles base64 encoding/decoding of images

### 7. **run_agent/5_run_llm_chain.sh** (NEW)
- Shell script to execute the LLM chain
- Based on existing 4_run_chain.sh pattern
- Includes error recovery loop for MineRL crashes
- Configurable parameters via command-line args

## Data Flow

```
1. Generate Plan (every replan_interval steps):
   Observations → PromptBuilder.build_tower_prompt()
   → LLM (query_model) → VLMResponseParser.parse_single_step()
   → Action embeddings (get_prior_embed)

2. Execute Actions:
   Action queue (action_name, steps, embed) 
   → agent.get_action(obs, embed)
   → env.step(minerl_action)
   → Update history with observations

3. Loop:
   Steps += actions_executed
   If steps % replan_interval == 0:
      Go to step 1
```

## Key Design Decisions

1. **Lazy Loading**: Embedding models (mineclip, prior) loaded only on first plan generation
2. **Embedding in Queue**: Actions stored with pre-computed embeddings to avoid repeated computation
3. **Structured Prompts**: Task configuration includes example outputs for VLM guidance
4. **Strict Parsing**: Parser fails fast on invalid output with clear error messages
5. **Modular Architecture**: Each component (builder, parser, processor) handles specific responsibility

## Integration Points (TO DO)

The following requires integration with actual agent/environment:

1. **Environment Setup**: Initialize agent and MineRL environment
2. **Agent Execution Loop**: Uncomment code in `run_llm_chain.py` lines 181-196
3. **Observation Processing**: 
   - Capture `obs['pov']` from env.step()
   - Encode to base64 for StateActionPair history
4. **Done Signal**: Handle episode termination

## Testing & Validation

- Config file syntax valid (JSON)
- Modules compile successfully (Python 3)
- Type hints compatible with Pydantic models
- Parser validates action format strictly
- Embedding creation follows existing patterns from run_chain.py

## Example Usage

```bash
# Basic execution
./run_agent/5_run_llm_chain.sh

# Manual execution with custom parameters
python steve1/run_agent/run_llm_chain.py \
  --task "Build a tower" \
  --replan-interval 10 \
  --max-steps 200 \
  --temperature 0.7 \
  --max-tokens 512 \
  --max-history 5
```

## Dependencies

- Pydantic (models validation)
- OpenAI client (LLM queries)
- PIL/Pillow (image processing)
- torch (embeddings - lazy loaded)
- VPT/MineRL libraries (agent execution - when integrated)
