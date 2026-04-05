# ALFWorld Evaluation Guide

## Prerequisites

1. **Install alfworld** (provides alfworld + textworld):
   ```bash
   pip install alfworld
   ```

2. **Download ALFWorld game data** and set the env var:
   ```bash
   export ALFWORLD_DATA=/path/to/alfworld_data
   ```
   The data directory should contain `json_2.1.1/` with `train/`, `valid_seen/`, `valid_unseen/` splits, plus `logic/alfred.pddl` and `logic/alfred.twl2`.

3. **Activate the conda environment** with rllm and ORBIT dependencies:
   ```bash
   conda activate icx
   ```

## Running Evaluation

### Option A: OpenAI API evaluation (lightweight, no GPU needed)

Evaluates any OpenAI-compatible model (GPT, or a locally-served model via vLLM/SGLang):

```bash
# Single-episode (1 episode per task)
python scripts/eval/openai.py \
    --config configs/eval_alfworld.yaml \
    --model gpt-4o-mini \
    --env-mode single \
    --n-parallel 32 \
    --output results/alfworld_single.json

# Multi-episode (3 episodes x 10 turns per task)
python scripts/eval/openai.py \
    --config configs/eval_alfworld_multi.yaml \
    --model gpt-4o-mini \
    --env-mode multi \
    --n-parallel 32 \
    --output results/alfworld_multi.json
```

To evaluate a locally-served model (e.g. via vLLM):
```bash
python scripts/eval/openai.py \
    --config configs/eval_alfworld_multi.yaml \
    --model Qwen/Qwen3-8B \
    --base-url http://localhost:8000/v1 \
    --api-key dummy \
    --n-parallel 16 \
    --output results/alfworld_vllm.json
```

Key flags:
- `--n-parallel`: Number of concurrent agent-environment pairs (default: 32)
- `--temperature`: Sampling temperature (default: 0.7)
- `--max-response-length`: Max tokens per response (default: 4096)
- `--trajectory-timeout`: Timeout per trajectory in seconds (default: 600)
- `--n-rollouts`: Rollouts per task for pass@k (default: 1)

### Option B: Full benchmark (GPU required)

Runs ReAct, Reflexion, and ORBIT baselines using the VERL training pipeline with zero training epochs:

```bash
export ALFWORLD_DATA=/path/to/alfworld_data
bash scripts/eval/alfworld_benchmark.sh
```

Edit the script to configure `BASE_MODEL`, `ORBIT_MODEL`, and which baselines to run.

## Configuration

### Task configs

- `configs/eval_alfworld.yaml` — Single-task ALFWorld config (134 valid_unseen games, 10 turns/episode)
- `configs/eval_alfworld_multi.yaml` — Same, with additional commented-out task types (maze, mastermind, grid)
- `configs/config_alfworld_tw.yaml` — TextWorld environment config (points to `$ALFWORLD_DATA` paths)

### Custom vendor path

If you have a custom alfworld installation (not pip-installed), set:
```bash
export ALFWORLD_VENDOR_PATH=/path/to/your/alfworld/vendor/dir
```

Resolution order: `ALFWORLD_VENDOR_PATH` env var > pip-installed alfworld > LaMer fallback.

## Output

Results are saved as JSON with:
- `summary.overall_success_rate`: Fraction of tasks with at least one successful episode
- `summary.per_task.<env_id>`: Per-task success rates, pass@k, avg episodes
- `aggregated_metrics`: Detailed per-episode metrics (success_rate, steps, truncation)
- `trajectories`: Per-trajectory metrics and metadata

Chat completions are logged to `results/chat_completions/` as JSONL files.

## Running Tests

```bash
# Unit tests (no ALFWORLD_DATA needed)
pytest tests/test_alfworld_env_adapter.py -v -k "not DeterministicBehavior and not FromDict"

# Full tests (needs ALFWORLD_DATA)
ALFWORLD_DATA=/path/to/alfworld_data pytest tests/test_alfworld_env_adapter.py -v
```
