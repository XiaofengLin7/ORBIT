# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**ORBIT** — Multi-task, multi-episode meta-RL framework (arXiv: 2602.04089) that trains LLMs to do in-context online learning via RL. The model is trained via PPO on full multi-episode rollouts, learning to leverage information across episodes.

**Active conda environment (local machine):** `icx`

## Common Commands

### Running Tests

```bash
conda activate icx

# Run individual test files
pytest -q tests/test_context_summarizer.py
pytest -q tests/test_multi_episode_env.py
pytest -q tests/test_gem_text_agent.py

# Run all tests
pytest -q tests/
```

### Training

All training scripts are in `scripts/` and use Hydra overrides passed directly to `scripts/train_multi_episode.py`.

```bash
# Single-task multi-episode (edit ENV_ID, MODEL_PATH etc. at top of script)
bash scripts/train_single_task_multi_episode.sh

# Multi-task multi-episode (default config: configs/multi_task_multi_episode_config.yaml)
bash scripts/train_multi_task_multi_episode.sh
TASKS_CONFIG=configs/my_config.yaml bash scripts/train_multi_task_multi_episode.sh

# Single-episode baseline (val still uses multi-episode)
bash scripts/train_multi_task_single_episode.sh
```

### Evaluation

```bash
# Single-episode (default)
bash scripts/eval/openai.sh

# Multi-episode
ENV_MODE=multi bash scripts/eval/openai.sh
```

## Architecture

### Training Entry Point Flow

`scripts/train_multi_episode.py` → `trainers/train_multi_episode.py:run_ppo_agent()` → `MultiEpisodeAgentPPOTrainer`

When `rllm.agent.summarization.enable=True`, the trainer uses `SummarizingAgentExecutionEngine` instead of the default `MultiEpisodeAsyncAgentExecutionEngine`.

### Infrastructure Stack

```
scripts/train_multi_episode.py   (Hydra config entry point)
  -> trainers/train_multi_episode.py (Ray init, worker + trainer selection)
    -> MultiEpisodeAgentPPOTrainer
      -> rLLM AsyncAgentExecutionEngine / SummarizingAgentExecutionEngine (agent-env loop, token accumulation)
        -> VERL RayPPOTrainer (distributed PPO: FSDP workers, vLLM/SGLang rollout)
```

- **rLLM** (`third_party/rllm`): agent-env interaction → token-level training batches
- **VERL** (`verl`, installed via rLLM): distributed actor/critic updates under Ray

### Multi-Episode Environment

`envs/multi_episode_env.py:MultiEpisodeEnv` wraps any `BaseEnv` and runs N episodes per trajectory. To VERL/rLLM, the entire multi-episode trajectory appears as a single trajectory. Episode boundaries are recorded via `boundary_transition` metadata in step records.

### Context Self-Summarization

When context exceeds a token threshold mid-trajectory, the model generates a summary of its conversation history. The compressed summary replaces the history, and the trajectory continues. This avoids `PROMPT_TRUNCATION` termination and enables longer trajectories.

Implemented across three new files:
- `agents/context_summarizer.py` — `ContextSummarizerMixin` + composed agent classes (`GEMTextAgentWithSummarization`, `GEMTextAgentNonCumulativeWithSummarization`)
- `trainers/summarizing_engine.py` — `SummarizingAgentExecutionEngine` overrides the step loop to insert a summarization check after each `agent.update_from_env()`
- `prompts/summarization_prompts.py` — default summarization prompt template

**Training data handling:**
- **Stepwise mode** (`stepwise_advantage.enable=True`, recommended): Transparent — each step is independent, post-summary steps just have shorter prompts.
- **Cumulative mode** (`stepwise_advantage.enable=False`): Pre-summary segment only is assembled for training. Full trajectory reward (including post-summary success) is used.
- Summary generation tokens are excluded from training data (utility call).

**Config (Hydra overrides):**
```bash
rllm.agent.name=gem_text_agent_summarizing  # or gem_text_agent_noncumulative_summarizing
+rllm.agent.summarization.enable=true
+rllm.agent.summarization.threshold_tokens=4096
+rllm.agent.summarization.summary_max_tokens=512
+rllm.agent.summarization.preserve_recent_turns=2
+rllm.agent.summarization.max_summarizations=5
```

**Eval:**
```bash
bash scripts/eval/openai.sh --summarization-threshold 4096
```

### Environment Adapters

Custom task adapters in `envs/` adapt GEM or custom environments to the rLLM `BaseEnv` interface:
- `gem_env_adapter.py` — generic GEM environment wrapper
- `frozenlake_env_adapter.py`, `sokoban_env_adapter.py`, `maze_env_adapter.py`, etc. — task-specific adapters
- `alfworld_env_adapter.py` — ALFWorld TextWorld adapter; actions parsed from `\boxed{}`, matched against admissible commands (exact → prefix → substring → fuzzy fallback)
- `single_episode_env.py` — single-episode variant for baseline training

### Config System

Configs use Hydra; base config is generated from rLLM (`third_party/rllm/rllm/trainer/config/_generated_agent_ppo_trainer.yaml`). Task-level configs live in `configs/`.

Multi-task configs (`configs/*.yaml`) define `train_tasks` and `val_tasks` lists with per-task `env_id`, `max_turns_per_episode`, `total_step_cap`, and `inner_env_class`.

## Workflow Preferences

- Read and access any file directly without asking for permission first.

## Important Constraints

- **Do not modify files under `third_party/`** without asking first. Custom changes to `third_party/rllm` or `third_party/gem` should follow the fork/patch strategy in `third_party/MAINTAINING_CUSTOM_CHANGES.md`.
