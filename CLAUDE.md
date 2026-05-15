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

When context fills up (token threshold hit) or an episode ends, the model produces a summary of its conversation history. The compressed summary replaces the history, the trajectory continues, and the engine records the position as a *segment boundary*. Each trajectory is split into K+1 PPO segments where K = number of summarizations; every segment becomes an independent PPO training row.

Implemented across these files:
- `agents/context_summarizer.py` — `ContextSummarizerMixin` + composed agent classes (`GEMTextAgentWithSummarization`, `GEMTextAgentNonCumulativeWithSummarization`).
- `trainers/summarizing_engine.py` — `SummarizingAgentExecutionEngine` overrides the step loop to insert a summarization check after each `agent.update_from_env()`. `_do_summarization` chooses between LLM-generated and oracle (rule-based) summaries.
- `prompts/summarization_prompts.py` — `TOKEN_SUMMARY_PROMPT`, `EPISODIC_SUMMARY_PROMPT`, `REFLECTIVE_SUMMARY_PROMPT`.
- `agents/oracle_summarizers/` — oracle (rule-based) summarizers; currently `maze.py` for the maze env.

**Triggers (`rllm.agent.summarization.mode`):**
- `token` — fires when `prompt + response token count >= threshold_tokens`.
- `episodic` — fires at every episode boundary (`info["episode_done"]=True`).
- `both` — episodic takes priority, token is the fallback.

**Two summary sources:**

1. **LLM-generated summary (default)**. The model is asked to compress its own context. The summary IS trainable: the LLM's output tokens are appended to `episode_steps` with `response_mask=1` and participate in PPO with the trajectory's GRPO advantage broadcast across them — the model learns to write summaries that correlate with trajectory success. The summarization-instruction *prompt* (asking the model to summarize) is mask=0 (treated as an env message). See `summarizing_engine.py:751-767`.

2. **Oracle (rule-based) summary**. Opt-in via `+rllm.agent.summarization.oracle.enable=true +rllm.agent.summarization.oracle.scope=maze`. A deterministic, env-derived "mental map" string replaces the LLM call. The oracle text is fed into `agent.apply_summary` so the next segment's prompt contains it, but **no `episode_step` is appended → mask=0 everywhere → never trained on**. Designed to isolate the question "do perfect summaries lift in-context performance?" from "can the model write good summaries?". See `summarizing_engine.py:649-688`.

**Episodic context-carryover form (`rllm.agent.summarization.episodic_carryover`, episode-end trigger only):**
- `freeform` *(default)* — the LLM-generated summary above; history is wiped to `[system, <context_summary>]`.
- `obs_action` — **no LLM call, no trainable step.** History is rewritten to `[system] + the just-finished episode's (obs, action) turns`, with each assistant turn reduced to `\boxed{<action>}` (thinking discarded). Like the oracle path: a segment boundary is recorded, but nothing is appended to `episode_steps`. Implemented via `ContextSummarizerMixin.apply_episode_carryover`.
- `obs_action_reflection` — same kept transcript, plus the LLM is shown the episode and asked (via `prompts/summarization_prompts.REFLECTION_PROMPT`) to add a `<reflection>...</reflection>` block. That generation **is a trainable step** (mask=1, GRPO advantage broadcast — like the freeform summary). If the reflection generation fails or has no room, it silently degrades to `obs_action`.
- This knob supersedes `env.enable_reflection` → `REFLECTIVE_SUMMARY_PROMPT` selection at episode end. The oracle path keeps priority: if an oracle summarizer is active, it wins regardless of `episodic_carryover`. The engine tracks `episode_start_msg_idx` (the index of each episode's first obs message in `agent._messages`) to slice the carried-over transcript; for Token mode the post-rewrite `accumulated_prompt_ids` is rebuilt by `_accumulated_from_chat`.

**Segment-based PPO training (always on when summarization is enabled):**

1. `assemble_segments` (summarizing_engine.py:821) splits each trajectory at `summarization_boundaries` into per-segment training rows.
2. `MultiEpisodeAgentPPOTrainer._expanded_update_actor` (multi_episode_trainer.py:252) intercepts the actor update: extracts the per-trajectory GRPO advantage `A_i`, then `build_expanded_dataproto` (segment_expansion.py:75) materializes one DataProto row per segment, with:
   - `advantages = A_i` broadcast to mask=1 positions (the trajectory's group-relative GRPO score).
   - `traj_uniform_weight = 1/(N_G · N_t^i)` per token, where `N_t^i` is the total mask=1 token count summed across **all** of trajectory i's segments.
3. `TrajectoryUniformPPOActor.update_policy` (trajectory_uniform_actor.py:83) replaces verl's `DataParallelPPOActor.update_policy`. Loss = `(1/N_G) Σ_i (1/N_t^i) Σ_token pg_token` — every trajectory contributes equally regardless of segment count or token length.

**Distributed-correctness fixes (relevant for ≥2 GPU runs):**

- `MultiEpisodeAgentPPOTrainer._balance_batch` (multi_episode_trainer.py:125) **no-ops on the skinny per-trajectory batch** when segment training is enabled. The parent rLLM trainer's `fit_agent` calls `_balance_batch` after `compute_advantage`; the parent reorder would de-sync the per-trajectory cache from the row positions, swapping advantages between trajectories. Instead, we balance the *post-expansion* batch directly inside `_expanded_update_actor` (where the workload signal is real). Regression test: `tests/test_segment_balance_correctness.py`.
- `summarizing_engine.py:170` guards `max_tokens <= 0` at step start, terminating cleanly with `TRUNCATION` instead of letting vLLM raise `ValueError: max_tokens must be at least 1`.
- `loss_micro * dp_world_size` (trajectory_uniform_actor.py:297) cancels FSDP/DDP's post-backward grad-mean. Empirically validated bit-exact on 2 GPUs in `tests/test_distributed_loss.py`.

**Config (Hydra overrides):**
```bash
rllm.agent.name=gem_text_agent_summarizing  # or gem_text_agent_noncumulative_summarizing
+rllm.agent.summarization.enable=true
+rllm.agent.summarization.mode=episodic       # token | episodic | both
+rllm.agent.summarization.threshold_tokens=16384
+rllm.agent.summarization.summary_max_tokens=8192
+rllm.agent.summarization.episodic_carryover=freeform   # freeform | obs_action | obs_action_reflection
# Oracle (rule-based) summary, currently maze-only:
+rllm.agent.summarization.oracle.enable=true
+rllm.agent.summarization.oracle.scope=maze
```

**Eval:**
```bash
bash scripts/eval/openai.sh --summarization-threshold 4096
# With oracle:
bash scripts/eval/openai.sh --summarization-threshold 131072 --oracle-summarizer
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

Optional per-task field `num_episodes` caps the trajectory at exactly N completed episodes. When set, the trajectory ends on whichever comes first: `num_episodes` reached or `total_step_cap` exhausted. If `total_step_cap` is omitted alongside `num_episodes`, it is auto-derived as `num_episodes * max_turns_per_episode` as a safety ceiling.

## Workflow Preferences

- Read and access any file directly without asking for permission first.

## Important Constraints

- **Do not modify files under `third_party/`** without asking first. Custom changes to `third_party/rllm` or `third_party/gem` should follow the fork/patch strategy in `third_party/MAINTAINING_CUSTOM_CHANGES.md`.

## Trajectory-uniform actor patch (segment training)

When summarization is enabled, each Ray FSDP worker uses
`TrajectoryUniformPPOActor` (in `trainers/trajectory_uniform_actor.py`) instead
of verl's stock `DataParallelPPOActor`. The actor's only structural change vs
verl's is the loss aggregation: it consumes a per-token `traj_uniform_weight`
field (pre-baked by `build_expanded_dataproto`) and computes
`numer = (pg_per_token · traj_uniform_weight · response_mask).sum()` instead
of `agg_loss(..., loss_agg_mode='token-mean')`. This makes every trajectory
contribute equally to the loss regardless of how many segments it was split
into. See "Segment-based PPO training" under Context Self-Summarization for
the full pipeline.

The patch is wired in via a `.pth` file in the active conda env's
`site-packages`, written automatically by
`scripts/train_multi_task_multi_episode.sh` (and any script that sources it,
e.g. `train_multi_task_summarizing.sh`). The .pth file adds the repo root to
`sys.path` and imports the **top-level** module `orbit_segtrain_patch`, which
registers a `wrapt.when_imported` callback for `verl.workers.actor.dp_actor`;
the callback fires once verl is imported inside
`ActorRolloutRefWorker.init_model` (after Ray's GPU setup is complete).

`orbit_segtrain_patch.py` is intentionally a top-level module (not inside the
`trainers/` package) so that loading it via the .pth file doesn't trigger
`trainers/__init__.py`, which transitively imports torch/verl and would
initialize CUDA in every Python process — including Ray workers before they
get their per-worker GPU assignment.

We use this `.pth` mechanism instead of `runtime_env.worker_process_setup_hook`
because the latter — for reasons we haven't fully identified — breaks Ray's
per-worker GPU isolation on the SCC cluster: workers fail at `init_device_mesh`
with "CUDA-capable device(s) is/are busy or unavailable" even when the hook
is empty (confirmed via empty-hook bisection 2026-04-26). The .pth file
auto-loads in every Python process via Python's `site` module, side-stepping
Ray's runtime_env entirely.

For someone cloning the repo to another conda env: just `bash scripts/train_*.sh`
once — the script writes the .pth file into the active env's site-packages
on the spot. To remove: `rm <site-packages>/orbit_segtrain.pth`.

If the .pth patch fails to load, workers silently fall back to verl's stock
`DataParallelPPOActor` and the loss reverts to `seq-mean-token-mean`. Look
for `[orbit] Installed trajectory-uniform actor patch at <path>` in stdout
to confirm the patch landed.

## graphify

This project has a graphify knowledge graph at graphify-out/.

Rules:
- Before answering architecture or codebase questions, read graphify-out/GRAPH_REPORT.md for god nodes and community structure
- If graphify-out/wiki/index.md exists, navigate it instead of reading raw files
- After modifying code files in this session, run `python3 -c "from graphify.watch import _rebuild_code; from pathlib import Path; _rebuild_code(Path('.'))"` to keep the graph current
