# SDPO Self-Distillation (v1)

## Overview

This document describes the implemented shared-weight SDPO self-distillation path used in multi-episode training.

At each training step:
1. Run the normal teacher PPO update on full rollout data.
2. Run one auxiliary distillation actor update on first-attempt tokens only.

Distillation is skipped per sample when the context-overflow guard is triggered.

## Scope

Current scope is trajectory-mode training (`stepwise_advantage.enable=false`).

Implemented files:
- `trainers/sdpo_self_distill_trainer.py`
- `trainers/train_multi_episode.py`
- `trainers/__init__.py`
- `third_party/rllm/rllm/engine/agent_execution_engine.py` (minimal metadata additions)

## Configuration

Distillation is enabled from `rllm.distill`:

```yaml
rllm:
  distill:
    enable: true
    lambda: 0.1
    mode: sdpo_self
    denominator_mode: teacher_adapted_feedback
    context_limit: null
    context_overflow_policy: skip_loss
    min_distill_tokens: 1
```

Semantics:
- `enable`: turn on SDPO auxiliary update.
- `lambda`: scales distill advantages.
- `mode`: currently supports `sdpo_self`.
- `denominator_mode`: currently `teacher_adapted_feedback`.
- `context_limit`: if `null`, defaults to `data.max_prompt_length`.
- `context_overflow_policy`: currently supports only `skip_loss`.
- `min_distill_tokens`: skip auxiliary update when valid token count is below threshold.

## Rollout Metadata Requirements

To support distillation, token rollouts now include:
- `first_attempt_response_mask`: token-level mask selecting attempt-0 completion tokens.
- `step_records`: per-step list containing prompt/completion ids, logprobs, text fields, and `episode_index`.

These are produced in `AgentExecutionEngine` token mode.

## Distillation Objective Used in Code

For valid tokens, compute:

- Numerator log-prob: `log πθ(a_t | x, y<t)`
- Denominator log-prob: `log πθ(a_t | x, f, y<t)`
- Coefficient:
  - `c_t = stopgrad(logp_num - logp_den)`
- Distill advantage:
  - `A_t^distill = lambda * c_t`

Masking:
- Only tokens with `response_mask=1` and `first_attempt_response_mask=1` are used.
- Invalid token-mismatch trajectories are fully masked by engine mismatch filtering.

## Context-Overflow Guard

Per sample:

1. Estimate student context length `L_s` from first-attempt `step_records` as:
   - max prompt length over `episode_index == 0` steps.
2. Serialize retry feedback `f` from `episode_index > 0` steps.
3. Tokenize feedback and estimate teacher context length:
   - `L_t = L_s + len(tokenize(f))`
4. Determine limit:
   - `limit = rllm.distill.context_limit or data.max_prompt_length`
5. Guard:
   - if `L_s + L_t > limit`, skip distillation for this sample.

Boundary behavior:
- `L_s + L_t > limit`: skip.
- `L_s + L_t == limit`: keep.

Overflow policy is strict skip-only (no truncation fallback).

## Training Flow

Within `JointSDPOSelfDistillTrainer.fit_agent()`:
1. Generate trajectories and compute standard PPO quantities.
2. Run normal teacher actor update.
3. Prepare distillation payload (numerator/denominator batches + mask).
4. If payload has enough valid tokens, run one extra actor update using distill advantages.

Checkpoints remain the normal actor checkpoints (single shared model stream).

## Metrics

Primary distill metrics:
- `distill/sdpo_loss`
- `distill/token_count`
- `distill/skipped_context_overflow`
- `distill/kept_ratio`

Additional diagnostics:
- `distill/log_ratio_mean`
- `distill/log_ratio_std`
- `distill/skipped_batches`
- `distill_actor/*` (auxiliary actor update metrics)

## Tests

Added tests:
- `tests/test_sdpo_self_distill.py`
  - overflow guard (`>` vs `==`)
  - SDPO detach + masking behavior
  - mixed batch partial skip + metric counts
  - distill update execution gating
- `tests/test_agent_execution_engine_distill.py`
  - `assemble_steps` first-attempt mask alignment
  - mismatch mask zeroing behavior
  - token-mode `step_records` with episode-index alignment

Run in project environment:

```bash
conda activate icx
pytest -q tests/test_sdpo_self_distill.py tests/test_agent_execution_engine_distill.py
```

