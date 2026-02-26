# SDPO Self-Distillation (v1)

## Overview

This document describes the implemented shared-weight SDPO self-distillation path used in multi-episode training.

At each training step:
1. Run the normal teacher PPO update on full rollout data.
2. Run one auxiliary distillation actor update on the first-attempt prefix sequence, with loss applied on first-attempt model tokens only.

Distillation can be skipped when context-overflow is triggered, hindsight context is unavailable, or the batch has insufficient valid distill tokens.

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
    teacher_context_attempts: null
    context_limit: null
    context_overflow_policy: skip_loss
    min_distill_tokens: 1
```

Semantics:
- `enable`: turn on SDPO auxiliary update.
- `lambda`: scales distill advantages.
- `mode`: currently supports `sdpo_self`.
- `denominator_mode`: currently `teacher_adapted_feedback`.
- `teacher_context_attempts`: if set to `N`, require at least `N` transition-confirmed complete attempts and use only the first `N`; if `null`, use all transition-confirmed complete attempts.
- `context_limit`: if `null`, defaults to `data.max_prompt_length + data.max_response_length`.
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
- Denominator log-prob: `log πθ(a_t | c_N, y<t)` where `c_N` is the selected teacher hindsight prompt from transition-confirmed complete attempts (practical prompt-context approximation)
- Coefficient:
  - `c_t = stopgrad(logp_num - logp_den)`
- Distill advantage:
  - `A_t^distill = lambda * c_t`

Scoring and masking:
- Distill `responses` are the first-attempt prefix sequence: response-token prefix ending at the last index where `first_attempt_response_mask==1`.
- This prefix keeps interleaved env/model tokens that appear before that endpoint.
- Distill loss applies only where `response_mask=1` and `first_attempt_response_mask=1`.
- Invalid token-mismatch trajectories are fully masked by engine mismatch filtering.

## Context-Overflow Guard

Per sample:

1. Reconstruct teacher hindsight context from raw `step_records` using transition-confirmed complete attempts:
   - validate cumulative token consistency across steps
   - detect completion only when `episode_index` increases between adjacent steps
   - each detected completion contributes a candidate context equal to the next step `prompt_ids`
   - if `teacher_context_attempts = N`, select candidate `N`
   - if `teacher_context_attempts = null`, select the last candidate (all complete attempts)
   - trailing partial attempt is excluded because it has no later episode transition
2. Extract first-attempt prefix response sequence from batch masks:
   - find `last_first_attempt_idx = max{i | first_attempt_response_mask[i]==1 and response_mask[i]==1}`
   - set `first_attempt_prefix = responses[: last_first_attempt_idx + 1]`
   - let `L_prefix = len(first_attempt_prefix)`
3. Estimate denominator scoring length:
   - `L_den = len(selected_teacher_context_tokens) + L_prefix`
4. Determine limit:
   - `limit = rllm.distill.context_limit or (data.max_prompt_length + data.max_response_length)`
5. Guard:
   - if `L_den > limit`, skip distillation for this sample.
   - if hindsight reconstruction is invalid/malformed or complete attempts are insufficient, skip distillation for this sample.

Boundary behavior:
- `L_den > limit`: skip.
- `L_den == limit`: keep.

Overflow policy is strict skip-only (no truncation fallback), while preserving raw trajectory tokens for teacher context.

Note:
- The method target is exact reverse-KL.
- The implementation uses the existing practical SDPO-style detached log-ratio surrogate (`compute_sdpo_advantages`) and does not add exact per-token reverse-KL recomputation.

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
- `distill/skipped_hindsight_unavailable`
- `distill/kept_ratio`

Additional diagnostics:
- `distill/log_ratio_mean`
- `distill/log_ratio_std`
- `distill/skipped_batches`
- `distill_actor/*` (auxiliary actor update metrics)

## Tests

Added tests:
- `tests/test_sdpo_self_distill.py`
  - first-attempt prefix extraction correctness (last-`1` index, not mask sum)
  - first-N complete-attempt hindsight reconstruction from raw step tokens
  - malformed hindsight reconstruction skip path
  - denominator overflow guard (`L_den > limit` vs `==`)
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
