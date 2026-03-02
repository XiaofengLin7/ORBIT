# SDPO Self-Distillation (v1)

## Overview

This document describes the implemented shared-weight SDPO self-distillation path used in multi-episode training.

At each training step:
1. Run the normal teacher PPO update on full rollout data.
2. Compute a distillation bonus on the first-attempt prefix sequence and merge it into the same actor update.

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
    teacher_regularization: none
    teacher_update_rate: 0.05
    teacher_update_interval: 10
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
- `teacher_regularization`: teacher policy source for denominator scoring.
  - `none`: denominator log-prob is computed from current actor.
  - `ema`: denominator log-prob is computed from a teacher snapshot updated by EMA after each actor step.
  - `every_n_steps`: denominator log-prob is computed from a teacher snapshot hard-copied from actor every `teacher_update_interval` steps.
- `teacher_update_rate`: EMA coefficient for `ema` mode (`[0, 1]`).
- `teacher_update_interval`: hard-sync interval `N` for `every_n_steps` mode (`>=1`).
- `context_limit`: if `null`, defaults to `data.max_prompt_length + data.max_response_length`.
- `context_overflow_policy`: currently supports only `skip_loss`.
- `min_distill_tokens`: skip auxiliary update when valid token count is below threshold.

Compatibility guard:
- Teacher regularization modes (`ema` and `every_n_steps`) are incompatible with KL reference-policy path (`algorithm.use_kl_in_reward` or `actor_rollout_ref.actor.use_kl_loss`) in v1.

## Rollout Metadata Requirements

To support distillation, token rollouts now include:
- `first_attempt_response_mask`: token-level mask selecting attempt-0 completion tokens.
- `step_records`: per-step list containing prompt/completion ids, logprobs, text fields, `episode_index`, and boundary metadata:
  - `boundary_transition`
  - `boundary_terminal_env_token_len`
  - `boundary_next_initial_env_token_len`

Additionally, multi-episode env info provides boundary observation split fields:
- `boundary_transition`
- `boundary_has_combined_observation`
- `boundary_terminal_observation`
- `boundary_next_initial_observation`

These are produced in `AgentExecutionEngine` token mode.

## Distillation Objective Used in Code

For valid tokens, compute:

- Numerator log-prob: `log πθ(a_t | x, y<t)`
- Denominator log-prob:
  - `teacher_regularization=none`: `log πθ(a_t | c_N, x/h_t context, y<t)` from current actor.
  - `teacher_regularization in {ema, every_n_steps}`: same conditioning, but scored by the teacher snapshot.
  - implementation input is `input_ids = [c_N ; prompts ; first_attempt_prefix]`, where `c_N` is the selected teacher hindsight prompt from transition-confirmed complete attempts
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
   - for each detected completion, use previous-step boundary metadata:
     - `delta = current_prompt_ids[len(accumulated):]`
     - require `boundary_terminal_env_token_len + boundary_next_initial_env_token_len == len(delta)`
     - candidate context is `accumulated + delta[:boundary_terminal_env_token_len]`
   - this keeps terminal boundary tokens from attempt `N` and excludes `(N+1)` initial-observation tokens
   - if `teacher_context_attempts = N`, select candidate `N`
   - if `teacher_context_attempts = null`, select the last candidate (all complete attempts)
   - trailing partial attempt is excluded because it has no later episode transition
   - malformed/missing boundary metadata at transition marks hindsight unavailable and skips distillation for that sample
2. Extract first-attempt prefix response sequence from batch masks:
   - find `last_first_attempt_idx = max{i | first_attempt_response_mask[i]==1 and response_mask[i]==1}`
   - set `first_attempt_prefix = responses[: last_first_attempt_idx + 1]`
   - let `L_prefix = len(first_attempt_prefix)`
3. Estimate denominator scoring length:
   - extract valid prompt tokens from numerator prompt segment using prompt-side attention mask
   - build denominator prompt tokens as `concat(selected_teacher_context_tokens, valid_prompt_tokens)`
   - `L_den = len(selected_teacher_context_tokens) + len(valid_prompt_tokens) + L_prefix`
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
2. Prepare distillation payload (denominator batch + mask + kept-row mapping) after trajectory filtering.
3. Compute a detached distillation bonus and add it to PPO actor advantages.
4. Run one actor update on the merged advantages (single shared update call).
5. If teacher regularization is enabled:
   - `ema`: update teacher snapshot by EMA after each actor step.
   - `every_n_steps`: hard-sync teacher snapshot after steps where `global_steps % teacher_update_interval == 0`.

Teacher initialization:
- After checkpoint load, teacher snapshot is hard-synced from actor once when teacher regularization is enabled.

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
- `distill/teacher_sync_applied`

## Tests

Added tests:
- `tests/test_sdpo_self_distill.py`
  - first-attempt prefix extraction correctness (last-`1` index, not mask sum)
  - first-N complete-attempt hindsight reconstruction from raw step tokens
  - malformed hindsight/boundary reconstruction skip path
  - denominator overflow guard (`L_den > limit` vs `==`)
  - SDPO detach + masking behavior
  - mixed batch partial skip + metric counts
  - teacher-regularization settings validation and KL-incompatibility guard
  - denominator teacher log-prob routing and teacher sync schedule hooks
- `tests/test_multi_episode_env.py`
  - boundary metadata emission for non-reflection combined observation
  - boundary metadata emission for reflection reset transition
- `tests/test_agent_execution_engine_distill.py`
  - `assemble_steps` first-attempt mask alignment
  - mismatch mask zeroing behavior
  - token-mode `step_records` with episode-index alignment

Run in project environment:

```bash
conda activate icx
pytest -q tests/test_sdpo_self_distill.py tests/test_agent_execution_engine_distill.py
```
