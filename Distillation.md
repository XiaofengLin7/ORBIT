# SDPO Self-Distillation

## Overview

This document describes the shared-weight SDPO self-distillation path used in multi-episode training.

Two distillation modes are supported:

### Direct-loss mode (default, `merge_to_advantages=false`)

At each training step:
1. Run normal PPO preparation on rollout data (reward, old log-prob, advantages, etc.).
2. Build a distillation payload from hindsight context + first-attempt tokens.
3. Run a single actor update with combined loss:
   - `total_actor_loss = ppo_pg_loss + lambda * sdpo_loss`
   - existing entropy / KL terms are still applied when enabled.

The SDPO loss is aggregated over first-attempt tokens only via `distill_mask`.

### Advantage-bonus mode (`merge_to_advantages=true`)

Reproduces the March 2nd architecture where the SDPO signal is added directly to PPO advantages before the standard actor update:
1. Run normal PPO preparation on rollout data.
2. Build a distillation payload from hindsight context + first-attempt tokens.
3. Compute bonus: `λ * stopgrad(s_old - t_old) * distill_mask`, scatter into full-batch advantages.
4. Run a standard PPO actor update (no SDPO forward passes in the actor).

The bonus is aggregated over ALL response tokens via `response_mask` in `agg_loss`, which naturally applies per-sequence F/T dilution (first-attempt tokens / total response tokens). This gives variable per-sequence weighting that a fixed lambda in direct-loss mode cannot replicate.

## Scope

- Current scope is trajectory-mode training (`stepwise_advantage.enable = false`).
- Distillation token scope is fixed to first-attempt tokens only (no token-scope config).

Key implementation files:
- `trainers/sdpo_actor.py`
- `trainers/sdpo_self_distill_trainer.py`
- `trainers/teacher_regularized_workers.py`
- `trainers/train_multi_episode.py`

## Configuration

Distillation is configured under `rllm.distill`.

```yaml
rllm:
  distill:
    enable: true
    lambda: 0.1
    mode: sdpo_self
    denominator_mode: teacher_adapted_feedback
    trajectory_selection: first_attempt_hindsight
    teacher_context_attempts: null
    teacher_regularization: none
    teacher_update_rate: 0.05
    teacher_update_interval: 10
    context_limit: null
    context_overflow_policy: skip_loss
    min_distill_tokens: 1

    # SDPO loss controls
    loss_variant: non_full        # non_full | full_logit
    alpha: 1.0                    # [0,1], must be 1.0 for non_full
    is_clip: false                # bool — uses GRPO clip_ratio_low/high when true
    full_logit_topk: 64           # positive int
    full_logit_add_tail: true     # bool
    negate_sdpo_loss: false       # bool
    use_stale_coefficient: false  # bool
    strip_system_from_teacher_prompt: true  # bool

    # Distillation mode
    merge_to_advantages: false    # bool — false: direct loss, true: advantage bonus
```

Semantics:
- `enable`: enable SDPO distillation path.
- `lambda`: scalar weight for SDPO term. In direct-loss mode, scales the SDPO loss added to the actor objective. In advantage-bonus mode, scales the bonus added to PPO advantages.
- `mode`: `sdpo_self` (self-distillation, shared-weight model acts as both student and teacher) or `sdpo_pure` (separate teacher model).
- `denominator_mode`: currently `teacher_adapted_feedback`.
- `trajectory_selection`: how to select which trajectory tokens to distill. Currently `first_attempt_hindsight` (distill first-attempt tokens with hindsight context from complete attempts).
- `teacher_context_attempts`:
  - integer `N`: require at least `N` complete attempts and use the first `N`.
  - `null`: use last available complete-attempt context.
- `teacher_regularization`: `none` (actor is its own teacher), `ema` (exponential moving average of actor weights), or `every_n_steps` (hard-sync teacher from actor every N steps).
- `teacher_update_rate`: EMA decay rate when `teacher_regularization=ema`. Lower values = slower teacher drift.
- `teacher_update_interval`: hard-sync interval (in training steps) when `teacher_regularization=every_n_steps`.
- `context_limit`: max total tokens for denominator context. If `null`, defaults to `data.max_prompt_length + data.max_response_length`.
- `context_overflow_policy`: currently only `skip_loss` — skip samples that exceed context limit.
- `min_distill_tokens`: if valid distill tokens < this value, distillation is disabled for that batch.
- `loss_variant`: SDPO branch (`non_full` or `full_logit`). Only applies to direct-loss mode; advantage-bonus mode always uses `non_full`-style log-ratio.
- `alpha`: KL/JSD interpolation for `full_logit`; for `non_full`, required to be `1.0`.
- `is_clip`: bool — when `true`, applies PPO-style pessimistic clipping using the actor's `clip_ratio_low`/`clip_ratio_high` (shared with GRPO). Direct-loss mode only.
- `full_logit_topk`: top-k support for full-logit branch.
- `full_logit_add_tail`: whether to add tail bucket in top-k mode.
- `negate_sdpo_loss`: if `true`, negate the SDPO loss/bonus sign. In direct-loss mode this flips the gradient direction. In advantage-bonus mode, set to `false` since the bonus is naturally anti-distillation (`bonus = λ*(s-t)`, PG loss `= -A*logπ` reinforces tokens where student > teacher).
- `use_stale_coefficient`: if `true`, apply importance-sampling correction using the ratio between current and old policy log-probs in the SDPO term.
- `strip_system_from_teacher_prompt`: if `true`, remove the system prompt from the teacher's denominator context. Set to `false` to keep the system prompt in teacher scoring.
- `merge_to_advantages`: if `false` (default), run SDPO as a separate loss in the actor. If `true`, compute SDPO bonus in the trainer, add to PPO advantages, and run standard PPO — the actor sees `distill_enabled=false` and does no SDPO forward passes.

Not added by design:
- no `rllm.distill.token_scope`
- no distill variant logging key
- no empty-target-batch logging key

Compatibility guard:
- `teacher_regularization != none` is incompatible with KL reference-policy path:
  - `algorithm.use_kl_in_reward`
  - `actor_rollout_ref.actor.use_kl_loss`

## Distillation Targets And Mask

Distillation is only applied to first-attempt model tokens:
- `distill_mask = response_mask * first_attempt_response_mask`
- first-attempt prefix is extracted up to the last valid first-attempt token index.
- interleaved env/model tokens before that endpoint remain in the scored prefix.

If no valid first-attempt token exists, that sample is skipped for distillation.

## Hindsight / Denominator Construction

Denominator context construction is unchanged from earlier logic:
- reconstruct complete attempts from `step_records` using episode transitions + boundary metadata.
- build denominator prompt as `concat(hindsight_context_tokens, student_prompt_tokens)`.
- denominator response is the first-attempt prefix response sequence.

Context overflow guard:
- `L_den = len(denominator_prompt_tokens) + len(first_attempt_prefix_tokens)`
- if `L_den > context_limit`: skip sample (`distill/skipped_context_overflow`).
- if hindsight unavailable/malformed/insufficient: skip sample (`distill/skipped_hindsight_unavailable`).
- `L_den == context_limit` is kept.

## Loss Formulation

### Direct-loss mode (`merge_to_advantages=false`)

Actor wrapper computes PPO and SDPO in the same optimizer step.

#### Combined objective

- `policy_loss = ppo_pg_loss + lambda * sdpo_loss`
- if enabled:
  - entropy term: `policy_loss -= entropy_coeff * entropy`
  - KL term: `policy_loss += kl_loss_coef * kl_loss`

#### SDPO branch: `non_full`

Per-token:
- `per_token = (student_logp - teacher_logp).detach() * student_logp`

If `negate_sdpo_loss=true`: `per_token = -per_token`.

This matches the non-full branch in `../SDPO`.

#### SDPO branch: `full_logit`

Uses full-logit KL/JSD-style distillation:
- `alpha = 0`: forward KL branch
- `alpha = 1`: reverse KL branch
- `0 < alpha < 1`: generalized JSD-style mixture branch

Top-k mode:
- gather student top-k logits
- gather teacher logits at the same student-selected indices
- optional tail bucket (`full_logit_add_tail=true`) or renormalization

If `negate_sdpo_loss=true`: `per_token = -per_token` (applied after branch computation).

#### Optional IS clip (PPO-style)

If `is_clip=true`, applies PPO-style pessimistic clipping using the actor's GRPO clip ratios:
- `ratio = exp(clamp((student_logp - old_logp).detach(), -20, 20))`
- `clipped_ratio = clamp(ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)`
- `per_token = max(per_token * ratio, per_token * clipped_ratio)` (pessimistic selection)

If rollout IS weights exist in batch, they are also multiplied into distill per-token loss.

#### Aggregation

`sdpo_loss = masked_mean(per_token, distill_mask)` — mean over first-attempt tokens only.

### Advantage-bonus mode (`merge_to_advantages=true`)

The SDPO signal is computed in the trainer and added to PPO advantages. The actor runs standard PPO — no SDPO forward passes, `distill_enabled=false`.

#### Bonus computation

For each kept sample `i` and token position `t` within the first-attempt mask:

```
bonus[i,t] = λ * stopgrad(s_old[i,t] - t_old[i,t]) * distill_mask[i,t]
```

Where:
- `s_old` = student (rollout) log-probs from `old_log_probs` in the batch
- `t_old` = teacher log-probs from one forward pass on denominator context
- `distill_mask` = first-attempt token mask

The bonus is scattered into a full-batch tensor and added to advantages:
```
advantages_combined = advantages_ppo + bonus
```

#### Effective PPO loss with bonus

Standard PPO PG loss with `loss_agg_mode=seq-mean-token-mean`:

```
ratio[i,t] = exp(log π_θ(a[i,t]) - log π_old(a[i,t]))
pg_loss[i,t] = -advantages_combined[i,t] * clip(ratio[i,t])
L = mean_i( mean_t( pg_loss[i,t] * response_mask[i,t] ) / sum_t(response_mask[i,t]) )
```

Expanding the bonus contribution at token `(i,t)`:

```
L_bonus[i,t] = -(λ * (s_old - t_old) * distill_mask) * clip(ratio)
```

With `ppo_epoch=1` and `ratio ≈ 1`:

```
L_bonus[i,t] ≈ -λ * (s_old[i,t] - t_old[i,t]) * log π_θ(a[i,t]) * distill_mask[i,t]
```

This is equivalent to the `non_full` per-token loss but aggregated differently.

#### Aggregation difference (key distinction from direct-loss)

Because `bonus` is part of `advantages_combined`, it is aggregated over `response_mask` (ALL response tokens), not `distill_mask` (first-attempt tokens only):

- **Per-sequence dilution**: `F_j / T_j` where `F_j` = first-attempt tokens, `T_j` = total response tokens in sequence `j`. Sequences with short first attempts get proportionally less SDPO influence.
- **Batch-level dilution**: mean over all `B` sequences (non-kept sequences contribute 0 bonus but still appear in the denominator).
- Effective lambda: `λ_eff ≈ λ * (K/B) * mean_j(F_j/T_j)`, typically 5–20% of the nominal `λ`.

The direct-loss path uses `masked_mean(sdpo_loss, distill_mask)`, which averages only over first-attempt tokens of kept sequences — no dilution.

#### `negate_sdpo_loss` does not apply

In advantage-bonus mode, `negate_sdpo_loss` has no effect — it is only used in the actor's SDPO loss branch. The bonus sign is controlled directly: `bonus = λ * (s_old - t_old)` is naturally anti-distillation (reinforces tokens where student > teacher). To get pro-distillation behavior, use the direct-loss mode with `negate_sdpo_loss=true` instead.

## Data Flow

### Direct-loss mode (`merge_to_advantages=false`)

1. Trainer generates trajectories and computes PPO prep tensors.
2. Trainer builds distill payload from current hindsight logic.
3. Trainer attaches distill tensors to batch:
   - `distill_teacher_input_ids`
   - `distill_teacher_attention_mask`
   - `distill_teacher_position_ids`
   - `distill_teacher_responses`
   - `distill_mask`
   - distill meta settings (`distill_*` in `batch.meta_info`)
4. Actor wrapper computes standard PPO pg loss.
5. Actor wrapper computes teacher-side scores (no grad) on denominator context.
6. Actor wrapper computes SDPO loss branch (`non_full` or `full_logit`) on first-attempt mask.
7. Actor wrapper backprops one combined loss (`ppo + lambda*sdpo`) per micro-batch.
8. Teacher sync schedule (`none` / `ema` / `every_n_steps`) remains unchanged.

### Advantage-bonus mode (`merge_to_advantages=true`)

1. Trainer generates trajectories and computes PPO prep tensors.
2. Trainer builds distill payload from current hindsight logic.
3. Trainer computes teacher log-probs via one forward pass (using teacher or actor module).
4. Trainer computes bonus: `λ * stopgrad(s_old - t_old) * distill_mask`, scatters into full-batch tensor.
5. Trainer adds bonus to `batch["advantages"]`.
6. Trainer sets `distill_enabled=false` in batch meta — actor runs standard PPO with no distill tensors.
7. Actor backprops standard PPO loss (advantages already contain SDPO signal).
8. Teacher sync schedule (`none` / `ema` / `every_n_steps`) remains unchanged.

## Worker Wiring

- Distill-enabled runs use local SDPO actor worker wrappers.
- For `teacher_regularization = none`: SDPO actor uses current actor module as teacher scorer.
- For `ema` / `every_n_steps`: teacher-regularized worker hosts ref snapshot and SDPO actor scores with that teacher module.
- No `third_party` files are modified.

## Metrics

Kept distill metrics:
- `distill/sdpo_loss`
- `distill/token_count`
- `distill/skipped_context_overflow`
- `distill/skipped_hindsight_unavailable`
- `distill/kept_ratio`
- `distill/teacher_sync_applied`

Additional metrics in advantage-bonus mode (`merge_to_advantages=true`):
- `distill/log_ratio_mean`
- `distill/log_ratio_std`
- `distill/skipped_batches`

Not logged:
- `distill/variant`
- `distill/empty_target_batch`

## Tests

Current tests covering this path:
- `tests/test_sdpo_self_distill.py`
  - hindsight reconstruction and boundary handling
  - first-attempt prefix extraction and overflow skipping
  - distill payload attachment and config validation
  - teacher sync schedule hooks
- `tests/test_sdpo_actor_loss.py`
  - SDPO formula checks for `non_full`
  - full-logit forward/reverse KL and JSD/top-k+tail behavior
- `tests/test_agent_execution_engine_distill.py`
  - rollout/assembly metadata for distillation masking and step records

Run:

```bash
conda activate icx
pytest -q tests/test_sdpo_self_distill.py tests/test_agent_execution_engine_distill.py tests/test_sdpo_actor_loss.py
```
