# SDPO Self-Distillation

## Quick Start

Self-distillation training is launched via task-specific shell scripts. Each script sets environment variables that map to `rllm.distill.*` Hydra overrides.

**Prerequisites:**
- `conda activate icx`
- 2+ GPUs (scripts default to 2 GPUs per node)
- A model checkpoint or HuggingFace model path (defaults to `Qwen/Qwen3-4B`)

**Available scripts:**

| Script | Environment | Default task config |
|---|---|---|
| `scripts/train_frozenlake_self_distill_multi_episode.sh` | FrozenLake | `configs/multi_task_frozenlake_self_distill_multi_episode.yaml` |
| `scripts/train_sokoban_self_distill_multi_episode.sh` | Sokoban | `configs/multi_task_sokoban_self_distill_multi_episode.yaml` |
| `scripts/train_towerofhanoi_self_distill_multi_episode.sh` | Tower of Hanoi | (single-task, env_id-based) |

**Basic usage (defaults: `selective_retry_success_n2`, EMA teacher, `non_full` loss):**

```bash
conda activate icx
bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

**Customizing via environment variables:**

```bash
# Use a different model and trajectory selection strategy
MODEL_PATH=Qwen/Qwen3-1.7B \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
DISTILL_LAMBDA=0.05 \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh

# Full-logit KL distillation with JSD (alpha=0.5)
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=0.5 \
  bash scripts/train_sokoban_self_distill_multi_episode.sh
```

**Key environment variables** (all have defaults in each script):

| Env var | Maps to | Description |
|---|---|---|
| `DISTILL_LAMBDA` | `rllm.distill.lambda` | SDPO loss weight |
| `DISTILL_MODE` | `rllm.distill.mode` | `sdpo_self` (PPO+SDPO) or `sdpo_pure` (SDPO only) |
| `DISTILL_TRAJECTORY_SELECTION` | `rllm.distill.trajectory_selection` | Which trajectories to distill (see Trajectory Selection Strategies) |
| `TEACHER_CONTEXT_ATTEMPTS` | `rllm.distill.teacher_context_attempts` | Number of complete attempts for teacher context (`null` = all) |
| `TEACHER_REGULARIZATION` | `rllm.distill.teacher_regularization` | `none`, `ema`, or `every_n_steps` |
| `DISTILL_LOSS_VARIANT` | `rllm.distill.loss_variant` | `non_full` or `full_logit` |
| `DISTILL_ALPHA` | `rllm.distill.alpha` | KL interpolation (must be 1.0 for `non_full`) |
| `DISTILL_IS_CLIP` | `rllm.distill.is_clip` | PPO-style clipping on SDPO term |
| `NEGATE_SDPO_LOSS` | `rllm.distill.negate_sdpo_loss` | Flip SDPO gradient direction |
| `MERGE_TO_ADVANTAGES` | `rllm.distill.merge_to_advantages` | Advantage-bonus mode instead of direct loss |

Any extra Hydra overrides can be appended as positional arguments to the script.

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
    trajectory_selection: first_attempt_hindsight
    teacher_context_attempts: null
    teacher_regularization: none
    teacher_update_rate: 0.05
    teacher_update_interval: 10
    context_limit: null
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
- `mode`: `sdpo_self` (PPO + SDPO combined loss) or `sdpo_pure` (SDPO loss only, no PPO PG loss component). Both use the same shared-weight self-distillation architecture — the only difference is whether the PPO policy gradient term is included in the actor objective.
- `trajectory_selection`: how to select which trajectory tokens to distill.
  - `first_attempt_hindsight`: distill first-attempt tokens with hindsight context from complete attempts.
  - `first_attempt_latest_success_hindsight`: distill first-attempt tokens with teacher context set to the isolated latest successful attempt.
  - `first_attempt_latest_success_hindsight_first_failure_only`: same as `first_attempt_latest_success_hindsight`, but only when the first completed attempt failed.
  - `selective_retry_success_n2`: distill retry-success trajectories for the 2-attempt selective gate.
- `teacher_context_attempts`:
  - integer `N`: require at least `N` complete attempts and use the first `N`.
  - `null`: use last available complete-attempt context.
- `teacher_regularization`: `none` (actor is its own teacher), `ema` (exponential moving average of actor weights), or `every_n_steps` (hard-sync teacher from actor every N steps).
- `teacher_update_rate`: EMA decay rate when `teacher_regularization=ema`. Lower values = slower teacher drift.
- `teacher_update_interval`: hard-sync interval (in training steps) when `teacher_regularization=every_n_steps`.
- `context_limit`: max total tokens for denominator context. If `null`, defaults to `data.max_prompt_length + data.max_response_length`. Samples that exceed this limit are skipped (skip-loss policy).
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
- for `first_attempt_latest_success_hindsight_first_failure_only`, samples are further gated to those with a failed first completed attempt and at least one later completed successful attempt; the teacher context remains the isolated latest successful attempt only.

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

Uses a cumulative future log-ratio advantage (REINFORCE-style):

```
delta[t] = (student_logp[t] - teacher_logp[t]).detach()
advantage[t] = sum_{t'=t}^{end_of_segment} delta[t']     # reverse cumsum within mask segments
per_token[t] = advantage[t] * student_logp[t]
```

The reverse cumsum (`_segmented_reverse_cumsum`) operates within contiguous segments of `distill_mask`, so each segment (e.g., separate first-attempt spans) gets its own independent cumulative advantage. `advantage[t]` represents the total future log-ratio sum from position `t` to the end of its segment.

If `use_stale_coefficient=true`, the delta uses pre-computed old log-probs (`stale_s - stale_t`) instead of fresh student/teacher log-probs, with gradient flowing only through the fresh `student_logp` multiplier.

This is the REINFORCE estimator for the gradient of the reverse KL $\text{KL}(\text{student} \| \text{teacher})$ (see `.codex/method.md`). Minimizing this loss pulls the student toward the teacher.

If `negate_sdpo_loss=true`: `per_token = -per_token` (flips gradient direction to push student away from teacher).

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

## Trajectory Selection Strategies

The `rllm.distill.trajectory_selection` config key controls **which samples are eligible for distillation** and **how the teacher's denominator context is constructed**. All strategies operate per-sample inside `_prepare_distill_batch`. Four strategies are currently implemented:

### 1. `first_attempt_hindsight` (default)

**What is distilled:** First-attempt model tokens (episode 0 response tokens).

**Teacher context:** First-N complete attempts, concatenated in chronological order. Controlled by `teacher_context_attempts`:
- `null` → use all transition-confirmed complete attempts (context from the last completed attempt, which is cumulative).
- integer `N` → require at least N complete attempts and use the first N (i.e., cumulative context through attempt N).

**Gating:** None — every sample with valid first-attempt tokens and sufficient hindsight is kept.

**Denominator prompt:** `cat(system_prompt (optional), hindsight_context_tokens, user_prompt_tokens)`, where hindsight context is the accumulated token sequence through the selected complete attempts.

**Intuition:** The teacher sees how the trajectory played out over N attempts, then re-scores the student's first-attempt tokens with that hindsight. This is the broadest strategy — it distills every trajectory that has at least one episode boundary.

### 2. `first_attempt_latest_success_hindsight`

**What is distilled:** First-attempt model tokens (same as above).

**Teacher context:** The **isolated** latest successful complete attempt only — not cumulative context. The isolation is computed by subtracting the previous attempt's cumulative context prefix, yielding only the tokens from the selected successful episode.

**Gating:** The trajectory must contain at least one completed successful attempt. If no successful attempt exists, the sample is skipped (`skipped_hindsight_teacher_context_unavailable`).

**Denominator prompt:** `cat(system_prompt (optional), isolated_success_attempt_tokens, user_prompt_tokens)`. The system prompt is stripped from the student prompt and optionally re-prepended separately.

**Intuition:** Rather than showing the teacher the entire trajectory history, show it only the best (latest) successful attempt. This gives a cleaner teaching signal — the teacher scores first-attempt tokens conditioned on "here is exactly what a successful attempt looks like" without noise from intermediate failed attempts.

### 3. `first_attempt_latest_success_hindsight_first_failure_only`

**What is distilled:** First-attempt model tokens, but only for trajectories where the first completed attempt **failed**.

**Teacher context:** Same as `first_attempt_latest_success_hindsight` — the isolated latest successful attempt.

**Gating (two conditions, both required):**
1. The first completed attempt (episode 0) must have **failed** (`success=False`). If it succeeded, the sample is skipped (`skipped_selective_gate`).
2. At least one later completed attempt (episode index > 0) must have **succeeded**. If no later success exists, the sample is skipped (`skipped_selective_gate`).

**Denominator prompt:** Same construction as strategy 2.

**Intuition:** This is the most selective first-attempt strategy. It targets the specific case where the agent initially failed but later figured it out — exactly the trajectories where hindsight distillation is most informative. No loss is applied to trajectories where the first attempt already succeeded (nothing to teach) or where the agent never succeeded (no good teacher signal available).

### 4. `selective_retry_success_n2`

**What is distilled:** **Second-attempt** (episode 1) response tokens — not first-attempt tokens. This is the only strategy that distills non-first-attempt tokens.

**Teacher context:** The first completed attempt's cumulative context tokens (episode 0 context).

**Gating (two conditions, both required):**
1. The first completed attempt (episode 0) must have **failed**.
2. The second completed attempt (episode 1) must have **succeeded**.

Requires `teacher_context_attempts=1` (enforced at config validation).

**Denominator prompt:** `cat(first_attempt_context_tokens, student_prompt_tokens)`. The teacher sees the failed first attempt and re-scores the student's successful retry.

**Intuition:** Instead of teaching the agent to do better on its first try, this strategy reinforces successful retry behavior. When the agent fails then succeeds, we distill the successful second attempt with the first attempt as context. This teaches the model "given that you saw a failure, here's how to produce a successful retry."

### Strategy Comparison

| Property | `first_attempt_hindsight` | `latest_success_hindsight` | `latest_success_first_failure_only` | `selective_retry_success_n2` |
|---|---|---|---|---|
| **Distill target** | Episode 0 tokens | Episode 0 tokens | Episode 0 tokens | Episode 1 tokens |
| **Teacher context** | First-N cumulative attempts | Isolated latest success | Isolated latest success | Episode 0 context |
| **Requires success** | No | Yes (any episode) | Yes (episode > 0) | Yes (episode 1) |
| **Requires failure** | No | No | Yes (episode 0) | Yes (episode 0) |
| **Selectivity** | Low (all valid samples) | Medium | High | High |
| **Uses `teacher_context_attempts`** | Yes | No | No | Must be 1 |
| **System prompt handling** | `strip_system_from_teacher_prompt` | Always splits system prompt | Always splits system prompt | `strip_system_from_teacher_prompt` |
