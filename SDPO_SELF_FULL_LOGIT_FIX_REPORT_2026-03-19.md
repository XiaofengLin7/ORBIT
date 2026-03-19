# SDPO Self + Full-Logit Review

Date: 2026-03-19

Scope:
- `trainers/sdpo_self_distill_trainer.py`
- `trainers/sdpo_actor.py`
- `tests/test_sdpo_actor_loss.py`
- `tests/test_sdpo_self_distill.py`
- `.codex/method.md`

## Executive Summary

The core `sdpo_self + full_logit` path is mostly coherent when used in the narrow configuration:

- `rllm.distill.mode=sdpo_self`
- `rllm.distill.loss_variant=full_logit`
- `rllm.distill.merge_to_advantages=false`
- `rllm.distill.use_stale_coefficient=false`
- usually `rllm.distill.is_clip=false`

In that regime, the implementation does the intended thing:

1. Roll out with the current policy.
2. Compute GRPO/PPO advantages on the rollout batch.
3. Build student/teacher distillation inputs from the same rollout.
4. In the actor update, compute GRPO loss and full-logit SDPO loss on each microbatch.
5. Sum them into one scalar loss and update the shared actor weights once per minibatch.

There are still several correctness issues or configuration traps that should be fixed before relying on the mode.

## Priority Fixes

### P0. `merge_to_advantages=true` silently bypasses `full_logit`

Files:
- `trainers/sdpo_self_distill_trainer.py`

Problem:
- When `merge_to_advantages=true`, the trainer does not call actor-side SDPO.
- Instead it computes a detached bonus from token log-ratios and adds that bonus into `advantages`.
- That path does not use `loss_variant`, `alpha`, `full_logit_topk`, or `full_logit_add_tail`.

Code path:
- `_prepare_distill_payload(...)`
- `_compute_distill_bonus(...)`
- branch at `if self.distill_settings.merge_to_advantages: ...`

Why this is wrong:
- For `loss_variant=full_logit`, the intended objective is a KL/JSD over distributions.
- The merged-advantage path is only a token log-prob ratio bonus.
- So `sdpo_self + full_logit + merge_to_advantages=true` is not implementing full-logit SDPO at all.

Fix:
- Add a hard validation error when:
  - `merge_to_advantages=true` and `loss_variant != "non_full"`
- Optional stronger validation:
  - also forbid `merge_to_advantages=true` with `is_clip`, `use_stale_coefficient`, or any future flag that only exists in actor-side SDPO.

Suggested patch location:
- `JointSDPOSelfDistillTrainer._load_distill_settings()`

Suggested test:
- add a unit test in `tests/test_sdpo_self_distill.py` asserting a `ValueError` for:
  - `{"merge_to_advantages": True, "loss_variant": "full_logit"}`

### P0. `use_stale_coefficient=true` is parsed for `full_logit` but ignored

Files:
- `trainers/sdpo_self_distill_trainer.py`
- `trainers/sdpo_actor.py`

Problem:
- The flag is loaded and propagated for all SDPO modes.
- The trainer computes and attaches `distill_student_old_log_probs` and `distill_teacher_old_log_probs`.
- In the actor, stale-coefficient logic is only used in the `non_full` branch.
- The `full_logit` branch ignores it completely.

Why this is wrong:
- This is a silent no-op plus extra compute.
- It makes configuration behavior misleading.

Fix:
- Add a hard validation error when:
  - `use_stale_coefficient=true` and `loss_variant != "non_full"`

Suggested patch location:
- `JointSDPOSelfDistillTrainer._load_distill_settings()`

Suggested test:
- add a unit test in `tests/test_sdpo_self_distill.py` asserting a `ValueError` for:
  - `{"use_stale_coefficient": True, "loss_variant": "full_logit"}`

## Important but Non-Blocking Issues

### P1. The method note says "two updates", but the code does one joint update

Files:
- `.codex/method.md`
- `trainers/sdpo_actor.py`

Problem:
- The method note describes:
  1. teacher RL update
  2. student self-distillation update
- The implementation instead optimizes a single combined actor loss:
  - `pg_loss + lambda * sdpo_loss`

Why this matters:
- With Adam or any non-linear optimizer, one joint update is not mathematically identical to two sequential optimizer updates.
- The code is valid, but the document is inaccurate.

Fix options:
- Option A: update the method note to say the current code uses a joint objective in one actor update.
- Option B: change the implementation to perform two separate optimizer steps.

Recommendation:
- Prefer Option A unless there is a strong research reason to insist on two sequential steps.

### P1. `full_logit + is_clip=true` is a heuristic stabilization, not an exact clipped full-logit objective

Files:
- `trainers/sdpo_actor.py`

Problem:
- In `full_logit`, the per-token SDPO term is a KL/JSD over distributions.
- If `is_clip` is enabled, the code applies a PPO-style ratio derived from sampled-token log-probs to that scalar per-token distribution loss.

Why this matters:
- This is not the exact clipped version of the KL/JSD objective.
- It may still be useful as a practical stabilizer, but it should be treated as an approximation.

Fix options:
- Option A: document it explicitly as a heuristic and keep it.
- Option B: forbid `is_clip` with `full_logit`.
- Option C: derive and implement a principled clipped full-logit surrogate.

Recommendation:
- Short term: either document or forbid.
- Do not leave it ambiguous.

### P1. `is_clip` is parsed as numeric in the trainer but collapsed to `bool` before the actor sees it

Files:
- `trainers/sdpo_self_distill_trainer.py`
- `trainers/sdpo_actor.py`
- `tests/test_sdpo_self_distill.py`

Problem:
- Trainer settings allow values like `is_clip=2.0`.
- Metadata writes `distill_is_clip = bool(self.distill_settings.is_clip)`.
- Actor parses it as `bool(...)`.
- So numeric values are reduced to "enabled / disabled" and do not carry numeric semantics downstream.

Why this matters:
- The config interface suggests more expressive behavior than the actor actually supports.
- This is easy to misread during experiments.

Fix options:
- Option A: normalize `is_clip` to plain boolean in config parsing and update tests/docs.
- Option B: preserve numeric semantics end-to-end and make the actor use them.

Recommendation:
- If no numeric semantics are intended, simplify to `bool`.

## Core Path Check

This is the step-by-step review for the intended path:

- Config mode:
  - `mode="sdpo_self"` correctly leaves GRPO enabled.
  - `use_grpo_loss = mode == "sdpo_self"`.

- Distillation data construction:
  - student side uses the first-attempt prefix
  - teacher side scores the same response tokens under hindsight context
  - this matches the intended first-attempt self-distillation setup

- Full-logit actor loss:
  - student forward is trainable
  - teacher forward is under `torch.no_grad()`
  - if `full_logit_topk > 0`, student top-k indices are reused for teacher gathering
  - optional tail bucket is added and normalized
  - `alpha=1` gives reverse KL
  - `alpha=0` gives forward KL
  - `alpha in (0, 1)` gives the interpolated JS-style loss

- Reduction:
  - `_compute_sdpo_loss(...)` always reduces with `seq-mean-token-mean`
  - the result is computed per microbatch
  - it is scaled by active-sequence share so the summed microbatch contributions match the minibatch-level objective

Conclusion:
- The narrow actor-side `sdpo_self + full_logit` implementation is internally consistent.
- The main problems are configuration interactions and documentation accuracy.

## What To Fix Tomorrow

Recommended order:

1. Add config validation in `JointSDPOSelfDistillTrainer._load_distill_settings()`:
   - reject `merge_to_advantages=true` with `loss_variant="full_logit"`
   - reject `use_stale_coefficient=true` with `loss_variant="full_logit"`
   - decide whether to reject or document `is_clip=true` with `loss_variant="full_logit"`

2. Clean up `is_clip` semantics:
   - either make it purely boolean everywhere
   - or preserve numeric meaning end-to-end

3. Update `.codex/method.md`:
   - replace "two updates" with "one joint actor update" if that is the intended implementation
   - explicitly note any approximations:
     - joint GRPO + SDPO optimization
     - clipped full-logit if retained

4. Add tests:
   - invalid config tests for the blocked combinations
   - a test that `merge_to_advantages=true` is rejected for `full_logit`
   - a test that `use_stale_coefficient=true` is rejected for `full_logit`
   - if keeping clipped full-logit, add a test that behavior is intentional and documented

## Validation Checklist

After patching:

1. Run unit tests for:
   - `tests/test_sdpo_actor_loss.py`
   - `tests/test_sdpo_self_distill.py`

2. Run at least one smoke experiment with:
   - `mode=sdpo_self`
   - `loss_variant=full_logit`
   - `merge_to_advantages=false`
   - `use_stale_coefficient=false`
   - `is_clip=false`

3. Confirm metrics are sensible:
   - `distill/sdpo_loss`
   - `distill/token_count`
   - `distill/active_seq_count`
   - no unexpected skips from the distill payload builder

## Local Test Environment Note

I could not get a clean local `pytest` run in this session because imports resolved `verl` from a different external location that does not export `agg_loss`.

Observed failure:
- import error from `verl.trainer.ppo.core_algos`
- resolved path was outside this repo’s intended local dependency layout

Before trusting local test results, make sure the Python environment matches the project’s expected `verl` / `rllm` setup.

