# SDPO Diagnosis Experiments — Run Guide

Reference: `sdpo_diagnosis_plan.md` for full rationale and hypotheses.

**Environment:** FrozenLake
**Script:** `scripts/train_frozenlake_self_distill_multi_episode.sh`
**Base config used in original observation:** full_logit, alpha=1, EMA teacher, `first_attempt_latest_success_hindsight_first_failure_only`

## Prerequisites

```bash
conda activate icx
```

Ensure you're on the `feature/exp_dist` branch (has diagnostic logging for Exp 1).

---

## Exp 1: Teacher Distribution Diagnostics (logging only)

No config changes needed — new diagnostic metrics are automatically logged when distillation is enabled. Just run any SDPO experiment and monitor these new metrics:

- `distill/diag_mean_student_logp` — mean student token log-prob on distilled tokens
- `distill/diag_mean_teacher_logp` — mean teacher token log-prob on distilled tokens
- `distill/diag_student_entropy` — student entropy (from top-k logits)
- `distill/diag_teacher_entropy` — teacher entropy (from top-k logits)
- `distill/diag_kl_per_token_mean` — mean per-token KL value

**What to look for:**
- `teacher_entropy << student_entropy` → teacher is too peaked (H2/H5 confirmed)
- `teacher_entropy ≈ student_entropy` → hindsight is noise (H4)
- `kl_per_token_mean` growing over training → distributions diverging

**Run (same as current best config):**

```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=1.0 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

---

## Exp 2: Alpha Comparison

Tests whether reverse KL (alpha=1) is the wrong divergence direction.

### Run 2A: alpha=1.0 (current, reverse KL)
```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=1.0 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

### Run 2B: alpha=0.0 (forward KL)
```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=0.0 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

### Run 2C: alpha=0.5 (JSD)
```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=0.5 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

**What to look for:** If alpha=0 shows less length collapse + better ep1 → reverse KL is wrong direction (H8). If all alphas degrade similarly → teacher signal itself is the problem.

---

## Exp 3: Teacher Context Ablation

Tests whether success-informed hindsight context is helping or hurting.

### Run 3A: Baseline (no SDPO)
```bash
ENABLE_SELF_DISTILL=False \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

### Run 3B: Current config (teacher sees latest success)
Same as Run 2A above.

### Run 3C: Teacher sees all attempts (not isolated success)

Uses `first_attempt_hindsight` which gives teacher the full cumulative context rather than the isolated success attempt.

```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=1.0 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_hindsight \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

**What to look for:** 3B ≈ 3C → hindsight content doesn't matter (noise). 3C better than 3B → isolated success context is harmful. 3C worse than 3B → success isolation helps.

---

## Exp 4: Selection Gate Ablation

Tests whether the failure-only gate creates harmful bias.

### Run 4A: failure-only gate (current)
Same as Run 2A.

### Run 4B: any-success gate (no failure requirement)
```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=1.0 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

### Run 4C: broadest selection (all trajectories)
Same as Run 3C.

**What to look for:** 4B or 4C better → failure-only gate creates harmful asymmetry (H7).

---

## Exp 5: Lambda Sweep

| Run | Lambda | Command addition |
|-----|--------|-----------------|
| 5A | 0.0 | `ENABLE_SELF_DISTILL=False` (same as 3A) |
| 5B | 0.001 | `DISTILL_LAMBDA=0.001` |
| 5C | 0.005 | `DISTILL_LAMBDA=0.005` |
| 5D | 0.01 | `DISTILL_LAMBDA=0.01` (same as 2A) |
| 5E | 0.05 | `DISTILL_LAMBDA=0.05` |
| 5F | 0.1 | `DISTILL_LAMBDA=0.1` |

Base for all (except 5A):
```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=1.0 \
DISTILL_LAMBDA=<VALUE> \
DISTILL_IS_CLIP=false \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

**What to look for:** Monotonically worse → teacher signal fundamentally harmful. Sweet spot → magnitude issue.

---

## Exp 6: Negation Mechanism

### Run 6A: Baseline
Same as 3A.

### Run 6B: Negated SDPO (known to work at lambda=0.01)
```bash
DISTILL_LOSS_VARIANT=full_logit \
DISTILL_ALPHA=1.0 \
DISTILL_LAMBDA=0.01 \
DISTILL_IS_CLIP=false \
NEGATE_SDPO_LOSS=True \
DISTILL_TRAJECTORY_SELECTION=first_attempt_latest_success_hindsight_first_failure_only \
TEACHER_CONTEXT_ATTEMPTS=null \
TEACHER_REGULARIZATION=ema \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh
```

### Run 6C: Entropy bonus (no SDPO, increased entropy_coeff)

Match response length / entropy level of Run 6B by tuning `entropy_coeff`. Start with:

```bash
ENABLE_SELF_DISTILL=False \
  bash scripts/train_frozenlake_self_distill_multi_episode.sh \
  actor_rollout_ref.actor.entropy_coeff=0.01
```

Adjust `entropy_coeff` until response length roughly matches 6B.

**What to look for:** 6B ≈ 6C → negated SDPO is just entropy regularization. 6B > 6C → pushing away from teacher has specific directional value.

---

## Metrics to Monitor (all experiments)

| Metric | Where | What it tells you |
|--------|-------|-------------------|
| `ep1_success`, `ep2_success` | trainer logs | Core performance |
| `ep2_minus_ep1_gap` | trainer logs | Gap widening symptom |
| `overall_success` | trainer logs | Aggregate performance |
| `mean_response_length` (ep1/ep2) | trainer logs | Length collapse symptom |
| `distill/sdpo_loss` | actor logs | Loss value |
| `distill/diag_mean_student_logp` | actor logs | Student confidence |
| `distill/diag_mean_teacher_logp` | actor logs | Teacher confidence |
| `distill/diag_student_entropy` | actor logs | Student distribution breadth |
| `distill/diag_teacher_entropy` | actor logs | Teacher distribution breadth |
| `distill/diag_kl_per_token_mean` | actor logs | Student-teacher divergence |
| `distill/active_seq_frac` | actor logs | What fraction of batch gets SDPO |
| `distill/kept_ratio` | trainer logs | Samples kept vs skipped |

## Priority Order

1. **Exp 1** — free, just check metrics on any SDPO run
2. **Exp 2** — alpha comparison (3 runs)
3. **Exp 3** — teacher context ablation (3 runs, shares runs with Exp 2)
4. **Exp 4** — selection gate (shares runs with Exp 2/3)
5. **Exp 5** — lambda sweep (6 runs, shares some with above)
6. **Exp 6** — negation analysis (3 runs)

**Note:** Many runs overlap — Runs 2A, 3B, 4A, 5D are all the same config. Runs 3A, 5A, 6A are all the same baseline. Plan carefully to avoid redundant runs.
