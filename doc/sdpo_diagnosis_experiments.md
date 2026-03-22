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

### Exp 1 Results

**Wandb run:** `yn5srku9` (160 training steps, full_logit, alpha=1, lambda=0.01, EMA teacher)
**Baseline run:** `zzcyem9j` (no SDPO, same FrozenLake config)

#### Diagnostic Metrics Over Training

| Phase (steps) | student_logp | teacher_logp | logp diff (t-s) | student_entropy | teacher_entropy | entropy ratio (t/s) | kl_per_token |
|---------------|-------------|-------------|-----------------|-----------------|-----------------|---------------------|-------------|
| 0–39 | -0.362 | -0.385 | -0.023 | 2.093 | 2.037 | 0.97 | 0.0055 |
| 40–79 | -0.357 | -0.390 | -0.033 | 1.974 | 1.937 | 0.98 | 0.0077 |
| 80–119 | -0.353 | -0.394 | -0.041 | 1.808 | 1.832 | 1.01 | 0.0103 |
| 120–159 | -0.386 | -0.432 | -0.046 | 1.789 | 1.898 | 1.06 | 0.0107 |

#### Baseline Comparison (at step 160)

| Metric | SDPO (yn5srku9) | Baseline (zzcyem9j) | Delta |
|--------|----------------|---------------------|-------|
| ep1 success | ~0.30 | ~0.60 | -0.30 |
| ep2 success | ~0.55 | ~0.68 | -0.13 |
| ep1-ep2 gap | ~0.25 | ~0.08 | +0.17 |
| response length | ~1200 | ~3500 | -2300 |
| active_seq_frac | 5–23% | n/a | — |

#### Analysis

1. **H2 (teacher too peaked) — RULED OUT.** Teacher entropy ≈ student entropy throughout training (ratio 0.97→1.06). The teacher is not over-certain; hindsight context does not collapse the teacher distribution.

2. **Teacher assigns lower log-prob than student on actual tokens.** The teacher logp is consistently lower than student logp (gap widens from -0.023 to -0.046 over training). This means the teacher is *less confident* on the tokens the student generates. Under reverse KL (alpha=1, mode-seeking), this pushes the student to concentrate on modes of the teacher distribution — which may differ from the student's current modes.

3. **KL divergence is growing.** `kl_per_token_mean` nearly doubles (0.0055 → 0.0107), indicating student and teacher distributions are diverging over training rather than converging. The SDPO signal is not achieving alignment.

4. **Very low SDPO coverage.** `active_seq_frac` ranges 5–23%, meaning SDPO fires on only a small fraction of the batch (requires ep1 failure + later success). The signal is sparse but still causes significant performance degradation. This suggests the per-sample impact is outsized.

5. **Both entropies decline together.** Student entropy drops from 2.09→1.79, teacher from 2.04→1.90. The student entropy drops faster early on, consistent with response length collapse. By late training the teacher is actually slightly *higher* entropy than the student.

#### Updated Hypothesis Status

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| H1 (gradient sign wrong) | Ruled out | Code audit confirmed correct direction |
| H2 (teacher too peaked) | **Ruled out** | Entropy ratio ≈ 1.0 throughout |
| H3 (gradient magnitude) | Open | KL growing suggests possible magnitude issue |
| H5 (teacher kills exploration) | Partially addressed | Teacher not peaked, but reverse KL mode-seeking may still suppress student exploration |
| H7 (selection bias) | Open | Very low coverage (5-23%) amplifies per-sample impact |
| H8 (reverse KL wrong direction) | **Now highest priority** | Teacher logp < student logp + growing KL + mode-seeking KL = student collapses onto wrong modes |

#### Revised Priority

Given Exp 1 findings, the most actionable next experiments are:

1. **Exp 2 (alpha comparison)** — highest priority. The growing KL + mode-seeking dynamics strongly suggest reverse KL (alpha=1) may be the wrong divergence. Forward KL (alpha=0) would mean-seek instead of mode-seek, potentially avoiding the collapse.
2. **Exp 4 (selection gate)** — the very low active_seq_frac (5-23%) means SDPO only fires on the hardest cases (ep1 failed). Broadening the gate may reduce per-sample impact and improve signal quality.
3. **Exp 5 (lambda sweep)** — KL per token is small (~0.01) but with lambda=0.01 still causes degradation. Understanding the dose-response is important.
4. **Exp 3 (teacher context)** — lower priority now that H2 is ruled out. The teacher distribution is not problematic in aggregate, so context changes may have limited effect.
5. **Exp 6 (negation)** — deferred; we already know negation helps, understanding *why* the positive direction hurts is more urgent.

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

## Priority Order (updated after Exp 1)

1. ~~**Exp 1**~~ — **DONE.** H2 ruled out; teacher not peaked. Key finding: reverse KL + growing KL divergence → H8 is top suspect.
2. **Exp 2** — alpha comparison (3 runs) — **highest priority.** Tests H8 directly.
3. **Exp 4** — selection gate ablation — low active_seq_frac (5-23%) suggests gate bias worth investigating.
4. **Exp 5** — lambda sweep (6 runs) — dose-response to quantify magnitude effects.
5. **Exp 3** — teacher context ablation — lower priority since H2 is ruled out.
6. **Exp 6** — negation analysis — deferred; positive direction is the priority.

**Note:** Many runs overlap — Runs 2A, 3B, 4A, 5D are all the same config. Runs 3A, 5A, 6A are all the same baseline. Plan carefully to avoid redundant runs.
