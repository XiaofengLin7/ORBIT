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

#### Run 1a (buggy attention mask): `yn5srku9`

**Note:** This run had a bug in the distillation context construction — the attention mask was incorrect, causing the teacher/student to not properly attend to context tokens. Results are kept for reference but superseded by Run 1b.

| Phase (steps) | student_logp | teacher_logp | student_entropy | teacher_entropy | kl_per_token |
|---------------|-------------|-------------|-----------------|-----------------|-------------|
| 0–39 | -0.248 | -0.296 | 0.019 | 0.019 | 0.084 |
| 40–79 | -0.534 | -0.575 | 0.044 | 0.046 | 0.100 |
| 80–119 | -0.632 | -0.666 | 0.052 | 0.054 | 0.088 |
| 120–159 | -0.799 | -0.825 | 0.051 | 0.054 | 0.084 |

#### Run 1b (attention mask fixed): `hw9278mu`

**Wandb run:** `hw9278mu` (168 training steps, full_logit, alpha=1, lambda=0.01, EMA teacher)
**Baseline run:** `zzcyem9j` (no SDPO, same FrozenLake config)

##### Diagnostic Metrics Over Training

| Phase (steps) | student_logp | teacher_logp | logp diff (t-s) | student_entropy | teacher_entropy | kl_per_token |
|---------------|-------------|-------------|-----------------|-----------------|-----------------|-------------|
| 0–42 | -0.004 | -0.121 | -0.117 | 0.0022 | 0.0019 | 0.128 |
| 42–84 | -0.001 | -0.123 | -0.121 | 0.0002 | 0.0004 | 0.120 |
| 84–126 | -0.002 | -0.163 | -0.161 | 0.0004 | 0.0006 | 0.167 |
| 126–168 | -0.001 | -0.164 | -0.163 | 0.0001 | 0.0007 | 0.165 |

##### Baseline Comparison (late phase)

| Metric | SDPO (hw9278mu) | Baseline (zzcyem9j) | Delta |
|--------|----------------|---------------------|-------|
| ep1 success | ~0.62 | ~0.63 | -0.01 |
| ep2 success | ~0.77 | ~0.79 | -0.02 |
| overall success | ~0.87 | ~0.87 | ~0 |
| response length | ~4763 | ~5017 | -254 |
| active_seq_frac | 5–23% | n/a | — |

##### Key Changes from Run 1a → 1b (attention mask fix)

| Metric | Run 1a (buggy) | Run 1b (fixed) |
|--------|---------------|----------------|
| Student logp | -0.25 → -0.80 | **-0.004 → -0.001** (near-deterministic) |
| Teacher logp | -0.30 → -0.83 | -0.12 → -0.16 |
| Student entropy | 0.019 → 0.051 | **0.0001 → 0.002** (~25x lower) |
| KL per token | 0.08 → 0.10 | **0.12 → 0.17** (~2x higher) |
| SDPO loss | 0.008 → 0.022 | **0.016 → 0.036** (~2x higher) |
| Grad norm | 0.14 → 0.58 | **0.32 → 1.59** (~2-3x higher) |
| Performance | degraded vs baseline | **~same as baseline** |

#### Analysis

1. **H2 (teacher too peaked) — RULED OUT.** Both student and teacher entropies are very low and comparable. The teacher is not over-certain relative to the student.

2. **Student is near-deterministic on distilled tokens.** Student logp ≈ 0 (P ≈ 1) — the student already assigns nearly all probability mass to its own generated tokens. This is expected: with the corrected attention mask, the student can properly attend to prior context, so it's very confident on tokens it generated during rollout.

3. **Teacher is less confident than student.** Teacher logp is -0.12 to -0.16 (≈ 85–88% probability) vs student's ≈100%. The hindsight context makes the teacher *less sure* about the student's token choices, not more. The logp gap is stable at -0.12 to -0.16 throughout training.

4. **KL is higher than the buggy run** (0.12–0.17 vs 0.08–0.10) and grows over training. The divergence is driven by the student being near-deterministic while the teacher spreads probability more broadly.

5. **SDPO signal is stronger but performance is neutral.** Despite 2x higher SDPO loss and 2–3x higher grad norms vs the buggy run, task performance is now essentially identical to the baseline. This suggests the SDPO gradient is largely **orthogonal to task-relevant directions** — it changes the distribution without improving (or in this config, meaningfully hurting) downstream performance.

6. **The hindsight context produces a slightly less confident teacher, not a better one.** The teacher doesn't appear to learn a substantially different policy from seeing the successful attempt — it just becomes slightly less certain about the student's specific token choices. This raises the question: is the hindsight context providing a useful teaching signal at all?

#### Hypothesis Status (after Exp 1 + Exp 2 partial)

| Hypothesis | Status | Evidence |
|-----------|--------|----------|
| H1 (gradient sign wrong) | Ruled out | Code audit confirmed correct direction |
| H2 (teacher too peaked) | Ruled out | Both entropies near-zero and comparable |
| H3 (gradient magnitude) | Open | SDPO loss dominates pg_loss by ~100x; grad norms 2-3x higher with reverse KL |
| H5 (teacher kills exploration) | **Partially confirmed** | Reverse KL collapses student entropy to 0.002; forward KL preserves at 0.101 |
| H7 (selection bias) | Open | Very low coverage (5-23%) |
| H8 (reverse KL wrong direction) | **Confirmed** | Forward KL outperforms reverse KL on all metrics; 50x more entropy preserved |
| H9 (weak teacher signal) | Open | Teacher logp -0.12 to -0.39 vs student ≈0; hindsight context produces a less confident teacher, not a clearly better one |

#### Revised Priority

1. ~~**Exp 2 (alpha comparison)**~~ — **2A, 2B done.** H8 confirmed: forward KL > reverse KL. **2C (JSD) still pending.**
2. **Exp 3 (teacher context)** — does richer context produce a stronger teacher signal? (H9)
3. **Exp 4 (selection gate)** — broadening gate may increase SDPO coverage beyond 5-23%.
4. **Exp 5 (lambda sweep)** — dose-response, especially with forward KL as the new default.
5. **Exp 6 (negation)** — lower priority now that forward KL works well.

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

### Exp 2 Partial Results (2A vs 2B)

**Run 2A (reverse KL, α=1):** `hw9278mu` — 170 steps, crashed
**Run 2B (forward KL, α=0):** `xuxe4l9q` — 168 steps, still running

#### Validation Performance

| Metric (val) | Forward KL (α=0) | Reverse KL (α=1) |
|---|---|---|
| overall success (late) | **0.867** | 0.849 |
| ep1 success (late) | **0.648** | 0.590 |
| ep2 success (late) | **0.794** | 0.760 |

#### Training Performance

| Metric (train, late) | Forward KL (α=0) | Reverse KL (α=1) |
|---|---|---|
| overall success | **0.867** | 0.837 |
| ep1 success | **0.703** | 0.651 |
| ep2 success | **0.792** | 0.763 |

#### Distillation Diagnostics

| Metric | Phase | Forward KL (α=0) | Reverse KL (α=1) |
|---|---|---|---|
| student_logp | Late | -0.065 (soft) | -0.001 (peaked) |
| teacher_logp | Late | -0.386 | -0.157 |
| student_entropy | Late | **0.052** | 0.0002 |
| teacher_entropy | Late | 0.015 | 0.0006 |
| KL/token | Late | **0.085** (stable) | 0.160 (growing) |
| sdpo_loss | Late | 0.011 | 0.017 |
| actor/entropy | Late | **0.101** | 0.002 |
| grad_norm (early) | Early | **0.506** | 1.311 |

#### Analysis

1. **H8 (reverse KL wrong direction) — CONFIRMED.** Forward KL outperforms reverse KL on all val metrics (+0.018 overall, +0.058 ep1, +0.034 ep2). The mechanism is clear: reverse KL (mode-seeking) collapses the student to near-deterministic (entropy 0.002), while forward KL (mean-seeking) preserves 50x more entropy (0.101). This entropy preservation translates directly to better exploration and higher ep1 success.

2. **Forward KL keeps the student soft.** Student logp is -0.065 (forward) vs -0.001 (reverse). The forward KL student maintains a broader distribution rather than collapsing onto a single mode.

3. **Forward KL has more stable optimization.** Early grad norms are 0.51 (forward) vs 1.31 (reverse), and KL/token is stable at 0.085 vs growing to 0.160 for reverse.

4. **Teacher logp is much lower under forward KL** (-0.39 vs -0.16). This is expected: the softer student distribution produces tokens that the teacher is less confident about, but the mean-seeking objective handles this gracefully by spreading probability rather than concentrating it.

5. **Run 2C (JSD, α=0.5) still pending** — would test whether a middle ground between forward and reverse KL finds a sweet spot.

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

## Priority Order (updated after Exp 2 partial)

1. ~~**Exp 1**~~ — **DONE.** H2 ruled out. Teacher signal is weak (H9).
2. ~~**Exp 2A/2B**~~ — **DONE.** H8 confirmed: forward KL (α=0) > reverse KL (α=1). Forward KL preserves 50x more entropy, better val performance.
3. **Exp 2C** — JSD (α=0.5) — does a middle ground improve further?
4. **Exp 3** — teacher context ablation — can richer context strengthen the weak teacher signal (H9)?
5. **Exp 4** — selection gate ablation — increase SDPO coverage beyond 5-23%.
6. **Exp 5** — lambda sweep with forward KL as new default α.
7. **Exp 6** — negation analysis — lower priority.

**Note:** Many runs overlap — Runs 2A, 3B, 4A, 5D are all the same config. Runs 3A, 5A, 6A are all the same baseline. Plan carefully to avoid redundant runs.
