# SDPO Full-Logit Diagnosis Plan

## Observed Phenomena (FrozenLake, full_logit, alpha=1)

1. With SDPO loss, the gap between ep1 and ep2 success rate widens; vanilla GRPO keeps it stable.
2. With SDPO loss, ep1, ep2, and overall performance all drop.
3. With SDPO loss, response length shrinks over training.
4. With negated SDPO loss (lambda=0.01), ep1 and overall performance increase and response length increases. Lambda=0.1 negated collapses quickly (response length and entropy increase too fast). Both negated lambdas increase response length and entropy; larger lambda = faster increase rate.

## Current Setup

| Setting | Value |
|---------|-------|
| Loss variant | `full_logit` (top-64 reverse KL) |
| Alpha | 1.0 → KL(student ‖ teacher), mode-seeking |
| Teacher | EMA of student weights (update_rate=0.05) |
| Teacher context | Isolated latest successful attempt (hindsight) |
| Trajectory selection | `first_attempt_latest_success_hindsight_first_failure_only` |
| Target tokens | First-attempt tokens, only when ep1 failed and a later ep succeeded |
| Lambda | 0.1 and 0.01 (positive), 0.1 and 0.01 (negated) |
| PPO | Joint GRPO + lambda * SDPO |
| Episodes | >= 2 per trajectory |

## Ruled-Out Hypotheses

- **Gradient sign inversion**: Verified correct for both `non_full` and `full_logit`. The loss correctly pushes student toward teacher (minimizing KL).
- **Cumulative advantage noise**: Not applicable — `full_logit` uses direct KL, not REINFORCE-style `_segmented_reverse_cumsum`.

## Active Hypotheses

| ID | Hypothesis | Reasoning | Explains |
|----|-----------|-----------|----------|
| H2 | **Teacher is over-certain** | Teacher sees the successful solution via hindsight, so its distribution is peaked. Reverse KL (mode-seeking) makes student collapse to teacher's single mode → shorter, less exploratory responses. | 2, 3 |
| H3 | **SDPO gradient magnitude dominates PPO** | lambda=0.1 may produce gradients that overwhelm the PPO reward signal, especially since SDPO only fires on a subset of sequences. | 1, 2, 3 |
| H5 | **Distilling toward success-informed teacher kills exploration** | Teacher "knows what works" and assigns low prob to exploratory/probing actions. Reverse KL punishes student for having mass where teacher doesn't → student stops exploring → ep1 drops, ep2 (which depends on ep1 exploration) also drops → gap widens. | 1, 2, 3, 4 |
| H7 | **Selection bias from failure-only gate** | SDPO only fires on trajectories where ep1 failed + later success. This creates asymmetric gradient pressure: only "bad" ep1 trajectories get the SDPO pull, while "good" ep1 trajectories only see PPO. This may systematically bias the policy. | 1, 2 |
| H8 | **Reverse KL is wrong direction for this task** | Reverse KL is mode-seeking: student collapses to teacher's highest-prob mode. For failed first attempts, teacher's mode is "do what worked" not "explore broadly." Forward KL (mean-covering) would preserve student diversity. | 3, 4 |

## Experiments

### Exp 1: Teacher distribution diagnosis (logging only)

**Goal:** Directly measure what the teacher distribution looks like vs student on distilled tokens.

**Add logging in `_compute_sdpo_loss` for the `full_logit` path:**

- `distill/mean_teacher_logp`: mean teacher log-prob on distilled tokens
- `distill/mean_student_logp`: mean student log-prob on distilled tokens
- `distill/teacher_entropy`: teacher entropy from top-k logits (= -sum p*logp over top-k)
- `distill/student_entropy`: student entropy from top-k logits
- `distill/entropy_ratio`: teacher_entropy / student_entropy (< 1 means teacher is more peaked)
- `distill/kl_per_token_mean`: mean per-token KL value before aggregation
- `distill/active_seq_frac`: fraction of batch that actually receives SDPO loss

**Diagnostic:**
- teacher_entropy << student_entropy → H2/H5 confirmed (teacher too peaked)
- teacher_entropy ≈ student_entropy, KL small → H4 direction (hindsight is noise)
- active_seq_frac very small → SDPO signal is sparse but concentrated, potentially destabilizing

**Priority:** Highest. Zero compute cost (piggybacks on existing runs). Run first, interpret results before committing to expensive sweeps.

### Exp 2: Alpha comparison (tests H8)

**Goal:** Test whether reverse KL is the wrong divergence for this distillation task.

| Run | Alpha | KL type | Expected behavior |
|-----|-------|---------|-------------------|
| A | 1.0 (current) | KL(student ‖ teacher) | Mode-seeking → student collapses to teacher's peak |
| B | 0.0 | KL(teacher ‖ student) | Mean-covering → student preserves diversity, teacher weights pull gently |
| C | 0.5 | JSD | Symmetric, bounded → stable middle ground |

Use lambda=0.01 for all runs (milder signal, less confounded by magnitude).

**Diagnostic:**
- alpha=0 shows less length collapse + better ep1 → H8 confirmed
- All alphas degrade similarly → KL direction doesn't matter; teacher signal itself is the problem (H5)

### Exp 3: Teacher context ablation (tests H5)

**Goal:** Is success-informed hindsight context actually helping the teacher, or is it the source of harm?

| Run | Teacher context | Description |
|-----|----------------|-------------|
| A | lambda=0 | No SDPO (baseline) |
| B | Latest successful attempt (current) | Teacher sees what worked |
| C | `teacher_context_attempts=0` equivalent: teacher sees same prompt as student | No hindsight |

Note: C requires using a different trajectory selection (e.g., `first_attempt_hindsight`) since the current strategy requires success context. Alternatively, modify the teacher context construction to pass empty hindsight.

**Diagnostic:**
- B ≈ C → hindsight is noise, teacher just provides a regularization signal (H5 weakened)
- C better than B → hindsight actively hurts — teacher "knowing the answer" is the problem (H5 confirmed)
- C worse than B → hindsight helps, but not enough to overcome other issues

### Exp 4: Selection gate ablation (tests H7)

**Goal:** Does the failure-only gate create harmful selection bias?

| Run | Trajectory selection | When SDPO fires |
|-----|---------------------|-----------------|
| A | `first_attempt_latest_success_hindsight_first_failure_only` (current) | Only failed ep1 + later success |
| B | `first_attempt_latest_success_hindsight` | All trajectories with any successful attempt |
| C | `first_attempt_hindsight` | All trajectories (broadest) |

Use lambda=0.01, alpha=1.0 for all.

**Diagnostic:**
- B or C better than A → failure-only gate creates harmful asymmetry (H7 confirmed)
- A better than B/C → filtering is correct; broader selection adds noise

### Exp 5: Lambda sweep (tests H3)

**Goal:** Dose-response curve under full_logit alpha=1.

| Run | Lambda |
|-----|--------|
| A | 0.0 |
| B | 0.001 |
| C | 0.005 |
| D | 0.01 |
| E | 0.05 |
| F | 0.1 |

**Diagnostic:**
- Monotonically worse with higher lambda → teacher signal fundamentally harmful at any dose
- Sweet spot exists → magnitude issue (H3), use that lambda going forward
- All non-zero lambdas equivalent → SDPO is noise, neither helping nor hurting

### Exp 6: Negation mechanism analysis (tests whether negated SDPO = entropy bonus)

**Goal:** Understand WHY negating SDPO helps. Is it the teacher content, or just entropy regularization?

**Already known:**
- Negated lambda=0.1 collapses quickly (entropy/length explode)
- Negated lambda=0.01 outperforms GRPO baseline
- Both negated lambdas increase response length and entropy; larger lambda = faster rate
- This dose-response confirms negation acts as entropy regularization, but the question is whether the teacher-specific direction matters

| Run | Config |
|-----|--------|
| A | lambda=0 (baseline) |
| B | lambda=-0.01, full_logit alpha=1 (negated, known to work) |
| C | lambda=0, increased entropy_coeff to roughly match B's response length / entropy level |

**Diagnostic:**
- B ≈ C → negated SDPO is just an entropy bonus; teacher content is irrelevant
- B > C → negated SDPO provides directional value beyond raw entropy; pushing away from the success-informed teacher specifically helps exploration

### Exp 7: Gradient magnitude (logging only, add to any run)

**Goal:** Measure whether SDPO gradients overwhelm PPO.

**Add logging in `update_policy()`:**

- Compute `||grad_ppo||` and `||grad_sdpo||` separately (via two backward passes on micro-batch, or `torch.autograd.grad`)
- `distill/grad_norm_ppo`, `distill/grad_norm_sdpo`, `distill/grad_ratio`

**Diagnostic:**
- grad_ratio >> 1 → SDPO dominates regardless of direction (H3)
- grad_ratio ≈ 1 → magnitude balanced, problem is directional

## Metrics To Monitor (all experiments)

| Category | Metric | Source |
|----------|--------|--------|
| Performance | ep1_success, ep2_success, ep2_minus_ep1_gap, overall_success | trainer |
| Length | mean_response_length_ep1, mean_response_length_ep2 | trainer/env |
| SDPO loss | distill/sdpo_loss, distill/sdpo_loss_scaled | sdpo_actor |
| Teacher quality | distill/mean_teacher_logp, distill/mean_student_logp | Exp 1 logging |
| Entropy | distill/teacher_entropy, distill/student_entropy | Exp 1 logging |
| Coverage | distill/active_seq_frac, distill/kept_ratio | sdpo_actor / trainer |
| Gradient | distill/grad_norm_ppo, distill/grad_norm_sdpo | Exp 7 logging |

## Execution Order

1. **Exp 1** (logging) — free, maximum insight, informs all other decisions
2. **Exp 2** (alpha comparison) — most specific to full_logit setup; reverse KL + success hindsight is top suspect
3. **Exp 3** (teacher context ablation) — directly tests whether hindsight is the root cause
4. **Exp 4** (selection gate) — tests the asymmetric training signal
5. **Exp 5** (lambda sweep) — dose-response, only if Exps 2-4 suggest SDPO can work at some scale
6. **Exp 6** (negation analysis) — mechanistic understanding
7. **Exp 7** (gradient norms) — add as logging to any run above
