# Summarization Modes: Logic, Diagnostics, and Observed Behavior

Scope: how `SummarizingAgentExecutionEngine` + `ContextSummarizerMixin` implement the `token`, `episodic`, and `both` modes; what log/warning messages each mode can emit; and what the patterns look like in the `meta-rl-summarization-eval` benchmark runs (base vs ORBIT Qwen3-8B on maze + mastermind, 3- and 5-episode horizons).

---

## 1. Architecture recap

Two collaborating components:

- `ContextSummarizerMixin` (`agents/context_summarizer.py`) exposes two trigger predicates (`should_summarize`, `should_summarize_on_episode_end`), a prompt builder (`build_summarization_prompt`), and `apply_summary` that mutates the agent's message history.
- `SummarizingAgentExecutionEngine` (`trainers/summarizing_engine.py`) checks those predicates once per step (after `env.step` + `update_from_env`), calls `_do_summarization` to run the summary LLM call, applies the summary, and handles post-summary bookkeeping.

Both modes share the same check point in the step loop, the same `_do_summarization` helper, the same context-window guard, and the same segment-assembly path. They differ only in **when** they fire and **how failures are handled**.

---

## 2. Mode: `token`

### Logic (post-env check, every step)

```
env.step returns → update_from_env appends env obs
      │
      ▼
should_summarize()?   (accumulated chat tokens ≥ threshold_tokens)
      │ no  → continue trajectory
      │ yes
      ▼
response_token_len + (summary_max_tokens + summ_instruction_len) ≥ max_response_length?
      │ yes → termination = SUMMARIZATION_BUDGET_EXCEEDED (red)
      │       break → post-loop zeros step rewards
      │ no
      ▼
_do_summarization(trigger="token")
      │
      ▼
summary LLM returned None?   (max_model_len overflow, empty output, exception)
      │ yes → termination = SUMMARIZATION_FAILED (red)
      │       break → post-loop zeros step rewards
      │ no
      ▼
apply_summary → [sys, summary]
response_token_len = 0
continue trajectory   (next env obs gets appended on the NEXT iteration's update_from_env)
```

Trigger predicate (`should_summarize`): fires when **all** of
- `summarization_mode in {"token", "both"}`
- `len(_messages) >= 3`
- the full chat history (system + every user + every assistant turn) rendered through the chat template tokenizes to ≥ `summarization_threshold_tokens` — this is the agent's entire accumulated context, not just the final prompt

Behavior when it fires:
1. **Budget check** — if `response_token_len + summary_max_tokens + summ_instruction_len >= max_response_length`, hard fail. `summ_instruction_len` is the tokenized length of the summarization prompt (`TOKEN_SUMMARY_PROMPT` for the token path), pre-computed once at trajectory start.
2. Call `_do_summarization(trigger="token", use_reflective_prompt=False)`.
3. Prompt = `TOKEN_SUMMARY_PROMPT` appended to the current history.
4. Post-summary layout via `apply_summary`: `[system, summary_user_msg]`. The summary is expected to carry forward whatever the agent needs to know about the current episode; the next env observation is appended automatically by the next iteration's `update_from_env`.
5. Reset `response_token_len = 0`.

### Termination paths (token)

The summarization-status prints (`Summarization #N applied`, `summarization skipped`, `summarization failed`, `Summarization prompt … exceeds max_model_len`, etc.) were removed from `_do_summarization` — they were noisy at scale (one per trajectory × N summarizations per step). Per-trajectory observability now comes from the W&B-logged metrics (`traj/summarization_count_*`, `traj/segment_count_*`).

The trajectory-completion prints in `summarizing_engine.py:480` (`Trajectory X completed due to: <reason>. Reward is Y.`) and the truncation print at `summarizing_engine.py:515` are still emitted; they're the canonical signal that a trajectory finished and what the engine did with it.

| Termination reason | Trigger | Effect on trajectory reward |
|---|---|---|
| `ENV_DONE` | `done=True` from env (outer trajectory complete) | Reward preserved |
| `MAX_STEPS` | Loop reached `max_steps - 1` | Reward preserved |
| `TRUNCATION` | `response_token_len >= max_response_length` after a step | Reward preserved |
| `PROMPT_TRUNCATION` | `prompt_len > max_prompt_length` before a step | All step rewards zeroed |
| `SUMMARIZATION_BUDGET_EXCEEDED` | Token-mode budget guard fired | All step rewards zeroed |
| `SUMMARIZATION_FAILED` | `_do_summarization` returned `None` | All step rewards zeroed |
| `ENV_TIMEOUT` / `TIMEOUT` | `asyncio.wait_for` exhausted `trajectory_timeout` | Reward preserved |

**Bottom line for token mode:** any summarization failure is fatal — the `termination_reason` lands in the `truncation_reasons` tuple inside `summarizing_engine.py`, and all per-step rewards are zeroed in the post-loop block (`for step in trajectory.steps: step.reward = 0.0`).

---

## 3. Mode: `episodic`

### Logic (at an episode boundary)

```
env.step returns terminal_obs only     (reflection_via_summarization=True)
update_from_env appends terminal_obs
      │
      ▼
should_summarize_on_episode_end(info)?   (info["episode_done"] and not trajectory_done)
      │ no  → continue trajectory
      │ yes
      ▼
response_token_len + (summary_max_tokens + summ_instruction_len) ≥ max_response_length?
      │ yes → termination = SUMMARIZATION_BUDGET_EXCEEDED
      │       break → post-loop zeros step rewards
      │ no
      ▼
_do_summarization(trigger="episode_end",
                  use_reflective_prompt=env.enable_reflection)
      │
      ▼
summary LLM returned None?   (max_model_len overflow, empty output, exception)
      │ yes → termination = SUMMARIZATION_FAILED
      │       break → post-loop zeros step rewards
      │ no
      ▼
agent._messages.append(new_obs) [bypass update_from_env to preserve prior step's reward]
env.start_new_episode() → new_obs
response_token_len = 0
continue trajectory
```

The trainer auto-sets `env_args.reflection_via_summarization=True` when mode ∈ `{episodic, both}`. That flips `MultiEpisodeEnv.step` into boundary-split mode: at `inner_done=True` the env returns **only** the terminal observation, sets `_pending_episode_start=True`, and does NOT combine with the next-episode obs or append a reflection prompt.

Trigger predicate (`should_summarize_on_episode_end`): fires when **all** of
- `summarization_mode in {"episodic", "both"}`
- `len(_messages) >= 3`
- `info["episode_done"] is True`

The engine additionally requires `done is False` (don't summarize on the trajectory-terminating step).

Behavior when it fires:
1. **Budget check** — `response_token_len + summary_max_tokens + summ_instruction_len >= max_response_length` ⇒ **hard fail**: set `termination_reason = SUMMARIZATION_BUDGET_EXCEEDED`, break the trajectory loop. The post-loop zeros all step rewards on truncation reasons. `summ_instruction_len` uses the pre-tokenized length of the instruction that *would* be sent — `REFLECTIVE_SUMMARY_PROMPT` when `env.enable_reflection=True`, else `EPISODIC_SUMMARY_PROMPT`.
2. Call `_do_summarization(trigger="episode_end", use_reflective_prompt=<env.enable_reflection>)`.
3. Prompt selection:
   - `env.enable_reflection=True` → `REFLECTIVE_SUMMARY_PROMPT`.
   - otherwise → `EPISODIC_SUMMARY_PROMPT`.
4. If `_do_summarization` returns `None` (max_model_len overflow, empty output, exception) ⇒ **hard fail**: set `termination_reason = SUMMARIZATION_FAILED`, break. Same post-loop zeroing as above.
5. On success: `apply_summary` rewrites the agent history to `[system, summary_user_msg]`; the engine calls `env.start_new_episode()` to reset the inner env and append the new-episode initial observation **directly to `agent._messages`** (bypassing `update_from_env` to preserve the just-finished episode's success reward on the trajectory's last step). Reset `response_token_len = 0`.

**Bottom line for episodic mode (post-2026-04-26 fix):** failure paths terminate the trajectory cleanly via explicit termination reasons. Earlier behavior tried to soft-skip and continue, but that left the inner env in `done` state and crashed the next env.step; the cleanup also let trajectories run far past their natural budget, thrashing vLLM's KV cache at scale. Both modes now have identical hard-fail semantics on summarization failure, distinguished only by *when* they fire (token: any-step threshold; episodic: episode boundary).

---

## 4. Mode: `both`

Both predicates are active; the engine evaluates them in this priority order (`summarizing_engine.py:316`):

```python
if agent.should_summarize_on_episode_end(info):
    summ_trigger = "episode_end"
elif agent.should_summarize(tokenizer, chat_parser):
    summ_trigger = "token"
```

Consequences:
- On an episode-boundary step that also crossed the token threshold, the **episodic** path wins.
- On a mid-episode step that crossed the threshold, the **token** path fires.
- Failure semantics are identical between the two modes (hard fail with explicit termination reason; see §3 and §5).
- There is no global cap on the number of summarizations per trajectory — both triggers fire as often as their predicates say.

---

## 5. Side-by-side summary

| Dimension | `token` | `episodic` |
|---|---|---|
| Trigger | Accumulated chat-history tokens ≥ `threshold_tokens` (full `_messages` rendered through the chat template) | `info["episode_done"]=True` |
| Fires within an episode? | Yes, mid-episode | No, only at boundaries |
| Post-summary history | `[sys, summary]` — next env obs lands on the following iteration's `update_from_env` | `[sys, summary, new_episode_obs]` (via `env.start_new_episode`) |
| Prompt | `TOKEN_SUMMARY_PROMPT` | `REFLECTIVE_SUMMARY_PROMPT` if `env.enable_reflection` else `EPISODIC_SUMMARY_PROMPT` |
| Budget exhausted | Hard fail → `SUMMARIZATION_BUDGET_EXCEEDED`, trajectory truncated | Hard fail → `SUMMARIZATION_BUDGET_EXCEEDED`, trajectory truncated |
| Context-window overflow | Hard fail → `SUMMARIZATION_FAILED`, trajectory truncated | Hard fail → `SUMMARIZATION_FAILED`, trajectory truncated |
| Env-side change | None | `MultiEpisodeEnv.step` split + `start_new_episode()` advance |
| `env.reflection_via_summarization` | `False` | `True` (auto-set by trainer when mode ∈ {episodic, both}) |
| Typical effect of failure | Trajectory reward zeroed (`compute_trajectory_reward` zeros all step rewards on truncation termination reasons) | Same — both modes terminate the trajectory cleanly when summarization can't complete |

---

## 6. Trajectory-uniform PPO loss (segment training)

When `summarization.enable=true`, every trajectory may produce K≥1 segments. The training loss aggregates over all of them with the **trajectory-uniform** formula:

```
L  =  (1/N_G)  ·  Σ_{i=1..N_G}  (1/N_t^i)  ·  Σ_{token in trajectory i, mask=1}  ℓ_token
```

where `N_G` is the number of distinct trajectories in the batch and `N_t^i` is the total mask=1 token count of trajectory `i` (summed across all of its segment rows). Each trajectory contributes weight 1 to the gradient regardless of how many segments it has — so a trajectory that hit one summarization (K=2) doesn't get twice the gradient of a single-segment one.

### 6.1 Data flow

```
rollout (SummarizingAgentExecutionEngine)
    └── per-trajectory dict with `segments`: [seg0, seg1, …, seg_{K-1}]
         where each seg has prompt_tokens, response_tokens, response_masks
    │
MultiEpisodeAgentPPOTrainer._transform_agent_trajectories
    └── caches `self._cached_raw_trajectories` (full segments preserved)
    └── returns the per-trajectory DataProto (segment[0] only) for verl's
        existing fit_agent flow → compute_log_prob, compute_advantage land
        per-trajectory `advantages[i, t] = A_i` on segment[0]'s tokens
    │
fit_agent calls actor_rollout_wg.update_actor(per-trajectory batch)
    │
[wrapped]  MultiEpisodeAgentPPOTrainer._expanded_update_actor
    ├── extract_advantage_per_trajectory(advantages, response_mask) → A_i scalar per traj
    ├── build_expanded_dataproto(trajectories, A_i, …)  ───────── trainers/segment_expansion.py
    │       └── Σ K_i rows, each with:
    │             input_ids, attention_mask, position_ids, prompts,
    │             responses, response_mask, advantages = A_i broadcast to mask=1
    │             traj_uniform_weight[i_row, t] = 1 / (N_G · N_t^i)  for mask=1
    │                                            = 0                  elsewhere
    │             non_tensor: traj_idx (which trajectory each row belongs to)
    ├── pad_dataproto_to_divisor(expanded_batch, world_size)
    │       └── pads to multiple of FSDP world size; padded rows get
    │             traj_uniform_weight[-pad_size:] = 0  → 0 contribution to loss
    ├── compute_log_prob(expanded_batch)   ← one extra forward pass on Σ K_i rows
    └── original update_actor(expanded_batch)
            │
            ▼
[in workers]  TrajectoryUniformPPOActor.update_policy   ───────── trainers/trajectory_uniform_actor.py
    └── for each micro-batch:
          new_log_prob, _ = _forward_micro_batch(...)
          # PPO surrogate, identical to verl vanilla:
          ratio        = exp(new_log_prob - old_log_prob)
          pg_per_token = standard PPO clip on advantages × ratio
          # Trajectory-uniform aggregation:
          numer_micro  = (pg_per_token · traj_uniform_weight · response_mask).sum()
          loss_micro   = numer_micro · dp_world_size + entropy/kl terms
          loss_micro.backward()
          # FSDP averages gradients across DP ranks → the dp_world_size
          # multiplication cancels the mean and reproduces the global sum.
```

### 6.2 Why each piece is the way it is

- **Per-trajectory advantage stays unexpanded for GRPO.** GRPO group statistics (`(R_i - mean(R)) / std(R)`) need exactly `n` rollouts per UID. We compute advantages on the per-trajectory batch (one row per trajectory) so the group composition is unaffected by per-trajectory segment count. A scalar `A_i` is then broadcast to every valid token of every segment row of trajectory `i` — that's what `extract_advantage_per_trajectory` and the `advantages.masked_fill_(valid, A_i)` inside `build_expanded_dataproto` are doing.

- **`traj_uniform_weight` bakes in `1/(N_G · N_t^i)` at expansion time.** The actor's per-token loss is `pg_token · traj_uniform_weight · response_mask`. Summing over **all rows on all DP ranks** then yields the global trajectory-uniform formula, no further normalization required. Padded rows get 0 weight so their contribution is exactly 0 regardless of pg or mask values.

- **`dp_world_size` multiplication compensates for FSDP's gradient averaging.** Each rank's `loss_micro.backward()` produces a local gradient; FSDP/DDP all-reduces these as a **mean**. Multiplying the local loss by `dp_world_size` first turns mean back into sum — i.e. each rank's contribution adds linearly into the final gradient.

- **One extra forward pass on the expanded batch.** Each segment k≥1 has different prompt+response tokens than segment[0], so the per-trajectory `compute_log_prob` (which already ran in fit_agent) doesn't have valid `old_log_probs` for the additional segment rows. The wrapper re-runs `compute_log_prob` on the expanded batch to fill them in. This is the dominant overhead added by segment training; for K=1 trajectories the wrapper short-circuits and skips the re-run.

### 6.3 How `TrajectoryUniformPPOActor` reaches Ray workers

The actor is a subclass of verl's `DataParallelPPOActor` (in `trainers/trajectory_uniform_actor.py`) that overrides only `update_policy`. For verl's worker code to *use* our subclass instead of the stock one, we need to monkey-patch `verl.workers.actor.dp_actor.DataParallelPPOActor` **inside each Ray worker process** — driver-side patches don't propagate.

We do this via a `.pth` file in the conda env's `site-packages`, written automatically by `scripts/train_multi_task_multi_episode.sh` on every launch:

```
<repo_root>
import orbit_segtrain_patch
```

Python's `site` module auto-loads `.pth` files at interpreter startup in every Python process. The `orbit_segtrain_patch` module (top-level at repo root, **not** inside the `trainers/` package — see below) imports only `wrapt` and registers a lazy callback:

```python
@wrapt.when_imported("verl.workers.actor.dp_actor")
def _patch(mod):
    from trainers.trajectory_uniform_actor import TrajectoryUniformPPOActor
    mod.DataParallelPPOActor = TrajectoryUniformPPOActor
```

The callback fires only when `verl.workers.actor.dp_actor` is first imported (inside `ActorRolloutRefWorker.init_model`), by which point Ray's per-worker GPU setup is complete. After the patch fires, every subsequent `from verl.workers.actor.dp_actor import DataParallelPPOActor` returns our subclass; the worker instantiates `TrajectoryUniformPPOActor` and `update_policy` runs the trajectory-uniform aggregation.

Three reasons we use a `.pth` file rather than Ray's own `runtime_env.worker_process_setup_hook`:
1. Setting any `worker_process_setup_hook` on this SCC cluster breaks Ray's per-worker GPU isolation; workers fail at `init_device_mesh` with "CUDA-capable device(s) is/are busy or unavailable" even with an empty hook (confirmed via empty-hook bisection).
2. The `.pth` mechanism is standard Python (`site.py`) and works in every process spawned in the env — including processes that would never see Ray.
3. It's idempotent and zero-setup for a new collaborator: `bash scripts/train_*.sh` writes the `.pth` into their conda env on first launch.

`orbit_segtrain_patch.py` is intentionally a **top-level module at repo root**, not inside `trainers/`, because importing `trainers.<anything>` would load `trainers/__init__.py` — which we keep deliberately empty (see its docstring) precisely so the patch's import doesn't cascade into `multi_episode_trainer` → `rllm` → `rllm.rewards.code_utils.taco` → `signal.signal(SIGALRM, …)` (which raises `ValueError: signal only works in main thread` when invoked from the wrapt callback's import-chain thread inside a Ray worker).

### 6.4 What gets `response_mask=1` in each segment

`response_mask` defines which tokens in a segment's `response_tokens` contribute to the PPO loss (i.e. which tokens count toward `N_t^i` in the trajectory-uniform formula). Per-segment, the mask is computed by the engine via `assemble_steps` (in `third_party/rllm/rllm/engine/agent_execution_engine.py:459-520`), called once per segment by `assemble_segments` in `trainers/summarizing_engine.py:770-837`.

**Two fields per segment**: each segment dict produced by `assemble_segments` has both:

- **`prompt_tokens`**: the conditioning context the model sees *before* generating the first action of this segment. This is **never masked** — PPO loss never applies to prompt positions, so there's no `prompt_mask`. This always contains, at minimum, the system prompt and the initial observation the model conditioned on at the start of the segment. For segments after the first, it also contains the previous segment's summary (as a `user`-role message — the output of `apply_summary`).
- **`response_tokens`**: everything from the model's first generated token of this segment onward, in interleaved order — model action completions, env observations between actions, optionally a summarization instruction + summarization completion at the end. The `response_mask` is over this field, marking which positions contribute to loss.

By definition, `response_tokens` always *starts* with the model's first action completion in the segment (mask=1) — anything before the first generated token is in `prompt_tokens`, not in `response_tokens`. So the system prompt, the inherited summary, and the new-episode initial obs all live in `prompt_tokens`, not in `response_tokens`.

**Mask=1** (token contributes to gradient) — the model's *generated* tokens:

- **Assistant action completions**: every `completion_ids` returned by vLLM for a model action call within the segment. These are the action tokens the policy is being trained to produce.
- **Summarization completion** (only on a segment that ends with a summarization, i.e. segments 0..K-2 in a K-segment trajectory): the summary text itself is the completion of an `is_summarization=True` step. The summarization instruction message preceding it is mask=0; the model's summary output is mask=1. So the model is also trained to produce summaries — they're not just a context-compression utility, they're a learned skill that contributes gradient.

**Mask=0** (input context, gradient is masked out) — tokens the model didn't generate but conditioned on:

- **Initial system prompt** (entirely; lives in `prompt_tokens`, not `response_tokens`).
- **Env observations** between assistant actions: when the engine assembles step k>0, the `prompt_diff = current_prompt_ids[len(accumulated):]` is the env-message tokens that arrived between actions. These get `[0] * len(prompt_diff)`.
- **Summarization instruction tokens** (e.g. `EPISODIC_SUMMARY_PROMPT`, `TOKEN_SUMMARY_PROMPT`, `REFLECTIVE_SUMMARY_PROMPT`): when the engine inserts the summarization step into `episode_steps`, the instruction is part of `prompt_ids` for that step (the `prompt_diff` between the prior accumulated history and this step's prompt). It gets mask=0.
- **Initial-episode observation that lands AFTER an episodic summarization**: the engine appends the new episode's first observation directly to `agent._messages` and to `response_tokens` with mask=0 (line 460-462 of `summarizing_engine.py`). Same convention as a regular env observation.

**Mask=0 fallback** — the entire response_mask of a segment is zeroed when:

- Token-mismatch is detected during `assemble_steps` (model's prompt_ids don't share a prefix with what the engine accumulated → `is_valid_trajectory=False`, all masks zeroed). Indicates a tokenizer / chat-template inconsistency.
- The trajectory was truncated for `TRUNCATION` / `PROMPT_TRUNCATION` / `SUMMARIZATION_BUDGET_EXCEEDED` / `SUMMARIZATION_FAILED`, AND `overlong_filter=True`. Then `summarizing_engine.py` zeros the mask in the post-loop block. (Default `overlong_filter=False`, so this rarely fires.)

**`N_t^i`** for trajectory `i` is then `sum_{k in 0..K_i-1} sum_t segment_k.response_masks[t]` — the count of mask=1 positions across all of trajectory `i`'s segment rows. The trajectory-uniform weight is `1/(N_G · N_t^i)` per mask=1 token.

**Concrete example**: trajectory with 3 episodes (so 2 episodic summarizations, K=3 segments) where each episode is 4 model actions. Each segment is a `(prompt, response)` pair — the prompt is the conditioning input (no mask, never contributes to loss), the response is what the engine assembled with `response_mask`:

```
═══════════════════════════════════════════════════════════════════════════════════════════
SEGMENT 0
─────────
PROMPT (no mask):
  sys_prompt  initial_obs_episode_1  gen_prompt_token

RESPONSE (with response_mask):
  action_0_completion  env_msg_1  action_1_completion  env_msg_2  action_2_completion  env_msg_3  action_3_completion  env_msg_4  summary_instruction  summary_completion
        1                 0                1               0                1              0                1               0              0                    1
     (model)           (env obs)        (model)         (env obs)         (model)        (env obs)        (model)         (env obs)    (engine's "please       (model's
                                                                                                                                       summarize" prompt        summary
                                                                                                                                       — mask=0 because the      output —
                                                                                                                                       model didn't gen it)      mask=1)

N_t(seg 0) = action_0 + action_1 + action_2 + action_3 + summary_completion lengths

═══════════════════════════════════════════════════════════════════════════════════════════
SEGMENT 1
─────────
PROMPT (no mask):
  sys_prompt  summary_user_msg  initial_obs_episode_2  gen_prompt_token
              ↑ from segment 0's         ↑ from env.start_new_episode()
                apply_summary

RESPONSE (with response_mask):
  action_0_completion  env_msg_1  action_1_completion  …  action_3_completion  env_msg_4  summary_instruction  summary_completion
        1                 0                1            …          1               0                0                    1

N_t(seg 1) = action_0 + … + action_3 + summary_completion lengths

═══════════════════════════════════════════════════════════════════════════════════════════
SEGMENT 2  (final segment — no summarization at the end)
─────────
PROMPT (no mask):
  sys_prompt  summary_user_msg  initial_obs_episode_3  gen_prompt_token

RESPONSE (with response_mask):
  action_0_completion  env_msg_1  action_1_completion  env_msg_2  action_2_completion  env_msg_3  last_action_completion
        1                 0                1               0                1              0                1

N_t(seg 2) = action_0 + action_1 + action_2 + last_action lengths

═══════════════════════════════════════════════════════════════════════════════════════════
N_t^i = N_t(seg 0) + N_t(seg 1) + N_t(seg 2)
```

Two non-obvious points worth flagging:

1. **The summary appears in TWO places per K-segment trajectory** (for k = 0..K-2): in segment k's RESPONSE with mask=1 (where the model generated it; this is what trains the model to produce summaries), AND in segment k+1's PROMPT (as conditioning context for the next episode; no mask, prompts don't contribute to loss). It's *not* in segment k+1's response.

2. **The new episode's initial observation appears in segment k+1's PROMPT** (since the model's next forward pass conditions on `[sys, summary, new_obs]`). At engine runtime, the new-episode obs gets briefly appended to a running `response_tokens` accumulator with mask=0, but `assemble_segments` re-walks `episode_steps` from scratch per segment, and segment k+1's first step's `prompt_ids` already contains the new-episode obs as part of the full prompt context — so in the assembled per-segment dict, it lands in `prompt_tokens`.

### 6.5 Where the pieces live

| File | Role |
|---|---|
| `trainers/trajectory_uniform_actor.py` | `TrajectoryUniformPPOActor` subclass + the loss math. |
| `trainers/segment_expansion.py` | `build_expanded_dataproto`, `extract_advantage_per_trajectory`. |
| `trainers/multi_episode_trainer.py:_install_segment_aware_update_actor` | Wraps `actor_rollout_wg.update_actor` to expand → pad → compute_log_prob → call original. |
| `orbit_segtrain_patch.py` | Top-level module loaded by the .pth file; registers the wrapt callback. |
| `scripts/train_multi_task_multi_episode.sh` | Writes the `.pth` file into the active conda env on every launch. |
| `tests/test_segment_expansion.py` | 12 tests for `build_expanded_dataproto` invariants (shape, weight tensor, padding semantics, reward broadcast). |
| `tests/test_trajectory_uniform_loss_e2e.py` | 6 tests verifying the full pipeline `raw trajectories → expansion → weighted-sum aggregation` matches the closed-form `(1/N_G) Σ_i (1/N_t^i) Σ_token ℓ_token` on synthetic batches. |
