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

### Diagnostic messages (token)

| Event | Color | Message | Effect |
|---|---|---|---|
| Summary applied | cyan | `Summarization #N applied (trigger=token, step K, summary_tokens=T). Messages: {...}` | Continue from `[sys, summary]`; the next env obs is appended on the following step's `update_from_env`. |
| Budget too tight at trigger time | **red** | `Trajectory X: summarization skipped — response budget exhausted (used/max).` | `termination_reason = SUMMARIZATION_BUDGET_EXCEEDED`, break, rewards zeroed. |
| Summary prompt would overflow `max_model_len` | **red** | `Summarization prompt (N tokens) exceeds max_model_len (M). Skipping summarization.` (inside `_do_summarization`) → then `Trajectory X: summarization failed.` (outer). | `termination_reason = SUMMARIZATION_FAILED`, break, rewards zeroed. |
| Summary `max_tokens` clamped to fit | yellow | `Summarization headroom limited to N tokens (requested M).` | Generation proceeds; summary may be shorter / truncated. |
| Generation exception | **red** | `Summarization generation failed: <exception>` | Returns `None` → `SUMMARIZATION_FAILED`. |
| Zero-length completion | **red** | `Summarization produced empty output. Skipping.` | Returns `None` → `SUMMARIZATION_FAILED`. |
| Trajectory end | **red** | `Trajectory X is truncated (SUMMARIZATION_BUDGET_EXCEEDED). Trajectory reward is 0.0.` | Trajectory marked bad for aggregation. |

**Bottom line for token mode:** any summarization failure is fatal — the `termination_reason` lands in `truncation_reasons` and all per-step rewards are zeroed in the post-loop.

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
      │ yes → yellow "episodic summarization skipped — response budget exhausted"
      │       continue trajectory (soft skip)
      │ no
      ▼
_do_summarization(trigger="episode_end",
                  use_reflective_prompt=env.enable_reflection)
      │
      ▼
summary LLM returned None?   (max_model_len overflow, empty output, exception)
      │ yes → yellow "episodic summarization skipped (context window would be exceeded)"
      │       continue trajectory (soft skip)
      │ no
      ▼
apply_summary → [sys, summary]
env.start_new_episode() → new_obs
update_from_env(new_obs) → [sys, summary, new_episode_obs]
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
1. **Budget check** — `response_token_len + summary_max_tokens + summ_instruction_len >= max_response_length` ⇒ **soft skip** (yellow log, continue trajectory). `summ_instruction_len` uses the pre-tokenized length of the instruction that *would* be sent — `REFLECTIVE_SUMMARY_PROMPT` when `env.enable_reflection=True`, else `EPISODIC_SUMMARY_PROMPT`.
2. Call `_do_summarization(trigger="episode_end", use_reflective_prompt=<env.enable_reflection>)`.
3. Prompt selection:
   - `env.enable_reflection=True` → `REFLECTIVE_SUMMARY_PROMPT`.
   - otherwise → `EPISODIC_SUMMARY_PROMPT`.
4. Post-summary layout via `apply_summary`: `[system, summary_user_msg]` (2 messages).
5. Engine calls `env.start_new_episode()` → resets inner env, bumps `episode_index`, returns new episode's initial obs.
6. Engine calls `agent.update_from_env(new_obs, …)` → history becomes `[system, summary, new_episode_obs]`.
7. Reset `response_token_len = 0`.

### Diagnostic messages (episodic)

| Event | Color | Message | Effect |
|---|---|---|---|
| Summary applied | cyan | `Summarization #N applied (trigger=episode_end, step K, summary_tokens=T). Messages: {...}` | Continue from `[sys, summary, new_episode_obs]`; trajectory proceeds into the next episode. |
| Budget too tight at trigger time | yellow | `Trajectory X: episodic summarization skipped — response budget exhausted (used/max).` | Soft skip. Trajectory continues uncompressed. |
| `_do_summarization` returned `None` | yellow | `Trajectory X: episodic summarization skipped (context window would be exceeded).` | Soft skip. |
| Sub-warnings from `_do_summarization` | red / yellow | Same set as token mode (`Summarization prompt … exceeds …`, `headroom limited …`, `generation failed …`, `empty output …`) | Only the first two cause `_do_summarization` to return `None`, which manifests as the yellow "skipped" line rather than a fatal error. |

**Bottom line for episodic mode:** summarization failure is never fatal for the trajectory. The trajectory just carries its uncompressed history into the next episode and may later trip `TRUNCATION` / `MAX_STEPS` (which are orthogonal termination reasons).

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
- On an episode-boundary step that also crossed the token threshold, the **episodic** path wins (soft-fail semantics).
- On a mid-episode step that crossed the threshold, the **token** path fires (hard-fail semantics).
- There is no global cap on the number of summarizations per trajectory — both triggers fire as often as their predicates say.

---

## 5. Side-by-side summary

| Dimension | `token` | `episodic` |
|---|---|---|
| Trigger | Accumulated chat-history tokens ≥ `threshold_tokens` (full `_messages` rendered through the chat template) | `info["episode_done"]=True` |
| Fires within an episode? | Yes, mid-episode | No, only at boundaries |
| Post-summary history | `[sys, summary]` — next env obs lands on the following iteration's `update_from_env` | `[sys, summary, new_episode_obs]` (via `env.start_new_episode`) |
| Prompt | `TOKEN_SUMMARY_PROMPT` | `REFLECTIVE_SUMMARY_PROMPT` if `env.enable_reflection` else `EPISODIC_SUMMARY_PROMPT` |
| Budget exhausted | Hard fail → `SUMMARIZATION_BUDGET_EXCEEDED` (red, truncation) | Soft skip (yellow) |
| Context-window overflow | Hard fail → `SUMMARIZATION_FAILED` (red, truncation) | Soft skip (yellow) |
| Env-side change | None | `MultiEpisodeEnv.step` split + `start_new_episode()` advance |
| `env.reflection_via_summarization` | `False` | `True` (auto-set by trainer when mode ∈ {episodic, both}) |
| Typical effect of failure | Trajectory reward zeroed even if last env step succeeded | No reward impact from summarization; trajectory continues |

---

## 6. Observed behavior (meta-rl-summarization-eval benchmark)

Data source: `checkpoints/meta-rl-summarization-eval/*/chat_completions/0.jsonl` (base vs ORBIT Qwen3-8B, each under `token` and `episodic`, on maze + mastermind with 3 and 5 episodes).

### 6a. How often does each mode actually fire?

Segment-count histogram per run (one segment = zero summaries; N segments = N−1 summarizations applied):

| Run | 1 seg | 2 seg | 3 seg | 4 seg | 5 seg | 6+ seg |
|---|---|---|---|---|---|---|
| base \| episodic | 4 | 2 | 252 | 1 | 253 | 0 |
| base \| token | 499 | 11 | 2 | 0 | 0 | 0 |
| orbit \| episodic | 85 | 63 | 186 | 3 | 175 | 0 |
| orbit \| token | 262 | 180 | 49 | 20 | 0 | 1 |

- **base | episodic** and **orbit | episodic** concentrate at 3 and 5 segments, aligned with the 3- and 5-episode horizons. Episodic fires reliably at each boundary.
- **base | token**: 499 / 512 trajectories had one segment — the token path essentially never fires, either because the base model's context growth doesn't reach 16 k before the trajectory ends, or because it trips `SUMMARIZATION_BUDGET_EXCEEDED` on the one time it tries.
- **orbit | token**: a long tail into 3+ segments, but 262 / ≈900 still end with zero summaries. ORBIT tolerates the token path better but doesn't always exercise it.

### 6b. Summary length distribution

| Run | n summaries | mean chars | max chars | runaway (>3k or contains `<think>`) |
|---|---|---|---|---|
| base \| episodic | 1521 | 611 | 2 677 | 1 |
| base \| token | 15 | 784 | 1 025 | 0 |
| orbit \| episodic | 1144 | 612 | **18 587** | 2 |
| orbit \| token | 596 | **2 371** | **30 520** | **28** |

- **Episodic summaries** cluster tightly around 500–800 chars regardless of model — the compression prompt works as intended at episode boundaries.
- **Token summaries from ORBIT** have a wide distribution with catastrophic outliers reaching 30 000+ characters. These are Qwen3's `<think>` reasoning bleeding into the `<context_summary>` tag when summarization fires mid-episode.
- **Token summaries from base** are rare but well-behaved when they do fire.

### 6c. Inferred outcome (proxy: last-episode env success message)

Success rate = fraction of trajectories whose final env message matches a success phrase (`Congratulations … goal`, `correct code`, etc.). Rough proxy — it undercounts trajectories that succeeded mid-way but failed on the last episode, and it doesn't distinguish reward-zeroing truncations from real failures.

| Run | mastermind ep3 | mastermind ep5 | maze ep3 | maze ep5 |
|---|---|---|---|---|
| base \| episodic | 58 % | 65 % | 38 % | 27 % |
| base \| token | 35 % | 27 % | 21 % | 20 % |
| orbit \| episodic | 41 % | 38 % | 46 % | 46 % |
| orbit \| token | **69 %** | **72 %** | 46 % | 48 % |

- **base + token is catastrophic** — hard-fail + no working summarization → low rates across the board.
- **base + episodic vs ORBIT + episodic** — surprisingly close or even inverted on mastermind. Episodic mode's compression may strip too much reasoning for a deductive task.
- **ORBIT + token wins on mastermind** (~70 %) despite producing the bloated summaries. The bloat contains the peg-elimination reasoning that the solver *needs* — the prompt says "no step-by-step reasoning" but the model ignores it, and that happens to help.
- **Maze outcomes** are nearly identical between the two ORBIT modes: the task rewards exploration memory more than reasoning compression.

---

## 7. Failure modes catalog

1. **"Runaway thinking" in token mode (orbit-dominated).** Mid-episode summary interrupts Qwen3 mid-reasoning; the model stuffs its `<think>…</think>` trace inside the `<context_summary>` tag. Produces 10 k–30 k-char summaries. Occurs in ~11 % of token-path summaries on orbit+mastermind.
2. **Degenerate empty summary (rare, episodic).** Observed once (body = `"..."`). Harmless in aggregate.
3. **Threshold never trips (base + token).** Base-model trajectories don't grow to 16 k fast enough without the *one* growth spike that simultaneously trips the threshold AND the budget guard. Net effect: token mode = 0 summaries on base + maze.
4. **Budget-exhausted hard-fail (token).** Even when the threshold trips, if the step that tripped it was a big one (long assistant response), the budget guard simultaneously triggers `SUMMARIZATION_BUDGET_EXCEEDED` and zeroes the trajectory — **even if the final env step succeeded with reward 1.0**. The "Reward is 1.0" log line is the last per-step env reward, not the final trajectory reward (which gets zeroed in the post-loop).

---

## 8. Reading the logs — quick reference

| You see | Meaning | Action |
|---|---|---|
| **cyan** `Summarization #N applied (trigger=…)` | Success | Nothing. |
| **yellow** `episodic summarization skipped …` | Soft skip (only episodic mode) | Note for diagnostics — usually means budget is too tight for the summary. |
| **red** `summarization skipped — response budget exhausted …` | Hard fail (token path only) | Trajectory is about to be zeroed. Consider lowering `SUMMARY_MAX_TOKENS` or `SUMMARIZATION_THRESHOLD`. |
| **red** `summarization failed.` | Hard fail (token path only) | Same outcome. Usually preceded by the context-window message from `_do_summarization`. |
| **red** `Trajectory X is truncated (SUMMARIZATION_BUDGET_EXCEEDED\|SUMMARIZATION_FAILED). Trajectory reward is 0.0.` | Post-loop confirmation | The trajectory no longer contributes real reward signal. |

---

## 9. Design implications

1. **Best mode is task-dependent.** For exploration-heavy tasks (maze), episodic is clearly preferable. For deductive tasks (mastermind), the "no step-by-step reasoning" instruction in `EPISODIC_SUMMARY_PROMPT` (and its token-path counterpart) strips state the solver actually needs — either token mode or a less aggressive episodic prompt works better.
2. **Mid-episode summarization with reasoning models is unreliable.** Qwen3's `<think>` bleed is a model-behavior issue, not an engine bug. A cleaner fix would be to disable thinking for summarization calls (`disable_thinking=True` on the summary LLM call only), or to use a lighter non-reasoning variant for summaries.
3. **Length bounds are advisory.** `summary_max_tokens=4096` doesn't prevent 30 k-char outputs, because character count ≠ token count for repetitive reasoning traces and because the cap measures completion tokens, not the blended `<think>` + `<context_summary>` block.
4. **Base + token is currently degenerate.** At the 16 k threshold and 31 k response length, the safe zone is only ~7 k tokens wide; any one long step crosses both thresholds. Either lower the threshold (`SUMMARIZATION_THRESHOLD=8192`) or accept that episodic is the only working mode for base.
5. **Episodic never penalizes successful trajectories the way token does.** For eval, this alone is a reason to prefer episodic — `SUMMARIZATION_BUDGET_EXCEEDED` can destroy a reward-positive trajectory's contribution despite the env having already rewarded the final action.
