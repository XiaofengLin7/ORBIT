# Prompt Behavior: System Prompt + Carryover Protocol

Scope: what the model actually sees during multi-episode training and eval — the system prompt produced by `prompts/system_prompts.py`, and the user messages produced by `agents/context_summarizer.py` (`apply_summary` / `apply_episode_carryover`) when summarization / carryover fires.

Read alongside `docs/summary.md` (mode-level trigger logic) and the "Context Self-Summarization" section of `CLAUDE.md`.

---

## 1. System prompt builder

Single source: `prompts/system_prompts.py:build_multi_episode_system_prompt(summarization=..., single_episode=...)`.

Four entry points call it:

| Entry point | Args |
|---|---|
| `scripts/train_multi_episode.py` | `summarization=<resolved rllm.agent.summarization>` |
| `scripts/train_multi_task_single_episode.py` | `single_episode=True` |
| `scripts/train_gem_multi_episode.py` | (none — no summarization configured) |
| `scripts/eval/openai.py:get_default_system_prompt` | `summarization={...}` from `--summarization-threshold`, `--summarization-mode`, `--episodic-carryover`, `--oracle-summarizer` |

User overrides via `rllm.agent.agent_args.system_prompt` (training) or `--system-prompt` (eval) bypass the builder entirely.

### Base prompt (always present)

Multi-episode (`single_episode=False`):
```
You are solving the same task across {num_episodes} episodes with a fixed total step budget.
Each episode resets the environment but keeps the task identical.
You will interact with the environment for exactly {num_episodes} episodes -
use earlier episodes to gather information so later episodes succeed faster.
Respond with your action inside \boxed{} each turn.
```

Single-episode:
```
You are solving a task in a single episode.
Analyze the situation carefully and take the best actions to succeed.
Respond with your action inside \boxed{} each turn.
```

The `{num_episodes}` placeholder is substituted by `GEMTextAgent._substitute_num_episodes_on_first_env_step` using `info["num_episodes"]` exposed by `MultiEpisodeEnv` — once, on the first `update_from_env`. Tasks that don't set `num_episodes` leave the placeholder literal.

Brevity / length-penalty language ("Think briefly", "Overlong responses will be penalized") was removed in 2026-05; the `\boxed{}` format instruction remains because the parser keys off it.

### Memory-protocol section (appended when summarization is enabled)

Triggered when `summarization["enable"] == True`. Layout:

```
[base prompt]

Memory protocol. Your conversation history may be compressed within or between
episodes. When that happens, the compressed form appears as a USER message
containing `<context_summary>...</context_summary>` or `<reflection>...</reflection>`
tags — that content is YOUR OWN prior memory carried forward, NOT a new
instruction from the environment. The marker `[Continuing task]` follows the
compressed block; the next env observation arrives as the next user message.
- <mode-specific section(s)>
```

Section selection logic (in order):

1. `mode in ("token", "both")` → append the **token** section.
2. `mode in ("episodic", "both")` → append exactly one of:
   - **oracle** if `summarization.oracle.enable=True` (wins over all carryover modes — matches engine precedence in `_do_summarization`).
   - **obs_action** if `episodic_carryover="obs_action"`.
   - **obs_action_reflection** if `episodic_carryover="obs_action_reflection"`.
   - **freeform** otherwise (the default).

So `mode=both` emits both a mid-episode section and a between-episodes section. `mode=token` emits only the mid-episode section. `mode=episodic` emits only the between-episodes section.

Defaults if missing: `mode="episodic"`, `episodic_carryover="freeform"`, `oracle.enable=False` (matches the agent's `__init__` defaults in `ContextSummarizerMixin`).

### Section bodies (verbatim, for reference)

The protocol **header** documents the markers that prefix every compressed block: `[End of episode K/N]` and `[Mid-episode K/N compression]`, with K defined as the 1-based episode index and N as the total (dropped when unknown). It also defines `[Continuing task]` as the boundary marker.

Per-mode bodies (each reiterates the relevant marker so the body and header agree):

- **token**: `- Mid-episode (token threshold): when your context grows large you will be asked to write a <context_summary> block. That block, prefixed with [Mid-episode K/N compression], replaces your history and you continue the same episode from it.`
- **freeform**: `- Between episodes: you will be asked to write a <context_summary> block. The next episode starts with that block, prefixed with [End of episode K/N], as your only memory of earlier episodes. Make it useful to your future self — preserve task rules, confirmed facts, ruled-out hypotheses, and a concrete plan.`
- **obs_action**: `- Between episodes: your prior history is reduced to the previous episode's observations paired with \boxed{action} only; the thinking around each action is dropped to save context. A trailing user message [End of episode K/N] — [Continuing task] marks the boundary before the next episode's first observation. This is a memory-compression artifact, NOT a format change — continue to think before each new action.`
- **obs_action_reflection**: `- Between episodes: your prior history is reduced to the previous episode's observations paired with \boxed{action} (the thinking around each action is dropped to save context), followed by a trailing user message [End of episode K/N]\n<reflection>...</reflection>\n\n[Continuing task] — you write the reflection. The stripped transcript is a compression artifact, NOT a format change — continue to think before each new action.`
- **oracle**: `- Between episodes: a deterministic environment-derived note appears inside [End of episode K/N]\n<context_summary>...</context_summary> in place of a summary you write. Treat it as reliable factual state about the task — the system provides it, you do not write it.`

The protocol header explicitly states that the compressed user message is the model's **own prior memory** — this addresses the role-inversion concern (the model writes the summary as an assistant turn, then sees it back as a user turn in the next episode).

The `obs_action*` sections explicitly say "NOT a format change — continue to think before each new action." This counteracts the demonstration mismatch: prior assistant turns in the carried-over transcript have been reduced to `\boxed{action}` only (thinking discarded by `ContextSummarizerMixin._boxify_action_turns`), but the policy must keep producing thinking + boxed action on new turns.

---

## 2. Carryover user messages (what the model sees back in the next episode)

Each carryover mode rewrites `agent._messages` and inserts a user message that carries the just-finished episode's content forward. All variants include an episode label sourced from `env._episode_index` (0-based, rendered 1-based) and `env.num_episodes` (optional).

Engine wiring: `trainers/summarizing_engine.py:_do_summarization` reads `env_episode_index = getattr(env, "_episode_index", None)` and `env_num_episodes = getattr(env, "num_episodes", None)` once at the top, then forwards both to every `apply_summary` / `apply_episode_carryover` call site (oracle path, freeform path, carryover path).

### Episode label format

Helper: `ContextSummarizerMixin._format_episode_label(episode_index, num_episodes)`.

| `episode_index` | `num_episodes` | Rendered label |
|---|---|---|
| 0 | 3 | `1/3` |
| 0 | None | `1` |
| None | * | `""` (empty — caller falls back) |

`K` (episode index) is always available on `MultiEpisodeEnv` — initialized to 0 in `__init__`, bumped on each boundary. At `episode_end` trigger time it's the **just-finished** episode (`MultiEpisodeEnv` defers the bump until `start_new_episode` is called by the engine after summarization). At `token` trigger time it's the **currently-running** episode.

`N` (total episodes) is the per-task config field `num_episodes` from the tasks YAML. Set on maze / summarization configs (e.g. `configs/multi_task_summarization_config.yaml`, `configs/maze.yaml`); not set on `configs/multi_task_multi_episode_config.yaml`, `configs/eval_alfworld_multi.yaml`, `configs/eval_webshop_multi.yaml`. When unset, only `K` is shown.

`SingleEpisodeEnv` doesn't expose either attribute; `getattr(..., None)` returns None and the label is dropped — `apply_summary` omits the header; `apply_episode_carryover` falls back to `[End of previous episode]`.

### Message shapes by mode

#### `freeform` / `oracle`, `trigger=episode_end`

Post-rewrite layout:
```
[system_prompt,
 {role: user, content: "[End of episode K/N]
<context_summary>
…summary body…
</context_summary>

[Continuing task]"}]
```

For oracle: the body is the env-derived string from `agents/oracle_summarizers/`. Not a trainable step.
For freeform: the body is the LLM's generation. Trainable — `episode_steps` records the step with `is_summarization=True`, the trajectory's GRPO advantage is broadcast across the completion tokens.

#### `freeform`, `trigger=token` (mid-episode)

Same shape, but the header changes:
```
[Mid-episode K/N compression]
<context_summary>...</context_summary>

[Continuing task]
```

The model can distinguish a between-episodes summary from a within-episode compression from the header alone.

#### `obs_action`, `trigger=episode_end`

Post-rewrite layout (transcript kept verbatim, thinking stripped):
```
[system_prompt,
 user: obs_1,
 assistant: \boxed{a_1},
 user: obs_2,
 assistant: \boxed{a_2},
 …,
 user: terminal_obs,
 user: "[End of episode K/N] — [Continuing task]"]
```

Trailing user message is a pure boundary marker — no `<context_summary>` or `<reflection>` block, no LLM call, no trainable step.

#### `obs_action_reflection`, `trigger=episode_end`

Same kept transcript + a trailing user message that combines the episode label, the reflection, and the continuation marker:
```
user: "[End of episode K/N]
<reflection>
…reflection body…
</reflection>

[Continuing task]"
```

The reflection is generated via `REFLECTION_PROMPT` (in `prompts/summarization_prompts.py`) and IS a trainable step. If reflection generation fails or has no token room, it silently degrades to plain `obs_action`.

### Legacy / missing inputs

`apply_summary` and `apply_episode_carryover` accept `episode_index=None, num_episodes=None` as defaults. Behavior in that case:

- `apply_summary`: header is dropped entirely; the user message starts with `<context_summary>` as before.
- `apply_episode_carryover` (obs_action): trailing message becomes `[End of previous episode] — [Continuing task]`.
- `apply_episode_carryover` (obs_action_reflection): trailing message starts with `[End of previous episode]\n<reflection>…`.

This keeps unit tests that don't wire up an env from breaking, and is the path used by any direct caller that doesn't have env state.

---

## 3. Coordination with engine state

The system prompt advertises a protocol the engine must actually implement. The wiring that keeps these in sync:

- `scripts/train_multi_episode.py` reads `cfg.rllm.agent.summarization` first, then passes it both to `build_multi_episode_system_prompt` (to compose the prompt) AND to the agent via `agent_args` (so `ContextSummarizerMixin.__init__` configures the same `mode` / `episodic_carryover`). One config drives both.
- `scripts/eval/openai.py` does the same: `--summarization-mode` and `--episodic-carryover` flags flow into both `get_default_system_prompt(...)` and `agent_args["mode"]` / `agent_args["episodic_carryover"]`. The engine's `_do_summarization` then dispatches on the same `carryover` value.
- The oracle path in `_do_summarization` short-circuits before the carryover branch, matching the system-prompt logic that gives oracle precedence over all carryover modes.

If the system prompt and the engine ever disagree (e.g. prompt says "you will write a `<context_summary>`" but engine is configured with `obs_action`), the policy sees a contradictory contract — sample efficiency drops. The shared-config wiring above is what prevents that.

---

## 4. Failure / fallback table

| Condition | Effect |
|---|---|
| `summarization.enable=False` or None | Base prompt only; no carryover ever fires. |
| `env._episode_index` missing | Episode label dropped (`apply_summary` omits header; carryover uses `[End of previous episode]`). |
| `env.num_episodes` missing | Label is just `K` instead of `K/N`. |
| Oracle returns empty / raises | Engine returns `summ_result=None`; trajectory terminates with `SUMMARIZATION_FAILED`. |
| `obs_action_reflection` reflection generation has no token room | Silently degrades to plain `obs_action` (no trailing reflection block, just the boundary marker). |
| Summarization budget exhausted (`response_token_len + summary_budget ≥ max_response_length`) | Trajectory terminates with `SUMMARIZATION_BUDGET_EXCEEDED` before any rewrite. |

---

## 5. Where to make changes

| To change | Edit |
|---|---|
| Base prompt wording | `prompts/system_prompts.py:_BASE_MULTI_EPISODE` / `_BASE_SINGLE_EPISODE` |
| Protocol section wording | `prompts/system_prompts.py:_PROTOCOL_*` constants |
| Selection logic | `prompts/system_prompts.py:build_multi_episode_system_prompt` |
| Carryover user-message format | `agents/context_summarizer.py:apply_summary` / `apply_episode_carryover` |
| Episode label format | `agents/context_summarizer.py:_format_episode_label` / `_episode_header` |
| What gets threaded from env | `trainers/summarizing_engine.py:_do_summarization` (top of function) |
| Summarization instruction itself (asked of the model) | `prompts/summarization_prompts.py` (`TOKEN_SUMMARY_PROMPT`, `EPISODIC_SUMMARY_PROMPT`, `REFLECTIVE_SUMMARY_PROMPT`, `REFLECTION_PROMPT`) |

Tests:
- `tests/test_system_prompts.py` — every protocol branch + base + defaults.
- `tests/test_context_summarizer.py:TestApplySummaryLayout` — `apply_summary` headers (episode_end / token / no-total / missing-index).
- `tests/test_context_summarizer.py:TestEpisodicCarryover` — `apply_episode_carryover` layouts for `obs_action` and `obs_action_reflection`, with and without env wiring.
