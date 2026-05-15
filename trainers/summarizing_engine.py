"""Execution engine with mid-trajectory context summarization.

Overrides the monolithic ``run_agent_trajectory_async`` loop from rLLM's
``AgentExecutionEngine`` to insert a summarization check after every
``agent.update_from_model()`` call.  When the agent's context exceeds a
configured token threshold, the engine generates a summary via the same
model and replaces the agent's message history with the compressed form.

The trajectory is split into **segments** at summarization boundaries.
Each segment is an independent training sample assembled via the parent's
``assemble_steps``.  The summarization output itself is trained on
(mask=1).  All segments share the same trajectory reward.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid

import torch

from rllm.agents.agent import Action, Trajectory
from rllm.agents.utils import (
    convert_messages_to_tokens_and_masks,
    get_recent_assistant_user_messages,
)
from rllm.environments.env_utils import compute_mc_return, compute_trajectory_reward
from rllm.utils import colorful_print

from agents.oracle_summarizers import get_oracle_summarizer
from prompts.summarization_prompts import (
    EPISODIC_SUMMARY_PROMPT,
    REFLECTION_PROMPT,
    REFLECTIVE_SUMMARY_PROMPT,
    TOKEN_SUMMARY_PROMPT,
)
from trainers.multi_episode_trainer import MultiEpisodeAsyncAgentExecutionEngine

logger = logging.getLogger(__name__)


class SummarizingAgentExecutionEngine(MultiEpisodeAsyncAgentExecutionEngine):
    """Execution engine that compresses context mid-trajectory.

    When the agent provides a ``should_summarize`` method (i.e. it uses
    :class:`~agents.context_summarizer.ContextSummarizerMixin`), the engine
    will trigger summarization when the token threshold is exceeded.

    The summarization check fires **after the model responds** (before
    ``env.step``).  The summarization prompt is appended to the accumulated
    token sequence like an environment message, and the summary completion
    is trained on (mask=1).
    """

    # ------------------------------------------------------------------
    # The full step loop
    # ------------------------------------------------------------------

    async def run_agent_trajectory_async(
        self, idx, application_id, seed=0, mode="Text", **kwargs
    ):
        """Run a single agent trajectory with optional mid-loop summarization."""
        agent = self.agents[idx]
        env = self.envs[idx]

        termination_reason = None
        prompt_token_len = 0
        prompt_tokens = []
        response_token_len = 0
        response_tokens = []
        response_masks = []
        total_time = 0.0
        reward_time = None
        llm_time = 0.0
        env_time = 0.0
        reward = 0.0

        episode_steps: list[dict] = []
        summarization_boundaries: list[int] = []
        segment_chat_histories: list[list[dict]] = []

        # verl-style token accumulation
        accumulated_prompt_ids: list[int] | None = None

        # Detect whether agent supports summarization.
        agent_can_summarize = hasattr(agent, "should_summarize")

        # Cache generation prompt tokens (used for incremental token
        # construction across summarization boundaries).
        gen_prompt_tokens = list(
            self.tokenizer.encode(
                self.chat_parser.generation_prompt, add_special_tokens=False
            )
        )

        # Pre-tokenize the two possible summarization instructions once per
        # trajectory, so the post-env budget check can include the instruction
        # length and `_do_summarization` doesn't have to re-tokenize each fire.
        def _tokenize_instruction(text: str) -> list[int]:
            toks, _ = convert_messages_to_tokens_and_masks(
                [{"role": "user", "content": text}],
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=False,
                contains_generation_msg=True,
            )
            return list(toks)

        cached_instruction_tokens = {
            "token": _tokenize_instruction(TOKEN_SUMMARY_PROMPT),
            "episodic": _tokenize_instruction(EPISODIC_SUMMARY_PROMPT),
            "reflective": _tokenize_instruction(REFLECTIVE_SUMMARY_PROMPT),
            "reflection": _tokenize_instruction(REFLECTION_PROMPT),
        }

        # Reset environment
        loop = asyncio.get_event_loop()
        observation, info = await loop.run_in_executor(self.executor, env.reset)
        info["max_steps"] = self.max_steps

        # Reset agent
        agent.reset()
        agent.update_from_env(
            observation=observation, reward=0.0, done=False, info=info
        )
        # Index in agent._messages of the current episode's first observation
        # message. Updated whenever a new episode is started (see the
        # env.start_new_episode() block below). Consumed by the
        # episodic_carryover="obs_action[_reflection]" path to slice off
        # everything before the just-finished episode.
        episode_start_msg_idx = len(getattr(agent, "_messages", [])) - 1
        messages = agent.chat_completions
        prompt_tokens, _ = convert_messages_to_tokens_and_masks(
            messages,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=True,
            contains_generation_msg=True,
        )
        prompt_token_len = len(prompt_tokens)
        if prompt_token_len > self.max_prompt_length:
            agent.reset()
            raise Exception(
                f"Trajectory {idx}: initial prompt length {prompt_token_len} "
                f"already exceeded max_prompt_length {self.max_prompt_length}, retrying"
            )

        # Cache system prompt tokens for post-summarization reconstruction.
        system_msgs = [m for m in messages if m["role"] == "system"]
        if system_msgs:
            initial_system_tokens, _ = convert_messages_to_tokens_and_masks(
                system_msgs,
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=True,
                contains_generation_msg=False,
            )
            initial_system_tokens = list(initial_system_tokens)
        else:
            initial_system_tokens = []

        for step_idx in range(self.max_steps):
            prompt_messages = agent.chat_completions.copy()

            if not self.enforce_max_prompt_length:
                max_tokens = self.max_response_length - response_token_len
                # Guard: vLLM raises ValueError("max_tokens must be at least 1")
                # if max_tokens <= 0. The downstream TRUNCATION check fires
                # only at the END of a step (after `response_token_len` is
                # incremented), so a step that overshoots the budget by
                # exactly the assistant+env tokens of its predecessor can
                # leave `response_token_len > max_response_length` at the
                # top of the next iteration. Terminate cleanly here instead
                # of letting vLLM raise — same semantics as TRUNCATION,
                # without the noisy traceback + retry storm.
                if max_tokens <= 0:
                    termination_reason = "TRUNCATION"
                    cur_step = agent.get_current_state()
                    cur_step.done = True
                    break

                # Second guard: vLLM's async server ignores our
                # `max_tokens` kwarg and recomputes it as
                # `max_model_len - len(prompt_ids)` (see
                # verl/workers/rollout/vllm_rollout/vllm_async_server.py
                # ::vLLMHttpServer.generate). When the accumulated
                # prompt fills the model context — e.g. obs/action
                # carryover grew the history past max_model_len, or
                # the response budget is generous but the prompt is
                # already saturated — vLLM raises
                # ValueError("max_tokens must be at least 1, got -N")
                # before our `max_tokens` ever takes effect. Terminate
                # cleanly here on the same condition vLLM checks.
                if (
                    accumulated_prompt_ids is not None
                    and getattr(self, "max_model_len", None)
                    and len(accumulated_prompt_ids) >= self.max_model_len
                ):
                    termination_reason = "TRUNCATION"
                    cur_step = agent.get_current_state()
                    cur_step.done = True
                    break
            else:
                max_tokens = self.max_response_length
                prompt_str = self.chat_parser.parse(
                    prompt_messages, add_generation_prompt=True, is_first_msg=True
                )
                prompt_len = len(
                    self.tokenizer.encode(prompt_str, add_special_tokens=False)
                )
                if prompt_len > self.max_prompt_length:
                    termination_reason = "PROMPT_TRUNCATION"
                    break

            kwargs["max_tokens"] = max_tokens

            if accumulated_prompt_ids is not None and self.engine_name == "verl":
                kwargs["accumulated_prompt_ids"] = accumulated_prompt_ids

            start_time = time.time()
            model_output = await self.get_model_response(
                prompt_messages, application_id, **kwargs
            )
            response = model_output.text
            delta_time = time.time() - start_time
            llm_time += delta_time
            total_time += delta_time

            if mode == "Token":
                accumulated_prompt_ids = list(model_output.prompt_ids) + list(
                    model_output.completion_ids
                )
            else:
                accumulated_prompt_ids = None

            prompt_response_pair = {
                "prompt": self.chat_parser.parse(
                    prompt_messages, add_generation_prompt=True, is_first_msg=True
                ),
                "response": response,
                "prompt_ids": model_output.prompt_ids,
                "completion_ids": model_output.completion_ids,
                "logprobs": model_output.logprobs,
            }
            episode_steps.append(prompt_response_pair)

            action: Action = agent.update_from_model(response)
            action = action.action

            # Step environment
            start_time = time.time()
            try:
                next_observation, reward, done, info = await asyncio.wait_for(
                    loop.run_in_executor(self.executor, env.step, action),
                    timeout=(self.trajectory_timeout - total_time),
                )
            except asyncio.TimeoutError:
                termination_reason = "ENV_TIMEOUT"
                if step_idx == 0:
                    colorful_print(
                        f"Warning: Trajectory {idx} completed due to: "
                        f"{termination_reason} before able to perform 1 complete "
                        "action. This might cause unexpected behavior. Consider "
                        "increasing trajectory timeout limit.\n",
                        "red",
                    )
                reward = 0
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            delta_time = time.time() - start_time
            env_time += delta_time
            total_time += delta_time
            info["max_steps"] = self.max_steps
            info["cur_tokens"] = response_token_len

            agent.update_from_env(
                observation=next_observation,
                reward=reward,
                done=done,
                info=info,
            )

            cur_step = agent.get_current_state()
            cur_step.reward = reward
            cur_step.done = done
            cur_step.info.update(info)

            # Stamp episode-level metadata onto the action step we just
            # appended (line ~217). Consumed by chunk-advantage methods
            # (see trainers/chunk_advantage.py) to attribute each chunk to
            # an episode-MDP. Action steps don't currently carry this info
            # in episode_steps; the prompt_response_pair was built before
            # env.step, so we patch it here once `info` is available.
            prompt_response_pair["episode_done"] = bool(
                info.get("episode_done", False)
            )
            prompt_response_pair["episode_index"] = int(
                info.get("episode_index", 0)
            )
            prompt_response_pair["raw_reward"] = float(
                info.get("raw_reward", reward)
            )
            prompt_response_pair["shaped_reward"] = float(reward)

            chat_completions_messages = agent.chat_completions
            assistant_message, env_messages = get_recent_assistant_user_messages(
                chat_completions_messages
            )

            # Summarization now fires after update_from_env (below), so the
            # agent history always ends with the step's env observation by
            # the time we tokenize here — assistant_message is always present.
            assert assistant_message is not None or mode != "Token", (
                "Assistant messages is none when accumulating token trajectories."
            )
            assert env_messages is not None or mode != "Token", (
                "Environment messages is none when accumulating token trajectories."
            )
            assistant_msg_tokens, assistant_msg_masks = [], []
            env_msg_tokens, env_msg_masks = [], []
            if assistant_message:
                assistant_msg_tokens, assistant_msg_masks = (
                    convert_messages_to_tokens_and_masks(
                        [assistant_message],
                        tokenizer=self.tokenizer,
                        parser=self.chat_parser,
                        contains_first_msg=False,
                        contains_generation_msg=False,
                    )
                )
            if env_messages:
                env_msg_tokens, env_msg_masks = convert_messages_to_tokens_and_masks(
                    env_messages,
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                    contains_first_msg=False,
                    contains_generation_msg=True,
                )

            response_token_len += len(assistant_msg_tokens) + len(env_msg_tokens)

            # Stamp per-step delta on the just-appended action step (the
            # current loop iteration's prompt_response_pair, line ~217).
            # Consumed by ``trainers/length_penalty.py`` to compute the
            # per-episode response-token total for the overlong reward
            # shaping. Defaults to 0 when not stamped, so absence is safe.
            prompt_response_pair["response_token_delta"] = (
                len(assistant_msg_tokens) + len(env_msg_tokens)
            )

            if (
                not self.enforce_max_prompt_length
                and response_token_len >= self.max_response_length
            ):
                truncation_length = self.max_response_length - response_token_len
                if truncation_length < 0:
                    truncated_response_tokens = (
                        assistant_msg_tokens + env_msg_tokens
                    )[:truncation_length]
                    truncated_response_masks = (
                        assistant_msg_masks + env_msg_masks
                    )[:truncation_length]
                else:
                    truncated_response_tokens = assistant_msg_tokens + env_msg_tokens
                    truncated_response_masks = assistant_msg_masks + env_msg_masks
                response_tokens.extend(truncated_response_tokens)
                response_masks.extend(truncated_response_masks)

                cur_step = agent.get_current_state()
                if response_token_len - len(env_msg_tokens) > self.max_response_length:
                    cur_step.reward = 0.0
                cur_step.done = True
                termination_reason = "TRUNCATION"
                break

            response_tokens.extend(assistant_msg_tokens)
            response_masks.extend(assistant_msg_masks)
            observation = next_observation

            if total_time >= self.trajectory_timeout:
                termination_reason = "TIMEOUT"
                cur_step = agent.get_current_state()
                done = True
                cur_step.done = done
                break

            if done:
                termination_reason = "ENV_DONE"
                break

            response_tokens.extend(env_msg_tokens)
            response_masks.extend(env_msg_masks)

            if (
                mode == "Token"
                and accumulated_prompt_ids is not None
                and env_msg_tokens
            ):
                accumulated_prompt_ids.extend(env_msg_tokens)

            # ---- SUMMARIZATION CHECK (post-env: unified token + episodic) ----
            # The agent has fully observed the env's response to its action by
            # this point. Episodic trigger takes priority; token is the fallback.
            summ_trigger: str | None = None
            if agent_can_summarize:
                if (
                    hasattr(agent, "should_summarize_on_episode_end")
                    and agent.should_summarize_on_episode_end(info)
                ):
                    summ_trigger = "episode_end"
                elif agent.should_summarize(
                    self.tokenizer,
                    self.chat_parser,
                    # The engine already tracks running token counts
                    # incrementally; passing them avoids an O(history)
                    # re-tokenization per step that pegs CPU and starves
                    # vLLM, idling the GPU between training steps.
                    current_token_count=prompt_token_len + response_token_len,
                ):
                    summ_trigger = "token"

            if summ_trigger is not None:
                # Episodic context-carryover mode (see ContextSummarizerMixin).
                # Only consulted at episode boundaries; the token trigger always
                # uses the freeform path. The oracle path, if active, still wins
                # inside _do_summarization regardless of this.
                carryover = (
                    getattr(agent, "episodic_carryover", "freeform")
                    if summ_trigger == "episode_end"
                    else "freeform"
                )
                # Select the instruction variant the next call would actually
                # use, so the budget check counts its tokens too.
                use_reflective = (
                    summ_trigger == "episode_end"
                    and carryover == "freeform"
                    and getattr(env, "enable_reflection", False)
                )
                if carryover == "obs_action":
                    # Deterministic carryover: no LLM call, no generation.
                    instr_kind = None
                    instr_tokens = []
                elif carryover == "obs_action_reflection":
                    instr_kind = "reflection"
                    instr_tokens = cached_instruction_tokens[instr_kind]
                elif use_reflective:
                    instr_kind = "reflective"
                    instr_tokens = cached_instruction_tokens[instr_kind]
                elif summ_trigger == "episode_end":
                    instr_kind = "episodic"
                    instr_tokens = cached_instruction_tokens[instr_kind]
                else:
                    instr_kind = "token"
                    instr_tokens = cached_instruction_tokens[instr_kind]
                if instr_kind is None:
                    # Deterministic carryover can never run out of budget.
                    summ_budget_needed = 0
                else:
                    summ_budget_needed = agent.summary_max_tokens + len(instr_tokens)
                budget_exhausted = (
                    not self.enforce_max_prompt_length
                    and summ_budget_needed > 0
                    and response_token_len + summ_budget_needed
                    >= self.max_response_length
                )
                if budget_exhausted:
                    # Both token and episodic modes terminate the trajectory
                    # here. We can't summarize (no room) AND we can't continue
                    # rolling out without summarizing — at the production
                    # default (max_response_length=31744, thinking-enabled
                    # responses, 256 concurrent rollouts) letting the
                    # trajectory keep going past its budget pegs vLLM's KV
                    # cache and effectively stalls the whole batch. Earlier
                    # behavior was an implicit termination via env-done crash;
                    # this is the clean version of the same.
                    termination_reason = "SUMMARIZATION_BUDGET_EXCEEDED"
                    break
                else:
                    start_time = time.time()
                    summ_result = await self._do_summarization(
                        agent,
                        application_id,
                        episode_steps,
                        summarization_boundaries,
                        segment_chat_histories=segment_chat_histories,
                        generation_kwargs=kwargs,
                        accumulated_prompt_ids=accumulated_prompt_ids,
                        initial_system_tokens=initial_system_tokens,
                        gen_prompt_tokens=gen_prompt_tokens,
                        mode=mode,
                        trigger=summ_trigger,
                        use_reflective_prompt=use_reflective,
                        summ_instruction_tokens=instr_tokens if instr_kind else None,
                        env=env,
                        carryover=carryover,
                        episode_start_msg_idx=episode_start_msg_idx,
                    )
                    delta_time = time.time() - start_time
                    llm_time += delta_time
                    total_time += delta_time

                    if summ_result is None:
                        # Both modes terminate on summarization failure for
                        # the same reason as the budget-exhausted branch:
                        # without a successful summary the agent has no clean
                        # way to compress context, and continuing to roll out
                        # past the budget thrashes vLLM at scale.
                        termination_reason = "SUMMARIZATION_FAILED"
                        break
                    else:
                        accumulated_prompt_ids, _ = summ_result
                        # Each segment is an independent training sample with
                        # its own response budget.
                        response_token_len = 0

                # Episodic env advance: must run whenever the env signaled an
                # episode boundary (`_pending_episode_start=True`), regardless
                # of whether summarization succeeded, was budget-skipped, or
                # failed. Skipping this leaves the inner env in `done` state
                # and crashes the next env.step with "Environment is done.
                # Call reset() before step()."
                if (
                    summ_trigger == "episode_end"
                    and getattr(env, "_pending_episode_start", False)
                ):
                    new_obs, new_info = await loop.run_in_executor(
                        self.executor, env.start_new_episode
                    )
                    new_info["max_steps"] = self.max_steps
                    new_info["cur_tokens"] = response_token_len
                    # Inject the new episode's initial observation into
                    # the agent's message history. We must NOT route
                    # this through agent.update_from_env(reward=0.0):
                    # that would overwrite the just-finished episode's
                    # success reward (already set on the trajectory's
                    # last step above) with 0, silently zeroing out
                    # every episode reward except the very last one.
                    agent._messages.append(
                        {"role": "user", "content": str(new_obs)}
                    )
                    episode_start_msg_idx = len(agent._messages) - 1
                    new_env_msgs = [
                        {"role": "user", "content": str(new_obs)}
                    ]
                    new_env_tokens, new_env_masks = (
                        convert_messages_to_tokens_and_masks(
                            new_env_msgs,
                            tokenizer=self.tokenizer,
                            parser=self.chat_parser,
                            contains_first_msg=False,
                            contains_generation_msg=True,
                        )
                    )
                    response_tokens.extend(new_env_tokens)
                    response_masks.extend(new_env_masks)
                    if (
                        mode == "Token"
                        and accumulated_prompt_ids is not None
                        and new_env_tokens
                    ):
                        accumulated_prompt_ids.extend(new_env_tokens)

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        # ---- Post-loop (identical to parent) ----

        masked_out = False
        if self.overlong_filter:
            if termination_reason in (
                "TRUNCATION",
                "MAX_STEPS",
                "TIMEOUT",
                "SUMMARIZATION_BUDGET_EXCEEDED",
                "SUMMARIZATION_FAILED",
            ):
                response_masks = [0] * len(response_masks)
                masked_out = True

        if hasattr(env, "compute_final_reward") and not masked_out:
            cur_step = agent.get_current_state()
            start_time = time.time()
            reward = await loop.run_in_executor(
                self.executor, env.compute_final_reward
            )
            reward_time = time.time() - start_time
            cur_step.reward = reward

        await loop.run_in_executor(self.executor, env.close)

        if termination_reason:
            color = "green" if reward > 0 else "yellow"
            colorful_print(
                f"Trajectory {idx} completed due to: {termination_reason}. "
                f"Reward is {reward}. \n",
                color,
            )
            if masked_out:
                colorful_print(
                    f"Trajectory {idx} is masked out due to overlong filter.",
                    "red",
                )

        trajectory: Trajectory = agent.trajectory
        truncation_reasons = (
            "TRUNCATION",
            "PROMPT_TRUNCATION",
            "SUMMARIZATION_BUDGET_EXCEEDED",
            "SUMMARIZATION_FAILED",
        )
        if termination_reason in truncation_reasons:
            for step in trajectory.steps:
                step.reward = 0.0
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)
        if termination_reason in truncation_reasons:
            colorful_print(
                f"Trajectory {idx} is truncated ({termination_reason}). "
                f"Trajectory reward is {trajectory.reward}. \n",
                "red",
            )

        # ---- Return by mode ----

        # Snapshot final segment's chat history.
        segment_chat_histories.append(agent.chat_completions)

        if mode == "Text":
            result = trajectory
        elif mode == "Token":
            segments = self.assemble_segments(
                episode_steps,
                summarization_boundaries,
                trajectory_reward=trajectory.reward,
            )

            # Per-step metadata (small, Ray-serializable) consumed by
            # chunk-advantage methods. Mirrors episode_steps order; pairs
            # with summarization_boundaries to recover segment-step ranges
            # via the same split_points logic as `assemble_segments`.
            step_metadata = [
                {
                    "is_summarization": bool(s.get("is_summarization", False)),
                    "trigger": s.get("trigger", None),
                    "episode_done": bool(s.get("episode_done", False)),
                    "episode_index": int(s.get("episode_index", 0)),
                }
                for s in episode_steps
            ]

            # Per-episode rewards (consumed by variant B per-episode chunk
            # discounting). Pulled from MultiEpisodeEnv when available;
            # otherwise fall back to the single trajectory reward as one
            # episode, which keeps single-episode envs correct under both
            # variants A and B.
            ep_rewards: list[float]
            if hasattr(env, "_episode_rewards"):
                ep_rewards = [float(r) for r in env._episode_rewards]
            else:
                ep_rewards = [float(trajectory.reward)]

            # ---- Optional per-episode length penalty (overlong reward
            # shaping, arXiv:2510.11701 Eq. 9). Off by default. Reads
            # config defensively via .get(...), same pattern as the
            # oracle summarizer block elsewhere in this file. The
            # penalty is added on top of ep_rewards BEFORE the result
            # dict is built — so chunk_advantage downstream sees shaped
            # rewards transparently. In token-summarization mode the
            # summarizer auto-manages length, so this is skipped unless
            # the user explicitly disables ``skip_if_token_mode``.
            lp_response_tokens: list[int] = []
            lp_penalties: list[float] = []
            try:
                lp_cfg = self.config.rllm.get("length_penalty", None)
            except Exception:
                lp_cfg = None
            if lp_cfg is not None and bool(lp_cfg.get("enable", False)):
                try:
                    summ_cfg_node = self.config.rllm.agent.get(
                        "summarization", None
                    )
                    summ_mode = (
                        str(summ_cfg_node.get("mode", "episodic"))
                        if summ_cfg_node is not None
                        else "episodic"
                    )
                except Exception:
                    summ_mode = "episodic"
                skip_token = bool(lp_cfg.get("skip_if_token_mode", True))
                if (
                    not (skip_token and summ_mode == "token")
                    and len(ep_rewards) > 0
                ):
                    from trainers.length_penalty import (
                        apply_length_penalty_to_episode_rewards,
                    )
                    # L_max default: data.max_response_length. Rationale:
                    # in episodic mode each episode starts from a fresh
                    # summary-compressed context, so any single episode
                    # has the full response budget available before it
                    # would risk truncation.
                    l_max_default = int(self.config.data.max_response_length)
                    l_max = int(
                        lp_cfg.get("episode_max_tokens", l_max_default)
                    )
                    l_cache_cfg = lp_cfg.get("episode_cache_tokens", None)
                    l_cache = (
                        int(l_cache_cfg)
                        if l_cache_cfg is not None
                        else max(1, l_max // 4)
                    )
                    shaped, lp_response_tokens, lp_penalties = (
                        apply_length_penalty_to_episode_rewards(
                            ep_rewards,
                            episode_steps,
                            l_max=l_max,
                            l_cache=l_cache,
                        )
                    )
                    ep_rewards = shaped

            # Aggregate length-penalty metrics, computed once so the
            # dict literal below stays readable. NOTE: the trainer's
            # metric aggregator filters values < 0 (used to drop the
            # env's -1 "missing episode" placeholders). Penalties are
            # always in [-1, 0], so they'd be silently dropped if logged
            # raw. We emit the MAGNITUDE (always in [0, 1]) so the
            # filter passes everything through.
            _lp_aggregate_metrics: dict = {}
            if lp_penalties:
                _lp_aggregate_metrics = {
                    "length_penalty/mean_penalty_magnitude": (
                        sum(-p for p in lp_penalties) / len(lp_penalties)
                    ),
                    "length_penalty/triggered_fraction": (
                        sum(1 for p in lp_penalties if p < 0.0)
                        / len(lp_penalties)
                    ),
                }

            # First segment as base for backward compatibility.
            result = {
                **segments[0],
                "idx": env.idx,
                "chat_completions": segment_chat_histories,
                "segments": segments if len(segments) > 1 else None,
                "step_metadata": step_metadata,
                "summarization_boundaries": list(summarization_boundaries),
                "episode_rewards": ep_rewards,
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "token_mismatch": 0.0,
                    "summarization_count": len(summarization_boundaries),
                    "segment_count": len(segments),
                    **{
                        f"length_penalty/episode_{i + 1}_response_tokens": float(t)
                        for i, t in enumerate(lp_response_tokens)
                    },
                    # Penalty magnitude (= -penalty), always in [0, 1].
                    # We emit the magnitude instead of the raw negative
                    # penalty so the trainer's metric aggregator (which
                    # filters values < 0) doesn't silently drop it.
                    **{
                        f"length_penalty/episode_{i + 1}_penalty_magnitude":
                            float(-p)
                        for i, p in enumerate(lp_penalties)
                    },
                    **_lp_aggregate_metrics,
                },
            }
        elif mode == "Conversation":
            result = agent.chat_completions
        elif mode == "Step":
            # Filter out summarization steps for stepwise training.
            task_steps = [s for s in episode_steps if not s.get("is_summarization")]
            result = {
                "steps": task_steps,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "mc_returns": [
                    step.mc_return for step in trajectory.steps
                ][: len(task_steps)],
            }
        else:
            raise ValueError(f"Mode {mode} not supported")

        # Extract env metrics (same as parent MultiEpisodeAsyncAgentExecutionEngine).
        if mode == "Token" and isinstance(result, dict) and "metrics" in result:
            if hasattr(env, "get_metrics") and callable(env.get_metrics):
                try:
                    env_metrics = env.get_metrics()
                    if isinstance(env_metrics, dict):
                        for key, value in env_metrics.items():
                            result["metrics"][key.replace("/", "_")] = value
                except Exception:
                    pass

        return result

    # ------------------------------------------------------------------
    # Summarization helpers
    # ------------------------------------------------------------------

    async def _do_summarization(
        self,
        agent,
        application_id: str,
        episode_steps: list[dict],
        summarization_boundaries: list[int],
        segment_chat_histories: list[list[dict]],
        generation_kwargs: dict,
        accumulated_prompt_ids: list[int] | None,
        initial_system_tokens: list[int],
        gen_prompt_tokens: list[int],
        mode: str,
        trigger: str = "token",
        use_reflective_prompt: bool = False,
        summ_instruction_tokens: list[int] | None = None,
        env: object | None = None,
        carryover: str = "freeform",
        episode_start_msg_idx: int = 1,
    ) -> tuple[list[int] | None, int] | None:
        """Generate a summary, apply it, and return updated accumulated_prompt_ids.

        The summarization instruction is appended to ``accumulated_prompt_ids``
        like an environment message (mask=0).  The summary completion is
        recorded as a regular step (mask=1, trained on).

        Args:
            trigger: Which condition triggered this summarization —
                ``"token"`` or ``"episode_end"``. Recorded as metadata.
            use_reflective_prompt: Select the reflective-style instruction
                (``REFLECTIVE_SUMMARY_PROMPT``) instead of the default
                compression prompt. Used for episode-end triggers when the
                env has ``enable_reflection=True``.
            summ_instruction_tokens: Optional pre-tokenized instruction
                tokens (the chat-template rendering of the summarization
                prompt as a user message). When provided, avoids a
                re-tokenization call. Must match ``use_reflective_prompt``.
            carryover: Episodic context-carryover mode (only meaningful when
                ``trigger == "episode_end"`` and no oracle is active).
                ``"freeform"`` — the default LLM-summary path. ``"obs_action"``
                — deterministic: keep the last episode's obs + boxed actions,
                no LLM call, no trainable step. ``"obs_action_reflection"`` —
                same kept transcript plus a trainable ``<reflection>`` block
                generated via ``REFLECTION_PROMPT``.
            episode_start_msg_idx: Index in ``agent._messages`` of the
                just-finished episode's first observation; used to slice the
                carried-over transcript.

        Returns:
            Tuple of (new_accumulated_prompt_ids, response_token_len_delta),
            or ``None`` if summarization failed (e.g. prompt exceeded
            max_model_len). Carryover modes never return ``None``.
        """
        # Episode context passed to apply_summary / apply_episode_carryover so
        # the carryover user message can label which episode the block is
        # from. `_episode_index` is 0-based on MultiEpisodeEnv; at
        # `episode_end` it's the just-finished episode (the env defers the
        # bump until `start_new_episode`). For `token` trigger it's the
        # currently-running episode. Both render 1-based for display.
        env_episode_index = (
            getattr(env, "_episode_index", None) if env is not None else None
        )
        env_num_episodes = (
            getattr(env, "num_episodes", None) if env is not None else None
        )

        # Oracle short-circuit: when an env-specific rule-based summarizer is
        # registered AND the user has opted in via Hydra, skip the LLM call
        # and feed the oracle's text into apply_summary directly. The oracle
        # output is *not* recorded as a trainable step — the model is never
        # asked to imitate it.
        oracle_fn = self._maybe_oracle_summarizer(env, trigger)
        if oracle_fn is not None:
            try:
                oracle_text = oracle_fn(env)
            except Exception as exc:  # never let oracle failures crash a rollout
                logger.warning("oracle summarizer failed: %s", exc)
                oracle_text = ""
            if not oracle_text:
                return None
            # Snapshot pre-summary chat for logging parity with the LLM path.
            segment_chat_histories.append(agent.chat_completions)
            agent.apply_summary(
                oracle_text,
                trigger=trigger,
                episode_index=env_episode_index,
                num_episodes=env_num_episodes,
            )
            # Mark the segment break at the LAST PRE-ORACLE step so
            # `assemble_segments` splits pre- and post-oracle steps into
            # separate segments. Without the split, `assemble_steps` walks
            # through the chat-history compression as if it were continuous
            # tokens and trips its retokenization-mismatch guard at the
            # boundary (zero-masking the whole trajectory). We do NOT add
            # a placeholder oracle step to `episode_steps`, so the oracle
            # text is still never a PPO training target.
            if episode_steps:
                summarization_boundaries.append(len(episode_steps) - 1)
            new_accumulated: list[int] | None = None
            if mode == "Token":
                summary_user_msg = agent.chat_completions[1]
                summary_msg_tokens, _ = convert_messages_to_tokens_and_masks(
                    [summary_user_msg],
                    tokenizer=self.tokenizer,
                    parser=self.chat_parser,
                    contains_first_msg=False,
                    contains_generation_msg=True,
                )
                new_accumulated = initial_system_tokens + list(summary_msg_tokens)
            return new_accumulated, 0

        # ---- Episodic context-carryover (non-freeform) -------------------
        # Keep the just-finished episode's observations + boxed actions in the
        # history (thinking discarded) rather than replacing everything with a
        # summary. ``obs_action_reflection`` additionally generates a trainable
        # ``<reflection>`` block on top. Reachable only at episode boundaries
        # and only when no oracle summarizer is active (oracle wins above).
        if carryover in ("obs_action", "obs_action_reflection"):
            reflection_text: str | None = None
            response_len_delta = 0

            if carryover == "obs_action_reflection":
                reflection_prompt = agent.build_reflection_prompt()
                summ_kwargs = {
                    k: v
                    for k, v in generation_kwargs.items()
                    if k not in ("accumulated_prompt_ids",)
                }
                summ_kwargs["max_tokens"] = agent.summary_max_tokens
                if summ_instruction_tokens is None:
                    summ_instruction_tokens, _ = convert_messages_to_tokens_and_masks(
                        [{"role": "user", "content": REFLECTION_PROMPT}],
                        tokenizer=self.tokenizer,
                        parser=self.chat_parser,
                        contains_first_msg=False,
                        contains_generation_msg=True,
                    )
                summ_instruction_len = len(summ_instruction_tokens)

                ok_to_generate = True
                if accumulated_prompt_ids is not None and mode == "Token":
                    acc_for_reflection = list(accumulated_prompt_ids)
                    acc_for_reflection.extend(summ_instruction_tokens)
                    if getattr(self, "max_model_len", None):
                        headroom = self.max_model_len - len(acc_for_reflection)
                        if headroom <= 0:
                            # No room to reflect — degrade to plain obs_action.
                            ok_to_generate = False
                        elif headroom < agent.summary_max_tokens:
                            summ_kwargs["max_tokens"] = headroom
                    if ok_to_generate:
                        summ_kwargs["accumulated_prompt_ids"] = acc_for_reflection

                if ok_to_generate:
                    try:
                        reflection_output = await self.get_model_response(
                            reflection_prompt, application_id, **summ_kwargs
                        )
                    except Exception:
                        reflection_output = None
                    if (
                        reflection_output is not None
                        and len(reflection_output.completion_ids) > 0
                    ):
                        reflection_text = reflection_output.text
                        summ_completion_len = len(reflection_output.completion_ids)
                        # Trainable step: mask=1 on the reflection completion,
                        # mask=0 on the instruction (same as the freeform
                        # summary step).
                        episode_steps.append(
                            {
                                "prompt": self.chat_parser.parse(
                                    reflection_prompt,
                                    add_generation_prompt=True,
                                    is_first_msg=True,
                                ),
                                "response": reflection_output.text,
                                "prompt_ids": list(reflection_output.prompt_ids),
                                "completion_ids": list(
                                    reflection_output.completion_ids
                                ),
                                "logprobs": getattr(
                                    reflection_output, "logprobs", []
                                ),
                                "is_summarization": True,
                                "trigger": trigger,
                                "response_token_delta": (
                                    summ_instruction_len + summ_completion_len
                                ),
                            }
                        )
                        response_len_delta = (
                            summ_instruction_len + summ_completion_len
                        )

            # Segment break at the last recorded step (the reflection step if
            # one was added, otherwise the last action step), so
            # assemble_segments splits at the history rewrite — same rationale
            # as the oracle path.
            if episode_steps:
                summarization_boundaries.append(len(episode_steps) - 1)

            segment_chat_histories.append(agent.chat_completions)
            agent.apply_episode_carryover(
                episode_start_msg_idx,
                reflection_text=reflection_text,
                trigger=trigger,
                episode_index=env_episode_index,
                num_episodes=env_num_episodes,
            )

            new_accumulated = self._accumulated_from_chat(
                agent, initial_system_tokens, mode
            )
            return new_accumulated, response_len_delta

        summary_prompt = agent.build_summarization_prompt(
            trigger=trigger, use_reflective_prompt=use_reflective_prompt
        )

        # Use a separate kwargs dict so we don't pollute the caller's.
        summ_kwargs = {
            k: v
            for k, v in generation_kwargs.items()
            if k not in ("accumulated_prompt_ids",)
        }
        summ_kwargs["max_tokens"] = agent.summary_max_tokens

        # Tokenize the summarization instruction as a user message (same
        # pattern as env_msg_tokens: user msg + generation prompt). When the
        # caller passed a pre-tokenized variant, reuse it directly.
        if summ_instruction_tokens is None:
            if use_reflective_prompt:
                instruction_text = REFLECTIVE_SUMMARY_PROMPT
            elif trigger == "episode_end":
                instruction_text = EPISODIC_SUMMARY_PROMPT
            else:
                instruction_text = TOKEN_SUMMARY_PROMPT
            summ_instruction_tokens, _ = convert_messages_to_tokens_and_masks(
                [{"role": "user", "content": instruction_text}],
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=False,
                contains_generation_msg=True,
            )
        summ_instruction_len = len(summ_instruction_tokens)

        # Build summarization prompt tokens incrementally.
        # After update_from_model, accumulated = [...history + gen_prompt + act_i].
        # Append summ_instruction (user msg + gen_prompt) like env_msg_tokens.
        if accumulated_prompt_ids is not None and mode == "Token":
            accumulated_prompt_ids = list(accumulated_prompt_ids)
            accumulated_prompt_ids.extend(summ_instruction_tokens)

            # Guard: check if the summarization prompt exceeds max_model_len.
            # If so, there's no room for the model to generate a summary.
            if hasattr(self, "max_model_len") and self.max_model_len:
                headroom = self.max_model_len - len(accumulated_prompt_ids)
                if headroom <= 0:
                    return None
                # Clamp summary_max_tokens to available headroom.
                if headroom < agent.summary_max_tokens:
                    summ_kwargs["max_tokens"] = headroom

            summ_kwargs["accumulated_prompt_ids"] = accumulated_prompt_ids

        try:
            summary_output = await self.get_model_response(
                summary_prompt, application_id, **summ_kwargs
            )
        except Exception:
            return None

        summ_completion_len = len(summary_output.completion_ids)

        if summ_completion_len == 0:
            return None

        # Record summarization step in episode_steps.
        # The summarization step is TRAINABLE: its completion_ids (the summary)
        # get mask=1 via assemble_steps, and the summ_instruction gap gets mask=0.
        episode_steps.append(
            {
                "prompt": self.chat_parser.parse(
                    summary_prompt, add_generation_prompt=True, is_first_msg=True
                ),
                "response": summary_output.text,
                "prompt_ids": list(summary_output.prompt_ids),
                "completion_ids": list(summary_output.completion_ids),
                "logprobs": getattr(summary_output, "logprobs", []),
                "is_summarization": True,
                "trigger": trigger,
                # Per-step delta consumed by trainers/length_penalty.py.
                # The summary contributes its instruction (mask=0) plus
                # its completion (mask=1) to the trajectory's response
                # tokens. Both are added to ``response_token_len`` via
                # the caller's ``response_len_delta`` below.
                "response_token_delta": summ_instruction_len + summ_completion_len,
            }
        )
        summarization_boundaries.append(len(episode_steps) - 1)

        # Snapshot this segment's full conversation before it's replaced.
        segment_chat_histories.append(agent.chat_completions)

        agent.apply_summary(
            summary_output.text,
            trigger=trigger,
            episode_index=env_episode_index,
            num_episodes=env_num_episodes,
        )

        # Build post-summary accumulated tokens incrementally.
        # After apply_summary: chat_completions = [sys, summary_user_msg].
        new_accumulated: list[int] | None = None
        if mode == "Token":
            summary_user_msg = agent.chat_completions[1]
            summary_msg_tokens, _ = convert_messages_to_tokens_and_masks(
                [summary_user_msg],
                tokenizer=self.tokenizer,
                parser=self.chat_parser,
                contains_first_msg=False,
                contains_generation_msg=True,
            )
            new_accumulated = initial_system_tokens + list(summary_msg_tokens)

        # Total response tokens added by summarization:
        # summ_instruction (mask=0) + summ_completion (mask=1)
        response_len_delta = summ_instruction_len + summ_completion_len

        return new_accumulated, response_len_delta

    def _accumulated_from_chat(
        self, agent, initial_system_tokens: list[int], mode: str
    ) -> list[int] | None:
        """Rebuild ``accumulated_prompt_ids`` from the agent's chat history.

        Used after an episodic-carryover history rewrite, which (unlike the
        freeform summary, ``[sys, summary]``) leaves an arbitrary number of
        non-system messages. Mirrors how the engine builds accumulated tokens
        incrementally elsewhere: ``initial_system_tokens`` followed by the
        chat-template rendering of the remaining messages plus a generation
        prompt. Returns ``None`` for non-Token modes.
        """
        if mode != "Token":
            return None
        non_system = [m for m in agent.chat_completions if m["role"] != "system"]
        if not non_system:
            return list(initial_system_tokens)
        body_tokens, _ = convert_messages_to_tokens_and_masks(
            non_system,
            tokenizer=self.tokenizer,
            parser=self.chat_parser,
            contains_first_msg=False,
            contains_generation_msg=True,
        )
        return list(initial_system_tokens) + list(body_tokens)

    def _maybe_oracle_summarizer(self, env, trigger: str):
        """Return an env-specific oracle summarizer, or ``None``.

        Oracle is opted into via ``rllm.agent.summarization.oracle.enable``
        and currently fires only at episode boundaries (the rule-based
        maze summarizer's contract is "after one episode").
        """
        if env is None or trigger != "episode_end":
            return None
        try:
            oracle_cfg = self.config.rllm.agent.summarization.get("oracle", None)
        except Exception:
            return None
        if oracle_cfg is None:
            return None
        try:
            enabled = bool(oracle_cfg.get("enable", False))
        except Exception:
            enabled = bool(getattr(oracle_cfg, "enable", False))
        if not enabled:
            return None
        return get_oracle_summarizer(env)

    # ------------------------------------------------------------------
    # Training data assembly
    # ------------------------------------------------------------------

    def assemble_segments(
        self,
        steps: list[dict],
        summarization_boundaries: list[int],
        trajectory_reward: float,
    ) -> list[dict]:
        """Assemble training data as one or more segments.

        Each segment is assembled via the parent's ``assemble_steps``.
        Summarization boundaries define the split points: the summarization
        step is the LAST step of its segment, and the next segment starts
        at the step after the boundary.

        All segments share the same ``trajectory_reward``.

        Returns:
            List of segment dicts, each with ``prompt_tokens``,
            ``response_tokens``, ``response_masks``, ``trajectory_reward``.
        """
        if not summarization_boundaries:
            # No summarization — single segment.
            prompt, resp, mask, is_valid = self.assemble_steps(steps)
            return [
                {
                    "prompt_tokens": prompt,
                    "response_tokens": resp,
                    "response_masks": mask,
                    "trajectory_reward": trajectory_reward,
                    "token_mismatch": 0.0 if is_valid else 1.0,
                }
            ]

        # Split into segments: [0..b0], [b0+1..b1], [b1+1..end]
        split_points = [0] + [b + 1 for b in summarization_boundaries] + [len(steps)]

        segments: list[dict] = []
        for i in range(len(split_points) - 1):
            seg_steps = steps[split_points[i] : split_points[i + 1]]
            if not seg_steps:
                continue

            prompt, resp, mask, is_valid = self.assemble_steps(seg_steps)
            segments.append(
                {
                    "prompt_tokens": prompt,
                    "response_tokens": resp,
                    "response_masks": mask,
                    "trajectory_reward": trajectory_reward,
                    "token_mismatch": 0.0 if is_valid else 1.0,
                }
            )

        if not segments:
            # Edge case: all steps filtered out.
            prompt_tokens = torch.tensor(
                steps[0]["prompt_ids"], dtype=torch.long
            )
            segments = [
                {
                    "prompt_tokens": prompt_tokens,
                    "response_tokens": torch.tensor([], dtype=torch.long),
                    "response_masks": torch.tensor([], dtype=torch.long),
                    "trajectory_reward": trajectory_reward,
                    "token_mismatch": 0.0,
                }
            ]

        return segments
