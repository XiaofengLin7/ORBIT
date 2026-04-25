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

from prompts.summarization_prompts import (
    CONTEXT_SUMMARY_PROMPT,
    REFLECTIVE_SUMMARY_PROMPT,
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
            "context": _tokenize_instruction(CONTEXT_SUMMARY_PROMPT),
            "reflective": _tokenize_instruction(REFLECTIVE_SUMMARY_PROMPT),
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
                elif agent.should_summarize(self.tokenizer, self.chat_parser):
                    summ_trigger = "token"

            if summ_trigger is not None:
                # Select the instruction variant the next call would actually
                # use, so the budget check counts its tokens too.
                use_reflective = (
                    summ_trigger == "episode_end"
                    and getattr(env, "enable_reflection", False)
                )
                instr_tokens = cached_instruction_tokens[
                    "reflective" if use_reflective else "context"
                ]
                summ_budget_needed = agent.summary_max_tokens + len(instr_tokens)
                budget_exhausted = (
                    not self.enforce_max_prompt_length
                    and response_token_len + summ_budget_needed
                    >= self.max_response_length
                )
                if budget_exhausted:
                    if summ_trigger == "token":
                        termination_reason = "SUMMARIZATION_BUDGET_EXCEEDED"
                        colorful_print(
                            f"Trajectory {idx}: summarization skipped — "
                            f"response budget exhausted "
                            f"({response_token_len}+{summ_budget_needed}"
                            f">={self.max_response_length}).\n",
                            "red",
                        )
                        break
                    else:
                        colorful_print(
                            f"Trajectory {idx}: episodic summarization skipped — "
                            f"response budget exhausted "
                            f"({response_token_len}+{summ_budget_needed}"
                            f">={self.max_response_length}).\n",
                            "yellow",
                        )
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
                        summ_instruction_tokens=instr_tokens,
                    )
                    delta_time = time.time() - start_time
                    llm_time += delta_time
                    total_time += delta_time

                    if summ_result is None:
                        if summ_trigger == "token":
                            termination_reason = "SUMMARIZATION_FAILED"
                            colorful_print(
                                f"Trajectory {idx}: summarization failed.\n",
                                "red",
                            )
                            break
                        else:
                            # Episodic path: soft-skip on context-window
                            # violation. Continue trajectory as usual.
                            colorful_print(
                                f"Trajectory {idx}: episodic summarization skipped "
                                f"(context window would be exceeded).\n",
                                "yellow",
                            )
                    else:
                        accumulated_prompt_ids, _ = summ_result
                        # Each segment is an independent training sample with
                        # its own response budget.
                        response_token_len = 0

                        # Episodic: advance env to the new episode and inject
                        # its initial observation into agent history + token
                        # accumulation. This is how we keep the post-summary
                        # layout `[sys, summary, new_episode_obs]` without a
                        # wasted LLM call.
                        if (
                            summ_trigger == "episode_end"
                            and getattr(env, "_pending_episode_start", False)
                        ):
                            new_obs, new_info = await loop.run_in_executor(
                                self.executor, env.start_new_episode
                            )
                            new_info["max_steps"] = self.max_steps
                            new_info["cur_tokens"] = response_token_len
                            agent.update_from_env(
                                observation=new_obs,
                                reward=0.0,
                                done=False,
                                info=new_info,
                            )
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

            # First segment as base for backward compatibility.
            result = {
                **segments[0],
                "idx": env.idx,
                "chat_completions": segment_chat_histories,
                "segments": segments if len(segments) > 1 else None,
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "token_mismatch": 0.0,
                    "summarization_count": len(summarization_boundaries),
                    "segment_count": len(segments),
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

        Returns:
            Tuple of (new_accumulated_prompt_ids, response_token_len_delta),
            or ``None`` if summarization failed (e.g. prompt exceeded
            max_model_len).
        """
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
            instruction_text = (
                REFLECTIVE_SUMMARY_PROMPT if use_reflective_prompt else CONTEXT_SUMMARY_PROMPT
            )
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
                    colorful_print(
                        f"Summarization prompt ({len(accumulated_prompt_ids)} tokens) "
                        f"exceeds max_model_len ({self.max_model_len}). "
                        "Skipping summarization.\n",
                        "red",
                    )
                    return None
                # Clamp summary_max_tokens to available headroom.
                if headroom < agent.summary_max_tokens:
                    colorful_print(
                        f"Summarization headroom limited to {headroom} tokens "
                        f"(requested {agent.summary_max_tokens}).\n",
                        "yellow",
                    )
                    summ_kwargs["max_tokens"] = headroom

            summ_kwargs["accumulated_prompt_ids"] = accumulated_prompt_ids

        try:
            summary_output = await self.get_model_response(
                summary_prompt, application_id, **summ_kwargs
            )
        except Exception as e:
            colorful_print(
                f"Summarization generation failed: {e}\n",
                "red",
            )
            return None

        summ_completion_len = len(summary_output.completion_ids)

        if summ_completion_len == 0:
            colorful_print(
                "Summarization produced empty output. Skipping.\n",
                "red",
            )
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
            }
        )
        summarization_boundaries.append(len(episode_steps) - 1)

        # Snapshot this segment's full conversation before it's replaced.
        segment_chat_histories.append(agent.chat_completions)

        agent.apply_summary(summary_output.text, trigger=trigger)

        colorful_print(
            f"Summarization #{len(summarization_boundaries)} applied "
            f"(trigger={trigger}, step {len(episode_steps) - 1}, "
            f"summary_tokens={summ_completion_len}). "
            f"Messages: {agent.get_summarization_metadata()}\n",
            "cyan",
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
