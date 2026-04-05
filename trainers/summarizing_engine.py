"""Execution engine with mid-trajectory context summarization.

Overrides the monolithic ``run_agent_trajectory_async`` loop from rLLM's
``AgentExecutionEngine`` to insert a summarization check after every
``agent.update_from_env()`` call.  When the agent's context exceeds a
configured token threshold, the engine generates a summary via the same
model and replaces the agent's message history with the compressed form.

For **Token** (cumulative) mode the ``assemble_steps`` override assembles
only the pre-summary steps — the full trajectory reward (including any
post-summary success) is still used.

For **Step** (stepwise-advantage) mode, summarization is transparent:
each step is an independent (prompt, response) pair, and post-summary
steps simply have a shorter prompt containing the summary.
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

from trainers.multi_episode_trainer import MultiEpisodeAsyncAgentExecutionEngine

logger = logging.getLogger(__name__)


class SummarizingAgentExecutionEngine(MultiEpisodeAsyncAgentExecutionEngine):
    """Execution engine that compresses context mid-trajectory.

    When the agent provides a ``should_summarize`` method (i.e. it uses
    :class:`~agents.context_summarizer.ContextSummarizerMixin`), the engine
    will trigger summarization when the token threshold is exceeded.
    """

    # ------------------------------------------------------------------
    # The full step loop — copied from AgentExecutionEngine with the
    # summarization hook inserted after ``agent.update_from_env()``.
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

        # verl-style token accumulation
        accumulated_prompt_ids: list[int] | None = None

        # Detect whether agent supports summarization.
        agent_can_summarize = hasattr(agent, "should_summarize")

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

            # ---- SUMMARIZATION CHECK (new) ----
            if agent_can_summarize and agent.should_summarize(
                self.tokenizer, self.chat_parser
            ):
                await self._do_summarization(
                    agent,
                    application_id,
                    episode_steps,
                    summarization_boundaries,
                    kwargs,
                )
                # Invalidate accumulated token sequence — force re-tokenize.
                accumulated_prompt_ids = None

            if step_idx == self.max_steps - 1:
                termination_reason = "MAX_STEPS"

        # ---- Post-loop (identical to parent) ----

        masked_out = False
        if self.overlong_filter:
            if termination_reason in ("TRUNCATION", "MAX_STEPS", "TIMEOUT"):
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
        if termination_reason in ("TRUNCATION", "PROMPT_TRUNCATION"):
            for step in trajectory.steps:
                step.reward = 0.0
        compute_trajectory_reward(trajectory)
        compute_mc_return(trajectory, gamma=self.gamma)
        if termination_reason in ("TRUNCATION", "PROMPT_TRUNCATION"):
            colorful_print(
                f"Trajectory {idx} is truncated. Trajectory reward is "
                f"{trajectory.reward}. \n",
                "red",
            )

        # ---- Return by mode ----

        if mode == "Text":
            result = trajectory
        elif mode == "Token":
            prompt_tokens, response_tokens, response_masks, is_valid = (
                self.assemble_steps_with_summarization(
                    episode_steps, summarization_boundaries
                )
            )
            result = {
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "response_masks": response_masks,
                "trajectory_reward": trajectory.reward,
                "idx": env.idx,
                "chat_completions": agent.chat_completions,
                "metrics": {
                    "steps": len(trajectory.steps),
                    "reward_time": reward_time,
                    "env_time": env_time,
                    "llm_time": llm_time,
                    "total_time": total_time,
                    "token_mismatch": 0.0 if is_valid else 1.0,
                    "summarization_count": len(summarization_boundaries),
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
        generation_kwargs: dict,
    ) -> None:
        """Generate a summary and apply it to the agent."""
        summary_prompt = agent.build_summarization_prompt()

        # Use a separate kwargs dict so we don't pollute the caller's.
        summ_kwargs = {
            k: v
            for k, v in generation_kwargs.items()
            if k not in ("accumulated_prompt_ids",)
        }
        summ_kwargs["max_tokens"] = agent.summary_max_tokens

        summary_output = await self.get_model_response(
            summary_prompt, application_id, **summ_kwargs
        )

        # Record summarization step for training data assembly.
        summ_prompt_str = self.chat_parser.parse(
            summary_prompt, add_generation_prompt=True, is_first_msg=True
        )
        summ_prompt_ids = list(
            self.tokenizer.encode(summ_prompt_str, add_special_tokens=False)
        )
        summ_completion_ids = list(
            self.tokenizer.encode(summary_output.text, add_special_tokens=False)
        )

        episode_steps.append(
            {
                "prompt": summ_prompt_str,
                "response": summary_output.text,
                "prompt_ids": summ_prompt_ids,
                "completion_ids": summ_completion_ids,
                "logprobs": getattr(summary_output, "logprobs", []),
                "is_summarization": True,
            }
        )
        summarization_boundaries.append(len(episode_steps) - 1)

        agent.apply_summary(summary_output.text)

        colorful_print(
            f"Summarization #{len(summarization_boundaries)} applied "
            f"(step {len(episode_steps) - 1}). "
            f"Messages: {agent.get_summarization_metadata()}\n",
            "cyan",
        )

    # ------------------------------------------------------------------
    # Training data assembly
    # ------------------------------------------------------------------

    def assemble_steps_with_summarization(
        self,
        steps: list[dict],
        summarization_boundaries: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        """Assemble training data, stopping at the first summarization boundary.

        In cumulative (Token) mode, the post-summary segment has a different
        prompt that breaks the accumulated-token invariant. We therefore
        assemble only the pre-summary task steps. The full trajectory reward
        (including any post-summary success) is still attributed to these
        tokens.

        If no summarization occurred, falls through to the parent's
        ``assemble_steps``.
        """
        if not summarization_boundaries:
            return self.assemble_steps(steps)

        # Assemble only pre-summary task steps (exclude the summarization step itself).
        first_boundary = summarization_boundaries[0]
        pre_summary_steps = [
            s for s in steps[:first_boundary] if not s.get("is_summarization")
        ]

        if not pre_summary_steps:
            # Edge case: summarization fired before any task step completed.
            prompt_tokens = torch.tensor(
                steps[0]["prompt_ids"], dtype=torch.long
            )
            return (
                prompt_tokens,
                torch.tensor([], dtype=torch.long),
                torch.tensor([], dtype=torch.long),
                True,
            )

        return self.assemble_steps(pre_summary_steps)
