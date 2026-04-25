"""Text-only GEM agent for rLLM.

This agent maintains a minimal chat history, formats user observations, and
parses the last ``\\boxed{...}`` integer from the model response as the action
to send to GEM environments such as GuessTheNumber.
"""

from __future__ import annotations

import copy
import re
from typing import Any, List

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore


BOXED_PATTERN = re.compile(r"\\boxed{([^}]+)}")


def extract_last_boxed(text: str) -> str:
    """Extract the last ``\\boxed{...}`` substring; return raw text if none found."""
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return text.strip()
    return matches[-1].group(1).strip()


class GEMTextAgent(BaseAgent):
    """A lightweight text-only agent for GEM tasks (no tool use).

    The agent:
    - Keeps a running chat history with an optional system prompt.
    - Appends environment observations as user messages.
    - Parses the model response for the last ``\\boxed{...}`` snippet and
      returns it as the action.
    """

    def __init__(self, system_prompt: str | None = None, max_steps: int = 20):
        self.system_prompt = system_prompt or "Solve the task. Return your final answer inside \\boxed{}."
        self.max_steps = max_steps
        self._messages: list[dict[str, str]] = []
        self._trajectory = Trajectory()
        self.reset()

    @property
    def chat_completions(self) -> list[dict[str, str]]:
        return copy.deepcopy(self._messages)

    @property
    def trajectory(self) -> Trajectory:
        return self._trajectory

    def reset(self):
        """Clear history and trajectory; re-add system prompt."""
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self._trajectory = Trajectory()

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs):
        """Record env feedback and add observation as user message.

        As a convenience, substitute simple task-level placeholders in the
        system message when the env first reports them. Currently supported:
        ``{num_episodes}`` — filled from ``info["num_episodes"]`` once, so a
        single template prompt can announce the per-task horizon when tasks
        with different horizons are evaluated together.
        """
        self._maybe_fill_system_placeholders(info)
        self._messages.append({"role": "user", "content": str(observation)})
        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = float(reward)
            last_step.done = bool(done)
            last_step.info.update(info or {})

    def _maybe_fill_system_placeholders(self, info: dict | None) -> None:
        """One-shot ``{num_episodes}`` substitution in the system message.

        If ``info["num_episodes"]`` is set, the placeholder is replaced with
        the integer count. If it's missing or None (e.g. single-task
        training without a hard cap), the placeholder is replaced with
        ``"multiple"`` so the prompt never leaks the literal ``{…}`` to the
        model.
        """
        if not self._messages:
            return
        sys_msg = self._messages[0]
        if sys_msg.get("role") != "system":
            return
        content = sys_msg.get("content", "")
        if "{num_episodes}" not in content:
            return
        num_ep = (info or {}).get("num_episodes")
        replacement = str(num_ep) if num_ep is not None else "multiple"
        sys_msg["content"] = content.replace("{num_episodes}", replacement)

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Parse model response, append to history, and create a new Step.

        The GEM environments expect boxed guesses (``\\boxed{42}``), so we
        re-wrap the extracted value before forwarding it to the env.
        """
        parsed_action = extract_last_boxed(response)
        boxed_action = f"\\boxed{{{parsed_action}}}"
        self._messages.append({"role": "assistant", "content": response})

        step = Step(
            chat_completions=copy.deepcopy(self._messages),
            observation=self._messages[-2]["content"] if len(self._messages) >= 2 else None,
            action=Action(action=boxed_action),
            model_response=response,
            info={},
        )
        self._trajectory.steps.append(step)
        return Action(action=boxed_action)

    def get_current_state(self) -> Step:
        if not self._trajectory.steps:
            return Step(chat_completions=copy.deepcopy(self._messages))
        return self._trajectory.steps[-1]


