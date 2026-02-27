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
        """Record env feedback and add observation messages.

        For multi-episode boundary transitions, the environment can provide
        explicit terminal/next-initial observation fields in ``info``. When
        available, we append them as two separate user messages to preserve an
        exact token boundary between attempts.
        """
        info = info or {}
        if bool(info.get("boundary_transition", False)):
            terminal_obs = info.get("boundary_terminal_observation", "")
            next_initial_obs = info.get("boundary_next_initial_observation", "")

            appended_any = False
            if isinstance(terminal_obs, str) and terminal_obs:
                self._messages.append({"role": "user", "content": terminal_obs})
                appended_any = True
            if isinstance(next_initial_obs, str) and next_initial_obs:
                self._messages.append({"role": "user", "content": next_initial_obs})
                appended_any = True

            # Fallback for malformed boundary metadata.
            if not appended_any:
                self._messages.append({"role": "user", "content": str(observation)})
        else:
            self._messages.append({"role": "user", "content": str(observation)})

        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = float(reward)
            last_step.done = bool(done)
            last_step.info.update(info or {})

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

