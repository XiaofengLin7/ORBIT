"""Non-cumulative text agent for rLLM.

Like ``GEMTextAgent`` but only preserves the parsed **action** in the
conversation history, discarding thinking / reasoning tokens between steps.
At step N the model can still generate chain-of-thought, but at step N+1
the history only contains previous actions (``\\boxed{...}``), not the
reasoning that produced them.

Designed for use with ``rllm.stepwise_advantage.enable=True`` so each step
is an independent (prompt, response) pair for training while the full
model response (including thinking) is captured in the training data via
``episode_steps[i]["response"]``.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from rllm.agents.agent import Action, BaseAgent, Step, Trajectory  # type: ignore


BOXED_PATTERN = re.compile(r"\\boxed{([^}]+)}")


def extract_last_boxed(text: str) -> str:
    """Extract the last ``\\boxed{...}`` substring; return raw text if none found."""
    matches = list(BOXED_PATTERN.finditer(text))
    if not matches:
        return text.strip()
    return matches[-1].group(1).strip()


class GEMTextAgentNonCumulative(BaseAgent):
    """A non-cumulative text agent for GEM tasks.

    Identical to ``GEMTextAgent`` except that ``update_from_model`` stores
    **only the parsed action** (``\\boxed{...}``) in the message history
    rather than the full model response.  This keeps the context window
    compact across multi-step trajectories.
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
        """Record env feedback and add observation as user message."""
        self._messages.append({"role": "user", "content": str(observation)})
        if self._trajectory.steps:
            last_step = self._trajectory.steps[-1]
            last_step.reward = float(reward)
            last_step.done = bool(done)
            last_step.info.update(info or {})

    def update_from_model(self, response: str, **kwargs) -> Action:
        """Parse model response, store only the action in history.

        The full model response (including any thinking / CoT) is preserved
        in ``Step.model_response`` for logging and training, but only the
        extracted action is appended to ``_messages`` so that future prompts
        see a compact action-only history.
        """
        parsed_action = extract_last_boxed(response)
        boxed_action = f"\\boxed{{{parsed_action}}}"

        # Store ONLY the action in the conversation history.
        self._messages.append({"role": "assistant", "content": boxed_action})

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
