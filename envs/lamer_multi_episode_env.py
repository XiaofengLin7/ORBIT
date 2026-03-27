"""LaMer-style multi-episode environment wrapper.

Implements the training strategy from LaMer (https://arxiv.org/abs/2512.16848):

1. Early stopping: trajectory ends immediately when any episode succeeds.
2. Reflection: after each *failed* episode, the agent is prompted to produce a
   <remark>...</remark> reflection. The reflection is stored and injected into
   the prompt at the start of the next episode.
3. History accumulation: at the start of episode N > 1 the observation is
   prefixed with formatted summaries of past failed episodes and their
   reflections (configurable via ``reflection_type``).
4. Cross-episode reward discounting: the success reward for episode k is scaled
   by ``traj_gamma ** k``, discouraging the model from relying on late episodes.

Relationship to MultiEpisodeEnv
--------------------------------
This class subclasses :class:`MultiEpisodeEnv` and overrides ``reset`` and
``step``.  The parent's reflection machinery (``enable_reflection`` flag) is
disabled so we can manage the reflection state ourselves.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple, Type

from rllm.environments.base.base_env import BaseEnv  # type: ignore

from envs.multi_episode_env import MultiEpisodeEnv


# ---------------------------------------------------------------------------
# Prompt templates (mirrors LaMer's webshop/alfworld prompt templates)
# ---------------------------------------------------------------------------

_REFLECT_PROMPT = """\
You will be given the history of a past experience in this environment.
Your job is to **reflect on the past sequence**, identify any **mistakes or
inefficiencies**, and then devise a **concise, improved plan** starting from
the original initial state.

Below are the actions you took and the corresponding observations:
{episode_text}
The task is NOT successfully completed.

Now reflect on the past experience and come up with a new plan of action.
- First, reason step-by-step about the strategy you took and where things went
  wrong or could be improved.
- Then devise a concise new plan that accounts for your mistakes.
- Finally, write your reflection and improved plan inside <remark> </remark>
  tags so it can guide the next trial.
"""

_PAST_HISTORY_AND_REFLECTION_TEMPLATE = """\

On trial #{idx}, the actions you took and the corresponding observations are:
{episode_text}
The task is NOT successfully completed. Your reflection is:
{reflection}"""

_PAST_HISTORY_ONLY_TEMPLATE = """\

On trial #{idx}, the actions you took and the corresponding observations are:
{episode_text}
The task is NOT successfully completed."""

_PAST_REFLECTION_ONLY_TEMPLATE = """\

On trial #{idx}, the task is NOT successfully completed. Your reflection is:
{reflection}"""

_EPISODE_START_WITH_HISTORY = """\
{history}

Currently you're on trial #{trial}, starting from the initial state.

{init_obs}"""

_EPISODE_START_NO_HISTORY = """\
Currently you're on trial #{trial}, starting from the initial state.

{init_obs}"""


class LaMerMultiEpisodeEnv(MultiEpisodeEnv):
    """Multi-episode env implementing LaMer's trial-and-reflection strategy.

    Args:
        inner_env_class: Inner environment class or dotted path string.
        inner_env_kwargs: Kwargs forwarded to the inner environment.
        total_step_cap: Maximum total steps across all episodes.  Acts as a
            hard budget; training also stops early on success.
        success_reward: Base reward for completing an episode successfully.
            The actual reward is ``success_reward * traj_gamma ** episode_idx``.
        traj_gamma: Cross-episode discount factor (LaMer default 0.6).
        reflection_type: How past episodes appear in the next episode's prompt.
            One of ``"reflection_only"`` (default), ``"history_only"``, or
            ``"history_and_reflection"``.
        max_history_steps: Maximum number of (action, obs) pairs to include
            when formatting a past episode's history.
        episode_header: Short text prepended to episode start observations.
    """

    def __init__(
        self,
        inner_env_class: Type[BaseEnv] | str,
        inner_env_kwargs: Optional[Dict[str, Any]] = None,
        total_step_cap: int = 30,
        success_reward: float = 1.0,
        traj_gamma: float = 0.6,
        reflection_type: str = "reflection_only",
        max_history_steps: int = 15,
        episode_header: str = "New episode begins.",
        **kwargs: Any,
    ) -> None:
        # Disable parent's reflection so we manage it ourselves.
        super().__init__(
            inner_env_class=inner_env_class,
            inner_env_kwargs=inner_env_kwargs,
            total_step_cap=total_step_cap,
            success_reward=success_reward,
            episode_header=episode_header,
            enable_reflection=False,
            **kwargs,
        )
        assert reflection_type in (
            "reflection_only",
            "history_only",
            "history_and_reflection",
        ), f"Unknown reflection_type: {reflection_type}"

        self.traj_gamma = float(traj_gamma)
        self.reflection_type = reflection_type
        self.max_history_steps = int(max_history_steps)

        # Per-trajectory state (re-initialised in reset())
        self._current_episode_steps: List[Tuple[str, str]] = []  # (action, obs)
        self._past_episode_texts: List[str] = []
        self._past_reflections: List[str] = []

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self, seed: Optional[int] = None, task: Optional[dict] = None
    ) -> Tuple[Any, dict]:
        obs, info = super().reset(seed=seed, task=task)
        self._current_episode_steps = []
        self._past_episode_texts = []
        self._past_reflections = []
        return obs, info

    def step(self, action: Any) -> Tuple[Any, float, bool, dict]:  # type: ignore[override]
        """Execute one step, implementing LaMer's trial-reflection strategy."""
        # ------------------------------------------------------------------
        # Reflection step: the agent just produced its <remark> reflection.
        # ------------------------------------------------------------------
        if self._waiting_for_reflection:
            self._waiting_for_reflection = False

            reflection, reflection_valid = self._extract_reflection(action)
            self._past_reflections.append(reflection)

            # LaMer: reward 1.0 if <remark> tag was parseable, 0.0 otherwise.
            reflect_reward = 1.0 if reflection_valid else 0.0

            # Reset inner env for the next trial.
            next_obs, reset_info = self._reset_inner_env()
            self._maybe_track_maze_state()
            self._episode_index += 1
            self._episode_step = 0
            self._current_episode_steps = []

            # Build LaMer-style observation that injects history + reflection.
            formatted_obs = self._build_next_episode_obs(next_obs)

            augmented_info = self._augment_info(
                base_info=reset_info,
                episode_index=self._episode_index,
                episode_step=self._episode_step,
                total_step=self._total_steps,
                episode_start=True,
                episode_done=False,
            )
            return formatted_obs, reflect_reward, False, augmented_info

        # ------------------------------------------------------------------
        # Normal action step.
        # ------------------------------------------------------------------
        from rllm.agents.agent import Action  # type: ignore

        raw_action = action.action if isinstance(action, Action) else action

        observation, env_reward, inner_done, info = self.inner_env.step(raw_action)
        self._maybe_track_maze_state()

        self._total_steps += 1
        self._episode_step += 1
        self._maybe_record_interval_unique_states()

        # Track (action, observation) for history.
        self._current_episode_steps.append((str(raw_action), str(observation)))

        success = self._is_episode_success(
            done=inner_done, info=info, reward=env_reward
        )

        # LaMer: apply cross-episode discount to the success reward.
        shaped_reward = (
            self.success_reward * (self.traj_gamma ** self._episode_index)
            if success
            else 0.0
        )

        # LaMer: early stop the whole trajectory on success.
        outer_done = success or (self._total_steps >= self.total_step_cap)

        if inner_done:
            self._episode_successes.append(success)
            self._episode_lengths.append(self._episode_step)
            self._maybe_record_episode_unique_states()

            if not outer_done:
                # Episode failed → store its text and prompt for reflection.
                ep_text = self._format_episode_text()
                self._past_episode_texts.append(ep_text)

                self._waiting_for_reflection = True
                observation = self._build_reflect_prompt(observation, ep_text)

        # Budget exhausted mid-episode.
        if outer_done and not inner_done and self._episode_step > 0:
            self._episode_successes.append(False)
            self._episode_lengths.append(self._episode_step)
            self._maybe_record_episode_unique_states()

        if outer_done:
            self._maybe_record_interval_unique_states(force_record=True)

        augmented_info = self._augment_info(
            base_info=info,
            episode_index=self._episode_index,
            episode_step=self._episode_step,
            total_step=self._total_steps,
            episode_start=(self._episode_step == 0),
            episode_done=inner_done,
            success=success if inner_done else None,
            raw_reward=env_reward,
            trajectory_done=outer_done,
        )

        return observation, shaped_reward, outer_done, augmented_info

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _format_episode_text(self) -> str:
        """Format the last ``max_history_steps`` steps as plain text."""
        steps = self._current_episode_steps[-self.max_history_steps :]
        lines: List[str] = []
        for action, obs in steps:
            lines.append(f"> {action}")
            lines.append(obs)
        return "\n".join(lines)

    def _build_reflect_prompt(self, terminal_obs: str, episode_text: str) -> str:
        """Append LaMer-style reflection prompt to the terminal observation."""
        reflect_section = _REFLECT_PROMPT.format(episode_text=episode_text)
        return f"{terminal_obs}\n\n{reflect_section}"

    def _build_next_episode_obs(self, init_obs: str) -> str:
        """Build the start-of-episode observation with injected history."""
        trial = self._episode_index + 1  # 1-based for display
        n_past = len(self._past_reflections)

        if n_past == 0:
            return _EPISODE_START_NO_HISTORY.format(trial=trial, init_obs=init_obs)

        history_parts: List[str] = []
        for idx in range(n_past):
            if self.reflection_type == "history_and_reflection":
                part = _PAST_HISTORY_AND_REFLECTION_TEMPLATE.format(
                    idx=idx + 1,
                    episode_text=self._past_episode_texts[idx],
                    reflection=self._past_reflections[idx],
                )
            elif self.reflection_type == "history_only":
                part = _PAST_HISTORY_ONLY_TEMPLATE.format(
                    idx=idx + 1,
                    episode_text=self._past_episode_texts[idx],
                )
            else:  # reflection_only (default)
                part = _PAST_REFLECTION_ONLY_TEMPLATE.format(
                    idx=idx + 1,
                    reflection=self._past_reflections[idx],
                )
            history_parts.append(part)

        history = "\n".join(history_parts)
        return _EPISODE_START_WITH_HISTORY.format(
            history=history, trial=trial, init_obs=init_obs
        )

    @staticmethod
    def _extract_reflection(action: Any) -> Tuple[str, bool]:
        """Extract text inside <remark>...</remark>.

        Returns:
            (reflection_text, valid) where valid=True if <remark> tags were found.
            Falls back to the full response text with valid=False.
        """
        text = action.action if hasattr(action, "action") else str(action)
        match = re.search(r"<remark>(.*?)</remark>", text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), True
        return text.strip(), False
