"""Multi-episode environment wrapper for rLLM.

This wrapper encapsulates multi-episode logic, allowing an agent to interact
with the same task across multiple episodes within a single trajectory. The
inner environment is reset when an episode ends, but the outer trajectory
continues until the total step cap is reached.
"""

from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Set, Tuple, Type

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

from prompts.prompt import reflection_prompt_2 as reflection_prompt

class MultiEpisodeEnv(BaseEnv):
    """Wrapper environment that runs multiple episodes of an inner environment.

    This environment wraps another BaseEnv and implements multi-episode logic:
    - When the inner environment returns done=True, it resets internally
    - The outer trajectory only ends when total_steps reaches total_step_cap
    - Rewards are shaped: success_reward on episode success, 0.0 otherwise

    Attributes:
        inner_env: The wrapped environment instance.
        total_step_cap: Maximum steps across all episodes.
        success_reward: Reward given when an episode succeeds.
        episode_header: Text prepended to observations on episode start.
    """

    def __init__(
        self,
        inner_env_class: Type[BaseEnv] | str,
        inner_env_kwargs: Optional[Dict[str, Any]] = None,
        total_step_cap: int = 30,
        success_reward: float = 1.0,
        episode_header: str = "New episode begins.",
        enable_reflection: bool = True,
        num_episodes: Optional[int] = None,
        reflection_via_summarization: bool = False,
        **kwargs: Any,
    ):
        """Initialize the multi-episode environment wrapper.

        Args:
            inner_env_class: The inner environment class or its fully qualified
                string name (e.g., "envs.gem_env_adapter.GEMEnvAdapter").
            inner_env_kwargs: Keyword arguments to pass to the inner environment
                constructor.
            total_step_cap: Maximum number of steps across all episodes.
            success_reward: Reward assigned when an episode succeeds.
            episode_header: Text prepended to observations at episode start.
            enable_reflection: Whether to prompt for reflection after each episode.
            reflection_prompt: The prompt to use for reflection.
            num_episodes: Optional hard cap on the number of completed episodes.
                When set, the trajectory terminates as soon as this many episodes
                have completed, even if total_step_cap has not been reached.
                total_step_cap continues to act as a safety ceiling.
            reflection_via_summarization: When True, the env's own reflection
                flow is suppressed (no reflection prompt appended to the
                terminal observation, no combining with the next episode's
                initial observation). At episode boundaries ``step()`` returns
                only the terminal observation and sets
                ``self._pending_episode_start=True``; the caller (execution
                engine) is responsible for calling :meth:`start_new_episode`
                to advance to the next episode. Typically auto-set by the
                trainer when episodic summarization is enabled.
            **kwargs: Additional arguments (ignored, for compatibility).
        """
        super().__init__()

        # Resolve inner environment class from string if needed
        if isinstance(inner_env_class, str):
            inner_env_class = self._resolve_class(inner_env_class)

        self.inner_env_class = inner_env_class
        self.inner_env_kwargs = dict(inner_env_kwargs or {})
        self.total_step_cap = max(1, int(total_step_cap))
        self.success_reward = float(success_reward)
        self.episode_header = episode_header
        self.enable_reflection = enable_reflection
        self.reflection_prompt = reflection_prompt
        self.num_episodes = int(num_episodes) if num_episodes is not None else None
        self.reflection_via_summarization = bool(reflection_via_summarization)

        # Create the inner environment
        self.inner_env: BaseEnv = self.inner_env_class(**self.inner_env_kwargs)

        # Episode tracking state (initialized in reset)
        self._total_steps: int = 0
        self._episode_index: int = 0
        self._episode_step: int = 0
        self._episode_successes: List[bool] = []
        self._episode_lengths: List[int] = []
        self._episode_rewards: List[float] = []
        # Maze-only: track unique visited positions across the full trajectory.
        # This is intentionally scoped to Maze to keep this wrapper simple.
        self._maze_visited_positions: Optional[Set[Tuple[int, int]]] = None
        # Maze-only: chronological per-episode path of visited positions.
        # Each inner list is one episode's positions in order, including the
        # episode's start. Consumed by the oracle summarizer.
        self._maze_episode_paths: Optional[List[List[Tuple[int, int]]]] = None
        self._task: Optional[dict] = None
        self._seed: Optional[int] = None
        # Store the initial task from dataset (extracted in from_dict)
        self._initial_task: Optional[dict] = None
        # Task type (e.g. ALFWorld task category), captured from inner env info
        self._task_type: Optional[str] = None

        # Reflection state
        self._waiting_for_reflection: bool = False
        # Episodic-summarization boundary state: when
        # reflection_via_summarization=True and an episode just ended,
        # start_new_episode() is expected to be called before the next step().
        self._pending_episode_start: bool = False

    @staticmethod
    def _resolve_class(class_path: str) -> Type[BaseEnv]:
        """Resolve a class from its fully qualified string path.

        Args:
            class_path: Fully qualified class name (e.g., "envs.gem_env_adapter.GEMEnvAdapter").

        Returns:
            The resolved class.

        Raises:
            ImportError: If the module or class cannot be found.
        """
        module_path, class_name = class_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)

    def reset(
        self, seed: Optional[int] = None, task: Optional[dict] = None
    ) -> tuple[Any, dict]:
        """Reset the multi-episode environment.

        Initializes tracking state and resets the inner environment.

        Args:
            seed: Optional random seed for reproducibility.
            task: Optional task dictionary to pass to inner environment.
                If None, uses the initial task from dataset (set in from_dict).

        Returns:
            observation: Initial observation with episode header.
            info: Initial info dictionary with multi-episode metadata.
        """
        self._total_steps = 0
        self._episode_index = 0
        self._episode_step = 0
        self._episode_successes = []
        self._episode_lengths = []
        self._episode_rewards = []
        is_maze = self._is_maze_inner_env()
        self._maze_visited_positions = set() if is_maze else None
        # Initial episode path bucket — initial position lands here via the
        # _maybe_track_maze_state() call after _reset_inner_env() below.
        self._maze_episode_paths = [[]] if is_maze else None
        # Use provided task, or fall back to initial task from dataset
        self._task = task if task is not None else self._initial_task
        self._seed = seed if seed is not None else self._seed

        self._waiting_for_reflection = False
        self._pending_episode_start = False

        # Reset inner environment
        observation, info = self._reset_inner_env()
        self._maybe_track_maze_state()
        # Capture task_type from inner env if provided (e.g. ALFWorld task category)
        if isinstance(info, dict) and "task_type" in info:
            self._task_type = info["task_type"]

        # Format observation with episode header
        observation = self._format_observation(observation, self._episode_index)

        # Augment info with multi-episode metadata
        augmented_info = self._augment_info(
            base_info=info,
            episode_index=self._episode_index,
            episode_step=self._episode_step,
            total_step=self._total_steps,
            episode_start=True,
            episode_done=False,
        )

        return observation, augmented_info

    def step(self, action: Any) -> tuple[Any, float, bool, dict]:
        """Execute one step in the multi-episode environment.

        If the inner environment returns done=True, it is reset and the
        trajectory continues. The outer trajectory only ends when total_steps
        reaches total_step_cap.

        Args:
            action: Action to execute in the inner environment.

        Returns:
            observation: Next observation (with episode header on new episode).
            reward: Shaped reward (success_reward on episode success, else 0.0).
            done: True only when total_steps >= total_step_cap.
            info: Augmented info dictionary with multi-episode metadata.
        """
        # Handle reflection step if waiting for reflection
        if self._waiting_for_reflection:
            self._waiting_for_reflection = False

            # Start the next episode
            observation, info = self._reset_inner_env()
            self._maybe_track_maze_state()
            self._episode_index += 1
            self._episode_step = 0
            observation = self._format_observation(observation, self._episode_index)

            augmented_info = self._augment_info(
                base_info=info,
                episode_index=self._episode_index,
                episode_step=self._episode_step,
                total_step=self._total_steps,
                episode_start=True,
                episode_done=False,
            )
            return observation, 0.0, False, augmented_info
        
        # Unwrap Action if needed
        raw_action = action.action if isinstance(action, Action) else action

        # Execute step in inner environment
        observation, env_reward, inner_done, info = self.inner_env.step(raw_action)
        self._maybe_track_maze_state()

        self._total_steps += 1
        self._episode_step += 1

        # Determine if this episode succeeded
        success = self._is_episode_success(
            done=inner_done, info=info, reward=env_reward
        )

        # Shape reward: success_reward on episode success, 0.0 otherwise
        shaped_reward = self.success_reward if success else 0.0

        # Determine if outer trajectory should end
        outer_done = self._total_steps >= self.total_step_cap

        # Handle episode completion
        if inner_done:
            # Record the just-finished episode.
            # Use raw_reward from info when available (e.g. WebShop stores the
            # continuous 0-1 score there while env_reward is binarized for
            # success determination).
            self._episode_successes.append(success)
            self._episode_lengths.append(self._episode_step)
            self._episode_rewards.append(
                float(info.get("raw_reward", env_reward))
            )

            # num_episodes cap: stop once we've completed the requested count.
            # Checked after recording so the just-finished episode counts toward
            # the cap and the reset-for-next-episode branch below is skipped.
            if (
                self.num_episodes is not None
                and len(self._episode_successes) >= self.num_episodes
            ):
                outer_done = True

            if not outer_done:
                if self.reflection_via_summarization:
                    # Summarization (in the engine) handles the episode boundary.
                    # Return only the terminal observation; the engine will call
                    # start_new_episode() after summarizing to advance the env.
                    self._pending_episode_start = True
                elif self.enable_reflection:
                    self._waiting_for_reflection = True

                    # Append reflection prompt to terminal observation
                    if isinstance(observation, str):
                        observation = f"{observation}\n\n{self.reflection_prompt}"

                    # We return the terminal observation + prompt, with episode_done=True
                    # The inner env is NOT reset yet.
                else:
                    # We want the agent to see the terminal observation of the
                    # finished episode *and* get primed for the next episode.
                    #
                    # To avoid losing the terminal observation, we first cache it,
                    # then reset the inner env and append the next-episode header +
                    # initial observation after the terminal one.
                    terminal_observation = observation

                    # Reset for next episode (same task/seed) and bump index.
                    next_obs, reset_info = self._reset_inner_env()
                    self._open_new_maze_episode_bucket()
                    self._maybe_track_maze_state()
                    self._episode_index += 1
                    self._episode_step = 0
                    next_obs = self._format_observation(next_obs, self._episode_index)

                    # Combine terminal obs and the next-episode header/obs when using
                    # string observations (the GEM adapter always returns strings).
                    if isinstance(terminal_observation, str) and isinstance(next_obs, str):
                        observation = f"{terminal_observation}\n\n{next_obs}"
                    else:
                        # Fallback: just expose the terminal observation.
                        observation = terminal_observation

                    # Merge reset info into info so downstream can see both.
                    info = {**info, **reset_info}
        
        # If trajectory ends and current episode is incomplete, count it as an attempted episode
        # (but not successful since it didn't complete)
        if outer_done and not inner_done and self._episode_step > 0:
            # Current episode didn't complete, count it as attempted but not successful
            self._episode_successes.append(False)
            self._episode_lengths.append(self._episode_step)
            self._episode_rewards.append(0.0)

        # Augment info with multi-episode metadata
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

    def start_new_episode(self) -> Tuple[Any, dict]:
        """Advance to the next episode without consuming an action.

        Called by the execution engine after episodic summarization fires
        (``reflection_via_summarization=True`` mode). Resets the inner
        environment, bumps ``episode_index``, resets ``episode_step``, and
        returns the new episode's initial observation plus augmented info
        with ``episode_start=True``.
        """
        if not self._pending_episode_start:
            raise RuntimeError(
                "start_new_episode() called without a pending episode boundary. "
                "This should only be called after step() returned "
                "episode_done=True under reflection_via_summarization=True."
            )
        next_obs, reset_info = self._reset_inner_env()
        self._open_new_maze_episode_bucket()
        self._maybe_track_maze_state()
        self._episode_index += 1
        self._episode_step = 0
        next_obs = self._format_observation(next_obs, self._episode_index)
        self._pending_episode_start = False

        augmented_info = self._augment_info(
            base_info=reset_info,
            episode_index=self._episode_index,
            episode_step=0,
            total_step=self._total_steps,
            episode_start=True,
            episode_done=False,
        )
        return next_obs, augmented_info

    def close(self) -> None:
        """Close the inner environment.
        
        Also counts the current incomplete episode if the trajectory ended early
        (e.g., due to TRUNCATION, TIMEOUT, or MAX_STEPS in the execution engine).
        """
        # If there's an incomplete episode (episode_step > 0), count it as attempted
        # This handles cases where the execution engine terminates early before
        # the episode completes or before outer_done=True is set
        if self._episode_step > 0 and len(self._episode_successes) == self._episode_index:
            # Current episode didn't complete, count it as attempted but not successful
            self._episode_successes.append(False)
            self._episode_lengths.append(self._get_max_turns())
        
        if hasattr(self.inner_env, "close"):
            self.inner_env.close()

    # Note: We do NOT implement compute_final_reward() because:
    # 1. Rewards are already given per episode in step() (success_reward on episode success)
    # 2. If we implemented it, it would overwrite the last step's reward, potentially
    #    zeroing out the reward from the final episode if it completed successfully
    # 3. The trajectory reward is correctly computed as sum of step rewards, which
    #    equals the number of successful episodes * success_reward

    def _get_max_turns(self) -> int:
        """Get max_turns from inner environment configuration.

        Returns:
            max_turns value from inner_env_kwargs or inner_env, or total_step_cap as fallback.
        """
        # Try to get from inner_env_kwargs first
        env_kwargs = self.inner_env_kwargs.get("env_kwargs", {})
        max_turns = env_kwargs.get("max_turns")
        if max_turns is not None:
            return int(max_turns)
        
        # Try to get from inner_env attribute
        if hasattr(self.inner_env, "max_turns"):
            return int(self.inner_env.max_turns)
        
        # Try to get from inner_env._env (for GEMEnvAdapter)
        if hasattr(self.inner_env, "_env") and hasattr(self.inner_env._env, "max_turns"):
            return int(self.inner_env._env.max_turns)
        
        # Fallback to total_step_cap (conservative estimate)
        return self.total_step_cap

    def get_metrics(self) -> Dict[str, Any]:
        """Collect metrics matching MultiEpisodeWorkflow format.

        Returns a dictionary with the same metric keys as MultiEpisodeWorkflow's
        collect_metrics method for consistent logging across both approaches.

        Returns metrics for minimum_episodes episodes (calculated as
        total_step_cap // max_turns). Missing episodes are handled differently
        based on termination type:
        - Early termination (total_steps < total_step_cap): Missing episodes
          get success_rate=0 and steps=max_turns
        - Normal completion: Missing episodes get success_rate=-1.0 and steps=-1
          (these -1 values are filtered out during aggregation as the trainer
          filters values < 0)

        Returns:
            Dictionary containing:
                - episode/success_rate: 1.0 if any episode succeeded, else 0.0
                - episode/num_episodes: Total number of episodes attempted
                - episode/success_count: Number of successful episodes
                - episode/total_steps: Total steps across all episodes (uses
                  total_step_cap for early termination, _total_steps otherwise)
                - episode/truncated: 1.0 if early termination, 0.0 otherwise
                - episode_N/success_rate: Success rate for episode N
                  (minimum_episodes episodes)
                - episode_N/steps: Steps taken in episode N
                  (minimum_episodes episodes)
        """
        max_turns = self._get_max_turns()
        if self.num_episodes is not None:
            minimum_episodes = self.num_episodes
        else:
            minimum_episodes = self.total_step_cap // max_turns
        # A trajectory that ended because we hit num_episodes is a *normal*
        # completion, not a truncation — distinguish from running out of steps.
        hit_episode_cap = (
            self.num_episodes is not None
            and len(self._episode_successes) >= self.num_episodes
        )
        # Check if trajectory ended early (before reaching step cap)
        if self._total_steps < self.total_step_cap and not hit_episode_cap:
            # Early termination: treat all episodes as failures with max_turns each
            
            metrics: Dict[str, Any] = {
                "episode/success_rate": 1.0 if sum(self._episode_successes) > 0 else 0.0,
                "episode/num_episodes": len(self._episode_successes),
                "episode/success_count": sum(self._episode_successes),
                "episode/total_steps": self.total_step_cap,
                "episode/truncated": 1.0,
            }
            if self._has_continuous_rewards() and self._episode_rewards:
                metrics["episode/avg_reward"] = sum(self._episode_rewards) / len(self._episode_rewards)
                metrics["episode/max_reward"] = max(self._episode_rewards)
            if self._maze_visited_positions is not None:
                metrics["maze/unique_visited_states"] = float(len(self._maze_visited_positions))

            for idx in range(1, minimum_episodes + 1):
                if idx <= len(self._episode_successes):
                    metrics[f"episode_{idx}/success_rate"] = 1.0 if self._episode_successes[idx - 1] else 0.0
                else:
                    metrics[f"episode_{idx}/success_rate"] = 0
                if idx <= len(self._episode_lengths):
                    metrics[f"episode_{idx}/steps"] = self._episode_lengths[idx - 1]
                else:
                    metrics[f"episode_{idx}/steps"] = max_turns
                if self._has_continuous_rewards():
                    if idx <= len(self._episode_rewards):
                        metrics[f"episode_{idx}/reward"] = self._episode_rewards[idx - 1]
                    else:
                        metrics[f"episode_{idx}/reward"] = 0.0

            return metrics
        
        # Normal completion: use recorded episode data
        successes = self._episode_successes
        any_success = 1.0 if any(successes) else 0.0

        metrics: Dict[str, Any] = {
            "episode/success_rate": any_success,
            "episode/num_episodes": len(successes),
            "episode/success_count": sum(successes),
            "episode/total_steps": self._total_steps,
            "episode/truncated": 0.0,
        }
        if self._has_continuous_rewards() and self._episode_rewards:
            metrics["episode/avg_reward"] = sum(self._episode_rewards) / len(self._episode_rewards)
            metrics["episode/max_reward"] = max(self._episode_rewards)
        if self._maze_visited_positions is not None:
            metrics["maze/unique_visited_states"] = float(len(self._maze_visited_positions))

        for idx in range(1, minimum_episodes + 1):
            if idx <= len(successes):
                metrics[f"episode_{idx}/success_rate"] = 1.0 if successes[idx - 1] else 0.0
            else:
                metrics[f"episode_{idx}/success_rate"] = -1.0  # Will be filtered

        for idx in range(1, minimum_episodes + 1):
            if idx <= len(self._episode_lengths):
                metrics[f"episode_{idx}/steps"] = self._episode_lengths[idx - 1]
            else:
                metrics[f"episode_{idx}/steps"] = -1  # Will be filtered

        if self._has_continuous_rewards():
            for idx in range(1, minimum_episodes + 1):
                if idx <= len(self._episode_rewards):
                    metrics[f"episode_{idx}/reward"] = self._episode_rewards[idx - 1]
                else:
                    metrics[f"episode_{idx}/reward"] = -1.0  # Will be filtered

        return metrics

    def get_trajectory_info(self) -> Dict[str, Any]:
        """Get trajectory-level info matching MultiEpisodeWorkflow format.

        Returns a dictionary with the same keys as trajectory.info.update()
        in MultiEpisodeWorkflow for consistent metadata across both approaches.

        Returns:
            Dictionary containing:
                - multi_episode: Always True
                - episode_successes: List of booleans for each episode
                - success_count: Number of successful episodes
                - num_episodes: Total episodes attempted
                - total_steps: Total steps across all episodes
                - step_cap: The configured step cap
                - episode_lengths: List of step counts per episode
        """
        info: Dict[str, Any] = {
            "multi_episode": True,
            "episode_successes": list(self._episode_successes),
            "episode_rewards": list(self._episode_rewards),
            "success_count": sum(self._episode_successes),
            "num_episodes": len(self._episode_successes),
            "total_steps": self._total_steps,
            "step_cap": self.total_step_cap,
            "episode_lengths": list(self._episode_lengths),
        }
        if self._maze_visited_positions is not None:
            info["unique_visited_states"] = len(self._maze_visited_positions)
        return info

    def _has_continuous_rewards(self) -> bool:
        """Return True iff the inner env uses continuous (non-binary) rewards.

        Currently only WebShop has continuous 0-1 partial match scores.
        Other envs use binary rewards so avg_reward would just duplicate
        success_rate — we skip emitting reward metrics for them.
        """
        try:
            from envs.webshop_env_adapter import WebShopEnvAdapter
        except Exception:
            return False
        return isinstance(self.inner_env, WebShopEnvAdapter)

    def _is_maze_inner_env(self) -> bool:
        """Return True iff the wrapped env is `MazeEnvAdapter`."""
        try:
            from envs.maze_env_adapter import MazeEnvAdapter
        except Exception:
            return False
        return isinstance(self.inner_env, MazeEnvAdapter)

    def _maybe_track_maze_state(self) -> None:
        """Track maze position if and only if inner env is Maze."""
        if self._maze_visited_positions is None:
            return
        try:
            state_id = self.inner_env.get_state_id()
        except Exception:
            # Tracking must never break the main env loop.
            return
        self._maze_visited_positions.add(state_id)
        if self._maze_episode_paths is not None:
            # Always append; an empty bucket was opened at trajectory start
            # or when a new episode began.
            self._maze_episode_paths[-1].append(state_id)

    def _open_new_maze_episode_bucket(self) -> None:
        """Open a fresh per-episode path bucket for the upcoming episode.

        Called immediately before tracking the new episode's start position,
        so the start is the first entry of the new bucket rather than the
        last entry of the previous one.
        """
        if self._maze_episode_paths is None:
            return
        self._maze_episode_paths.append([])

    @property
    def task_type(self) -> Optional[str]:
        """Task type captured from inner env (e.g. ALFWorld task category)."""
        return self._task_type

    @property
    def is_correct(self) -> bool:
        """Whether the trajectory is considered correct.

        A trajectory is correct if at least one episode succeeded,
        matching the logic in MultiEpisodeWorkflow.assign_episode_correctness.

        Returns:
            True if any episode succeeded, False otherwise.
        """
        return sum(self._episode_successes) > 0

    def _reset_inner_env(self) -> tuple[Any, dict]:
        """Reset the inner environment with the stored seed and task.

        Always passes the task to ensure the same task is reused across episodes
        within a trajectory. Extracts seed from task dict if available.

        Returns:
            observation: Initial observation from inner environment.
            info: Info dictionary from inner environment.
        """
        # Extract seed from task dict if available, otherwise use stored seed
        reset_seed = None
        if self._task is not None and isinstance(self._task, dict):
            reset_seed = self._task.get("seed")
        if reset_seed is None:
            reset_seed = self._seed

        # Always try to pass task first to preserve the same task across episodes
        if self._task is not None:
            try:
                # Try with both seed (from task) and task dict
                return self.inner_env.reset(seed=reset_seed, task=self._task)
            except TypeError:
                try:
                    # Try with just task dict
                    return self.inner_env.reset(task=self._task)
                except TypeError:
                    # Fallback: try with seed extracted from task
                    if reset_seed is not None:
                        try:
                            return self.inner_env.reset(seed=reset_seed)
                        except TypeError:
                            pass
                    # Last resort: no args
                    return self.inner_env.reset()
        else:
            # No task available, use seed if provided
            if reset_seed is not None:
                try:
                    return self.inner_env.reset(seed=reset_seed)
                except TypeError:
                    return self.inner_env.reset()
            else:
                return self.inner_env.reset()

    def _format_observation(self, observation: Any, episode_index: int) -> Any:
        """Prepend episode marker to observation.

        Args:
            observation: Raw observation from inner environment.
            episode_index: Current episode index (0-based).

        Returns:
            Observation with episode header prepended.
        """
        header = f"[Episode {episode_index + 1}] {self.episode_header}".strip()
        if isinstance(observation, str):
            return f"{header}\n{observation}"
        return observation

    @staticmethod
    def _is_episode_success(done: bool, info: Dict[str, Any], reward: float) -> bool:
        """Determine whether the just-finished episode succeeded.

        An episode is considered successful if it terminated (not truncated)
        with a positive reward.

        Args:
            done: Whether the episode ended.
            info: Info dictionary from the environment.
            reward: Reward from the final step.

        Returns:
            True if the episode succeeded, False otherwise.
        """
        if not done:
            return False
        terminated = bool(info.get("terminated", False))
        truncated = bool(info.get("truncated", False))
        if truncated:
            return False
        return terminated and reward > 0

    def _augment_info(
        self,
        base_info: Optional[dict],
        episode_index: int,
        episode_step: int,
        total_step: int,
        episode_start: bool,
        episode_done: bool,
        success: Optional[bool] = None,
        raw_reward: Optional[float] = None,
        trajectory_done: bool = False,
    ) -> dict:
        """Inject multi-episode metadata into the info dict.

        Args:
            base_info: Original info from inner environment.
            episode_index: Current episode index (0-based).
            episode_step: Step within current episode.
            total_step: Total steps across all episodes.
            episode_start: Whether this is the start of an episode.
            episode_done: Whether the current episode just ended.
            success: Whether the episode succeeded (if episode_done).
            raw_reward: Raw reward from inner environment.
            trajectory_done: Whether the outer trajectory is complete.

        Returns:
            Augmented info dictionary.
        """
        info = dict(base_info or {})
        # When the task config didn't set an explicit num_episodes (i.e. the
        # trajectory is bounded only by total_step_cap), advertise a sensible
        # estimated count to consumers — derived as the worst-case full-length
        # episode count that fits inside total_step_cap. This is informational
        # only; it does not become a hard trajectory cap.
        displayed_num_episodes = self.num_episodes
        if displayed_num_episodes is None:
            try:
                inner_max_turns = self._get_max_turns()
                if inner_max_turns and inner_max_turns > 0:
                    displayed_num_episodes = max(1, self.total_step_cap // int(inner_max_turns))
            except Exception:
                pass

        info.update(
            {
                "episode_index": episode_index,
                "episode_step": episode_step,
                "total_step": total_step,
                "episode_start": episode_start,
                "episode_done": episode_done,
                "multi_episode": True,
                "step_cap": self.total_step_cap,
                "num_episodes": displayed_num_episodes,
                "episode_successes": list(self._episode_successes),
                "episode_lengths": list(self._episode_lengths),
            }
        )
        if success is not None:
            info["episode_success"] = success
        if raw_reward is not None:
            info["raw_reward"] = raw_reward

        # Include full metrics and trajectory info when trajectory completes
        if trajectory_done:
            info["metrics"] = self.get_metrics()
            info["trajectory_info"] = self.get_trajectory_info()
            info["is_correct"] = self.is_correct

        return info

    @staticmethod
    def from_dict(info: dict) -> "MultiEpisodeEnv":
        """Factory method to create MultiEpisodeEnv from a dictionary.

        Expected keys:
            - inner_env_class: Class or string path of inner environment.
            - inner_env_kwargs: Dict of kwargs for inner environment.
            - total_step_cap: Maximum steps across all episodes (global default).
            - success_reward: Reward for successful episodes.
            - episode_header: Text to prepend at episode start.
            - task: Optional task dictionary from dataset (e.g., {"env_id": ..., "seed": ..., "uid": ...}).

        Per-task configuration (from dataset task dict):
            - max_turns_per_episode: Overrides inner_env_kwargs.env_kwargs.max_turns if present.
            - total_step_cap: Overrides global total_step_cap if present.

        Args:
            info: Configuration dictionary. May contain per-sample task data from dataset
                (e.g., env_id, seed, uid, max_turns_per_episode, total_step_cap) mixed with config keys.

        Returns:
            A new MultiEpisodeEnv instance.
        """
        import logging

        logger = logging.getLogger(__name__)
        
        inner_env_class = info.get("inner_env_class")
        inner_env_kwargs = info.get("inner_env_kwargs", {})
        # Global default, may be overridden by per-task config
        total_step_cap = info.get("total_step_cap", 30)
        success_reward = info.get("success_reward", 1.0)
        episode_header = info.get("episode_header", "New episode begins.")
        enable_reflection = info.get("enable_reflection", True)
        reflection_prompt = info.get("reflection_prompt", "")
        reflection_via_summarization = info.get("reflection_via_summarization", False)

        if inner_env_class is None:
            raise ValueError("inner_env_class must be provided in env_args")

        # Extract task-related keys from info (these come from dataset extra_info)
        # Common task keys: env_id, seed, uid, max_turns_per_episode, total_step_cap
        # Filter out config keys to get the task dict
        config_keys = {
            "inner_env_class",
            "inner_env_kwargs",
            "total_step_cap",
            "success_reward",
            "episode_header",
            "enable_reflection",
            "reflection_prompt",
            "reflection_via_summarization",
        }
        task_dict = {k: v for k, v in info.items() if k not in config_keys and v is not None}
        
        # Extract per-task configuration from task dict
        # These override global/default values
        per_task_env_id = task_dict.get("env_id")
        per_task_max_turns = task_dict.get("max_turns_per_episode")
        per_task_total_step_cap = task_dict.get("total_step_cap")
        per_task_num_episodes = task_dict.get("num_episodes")

        # Override total_step_cap if provided in task dict
        if per_task_total_step_cap is not None:
            total_step_cap = int(per_task_total_step_cap)
            logger.debug(f"Using per-task total_step_cap: {total_step_cap}")
        elif per_task_num_episodes is not None:
            # Auto-derive a safety ceiling when the user only specified
            # num_episodes. Falls back to the inner env's max_turns if
            # max_turns_per_episode isn't provided per-task.
            max_turns_fallback = per_task_max_turns
            if max_turns_fallback is None:
                max_turns_fallback = (
                    inner_env_kwargs.get("env_kwargs", {}).get("max_turns")
                )
            if max_turns_fallback is None:
                max_turns_fallback = 30
            total_step_cap = int(per_task_num_episodes) * int(max_turns_fallback)
            logger.debug(
                f"Auto-derived total_step_cap from num_episodes: {total_step_cap} "
                f"(num_episodes={per_task_num_episodes} * max_turns={max_turns_fallback})"
            )
        
        # Override inner_env_kwargs with per-task configuration
        # inner_env_kwargs may have nested structure: env_kwargs.max_turns
        inner_env_kwargs = dict(inner_env_kwargs)  # Make a copy to avoid mutating original
        
        # Extract env_id from task dict and set it in inner_env_kwargs
        # GEMEnvAdapter expects env_id as a direct argument
        if per_task_env_id is not None:
            inner_env_kwargs["env_id"] = per_task_env_id
            logger.debug(f"Using per-task env_id: {per_task_env_id}")
        
        if per_task_max_turns is not None:
            # Ensure env_kwargs exists
            if "env_kwargs" not in inner_env_kwargs:
                inner_env_kwargs["env_kwargs"] = {}
            inner_env_kwargs["env_kwargs"] = dict(inner_env_kwargs["env_kwargs"])
            inner_env_kwargs["env_kwargs"]["max_turns"] = int(per_task_max_turns)
            # add key value pairs to inner_env_kwargs from task_dict except max_turns_per_episode and total_step_cap
            for k, v in task_dict.items():
                if k not in ("env_id", "max_turns_per_episode", "total_step_cap", "num_episodes", "data_source", 'seed', 'uid'):
                    inner_env_kwargs["env_kwargs"][k] = v
            logger.debug(f"Using per-task max_turns_per_episode: {per_task_max_turns}")

        # If we have task-like keys (env_id, seed, uid), use them as the task
        # Remove per-task config keys from task dict before storing (they're already applied)
        task_dict_clean = {
            k: v
            for k, v in task_dict.items()
            if k not in ("max_turns_per_episode", "total_step_cap", "num_episodes")
        }
        initial_task = task_dict_clean if task_dict_clean else None
        
        # Debug logging to verify task extraction and config application
        logger.debug(
            f"MultiEpisodeEnv.from_dict: info keys={sorted(info.keys())}, "
            f"task_dict={task_dict}, initial_task={initial_task}, "
            f"total_step_cap={total_step_cap}, max_turns={inner_env_kwargs.get('env_kwargs', {}).get('max_turns', 'default')}"
        )
        
        env = MultiEpisodeEnv(
            inner_env_class=inner_env_class,
            inner_env_kwargs=inner_env_kwargs,
            total_step_cap=total_step_cap,
            success_reward=success_reward,
            episode_header=episode_header,
            enable_reflection=enable_reflection,
            reflection_prompt=reflection_prompt,
            num_episodes=per_task_num_episodes,
            reflection_via_summarization=reflection_via_summarization,
        )
        # Store the initial task so it can be reused across episodes
        env._initial_task = initial_task
        env._task = initial_task  # Set initial task for first reset

        return env

    @staticmethod
    def is_multithread_safe() -> bool:
        """Indicate whether this environment is safe for multi-threaded use.

        Returns:
            True, as the wrapper itself is thread-safe. The inner environment's
            thread safety should be verified separately.
        """
        return True

