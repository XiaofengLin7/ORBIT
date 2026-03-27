"""Train GEMTextAgent using LaMer's trial-and-reflection multi-episode strategy.

Key differences from train_multi_episode.py:
- Uses LaMerMultiEpisodeEnv instead of MultiEpisodeEnv
- Early stops the trajectory on success (LaMer behaviour)
- Injects reflection + history into the next episode's prompt
- Applies cross-episode reward discounting (traj_gamma)
- System prompt explains the trial-and-reflect format to the agent

Supports both single-task and multi-task training via the same interface as
train_multi_episode.py (tasks_config_path or single env_id).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from data.prepare_gem_data import (  # noqa: E402
    prepare_gem_data,
    prepare_multi_task_gem_data,
)
from envs.lamer_multi_episode_env import LaMerMultiEpisodeEnv  # noqa: E402
from trainers.train_multi_episode import run_ppo_agent  # noqa: E402


def _lamer_system_prompt() -> str:
    """System prompt that explains LaMer's trial-and-reflect multi-episode format."""
    return (
        "You are solving the same task across multiple trials. "
        "Each trial resets the environment to the same initial state. "
        "If you fail a trial, you will be asked to reflect on your mistakes and "
        "produce a concise improved plan inside <remark> </remark> tags. "
        "Your reflection will be shown at the start of the next trial to guide you. "
        "Use past reflections to try harder and smarter on each new trial. "
        "Think step-by-step and respond with actions inside \\boxed{} each turn."
    )


@hydra.main(
    config_path="pkg://rllm.trainer.config",
    config_name="agent_ppo_trainer",
    version_base=None,
)
def main(cfg) -> None:  # type: ignore
    """Entry point for LaMer-style multi-episode GEM training.

    Reads the same config keys as train_multi_episode.py so existing shell
    scripts only need to change the Python entry-point and optionally pass
    LaMer-specific env_args (traj_gamma, reflection_type, max_history_steps).
    """
    env_args = OmegaConf.to_container(cfg.rllm.env.env_args, resolve=True)  # type: ignore

    tasks_config_path: Optional[str] = cfg.data.get("tasks_config_path", None)

    if tasks_config_path:
        tasks_config_path = str(Path(tasks_config_path).expanduser().resolve())
        train_dataset, val_dataset = prepare_multi_task_gem_data(
            tasks_config_path=tasks_config_path
        )

        import yaml

        with open(tasks_config_path) as f:
            config = yaml.safe_load(f)
        all_tasks = config.get("train_tasks", []) + config.get("val_tasks", [])
        max_total_step_cap = max(
            (task.get("total_step_cap", 30) for task in all_tasks), default=30
        )
        if not hasattr(cfg.rllm.agent, "max_steps") or cfg.rllm.agent.max_steps < max_total_step_cap:
            cfg.rllm.agent.max_steps = max_total_step_cap
    else:
        inner_env_kwargs = env_args.get("inner_env_kwargs", {})
        env_id = inner_env_kwargs.get("env_id", "game:GuessTheNumber-v0-hard")
        train_dataset, val_dataset = prepare_gem_data(env_id=env_id)

    if train_dataset is not None and hasattr(cfg, "data"):
        cfg.data.train_files = train_dataset.get_verl_data_path()
    if val_dataset is not None and hasattr(cfg, "data"):
        cfg.data.val_files = val_dataset.get_verl_data_path()

    agent_args = OmegaConf.to_container(
        cfg.rllm.agent.get("agent_args", {}), resolve=True
    )  # type: ignore
    agent_args = dict(agent_args or {})
    agent_args.setdefault("system_prompt", _lamer_system_prompt())

    run_ppo_agent(
        cfg,
        env_class=LaMerMultiEpisodeEnv,
        agent_class=GEMTextAgent,
        agent_args=agent_args,
    )


if __name__ == "__main__":
    main()
