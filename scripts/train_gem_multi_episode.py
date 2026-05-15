"""Train GEMTextAgent in a multi-episode workflow while keeping vendor code intact."""

from __future__ import annotations

from pathlib import Path
import sys

import hydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from data.prepare_gem_data import prepare_gem_data  # noqa: E402
from envs.gem_env_adapter import GEMEnvAdapter  # noqa: E402
from rllm.data import DatasetRegistry  # type: ignore  # noqa: E402
from rllm.trainer.agent_trainer import AgentTrainer  # type: ignore  # noqa: E402
from workflows.multi_episode_workflow import MultiEpisodeWorkflow  # noqa: E402


def _inject_workflow_flag(cfg) -> None:
    """Ensure workflow mode is enabled without mutating vendor defaults."""
    if hasattr(cfg, "rllm") and hasattr(cfg.rllm, "workflow"):
        cfg.rllm.workflow.use_workflow = True


def _default_multi_episode_prompt() -> str:
    """System prompt that reminds the policy about multi-episode control.

    Delegates to :func:`prompts.system_prompts.build_multi_episode_system_prompt`
    so all training/eval entry points share one prompt source. This entry
    point does not configure mid-trajectory summarization, so no memory-
    protocol section is appended.
    """
    from prompts.system_prompts import build_multi_episode_system_prompt

    return build_multi_episode_system_prompt()


@hydra.main(config_path="pkg://rllm.trainer.config", config_name="agent_ppo_trainer", version_base=None)
def main(cfg):  # type: ignore
    """Entry point for multi-episode GEM training."""
    _inject_workflow_flag(cfg)

    env_id = cfg.rllm.env.env_args.get("env_id", "game:GuessTheNumber-v0-hard")
    prepare_gem_data(env_id=env_id)
    train_dataset = DatasetRegistry.load_dataset("gem_tasks", "train")
    val_dataset = DatasetRegistry.load_dataset("gem_tasks", "test")

    env_args = OmegaConf.to_container(cfg.rllm.env.env_args, resolve=True)  # type: ignore
    agent_args = OmegaConf.to_container(cfg.rllm.agent.get("agent_args", {}), resolve=True)  # type: ignore

    # Provide a multi-episode-aware system prompt by default.
    agent_args = dict(agent_args or {})
    agent_args.setdefault("system_prompt", _default_multi_episode_prompt())

    step_ref = {"step": None}

    # Monkeypatch AgentWorkflowEngine.set_training_step to capture global_step without editing vendor files.
    try:
        from rllm.engine.agent_workflow_engine import AgentWorkflowEngine  # type: ignore

        _orig_set_training_step = AgentWorkflowEngine.set_training_step

        def _patched_set_training_step(self, step: int, mode: str = "train", epoch: int = 0):
            step_ref["step"] = step
            return _orig_set_training_step(self, step=step, mode=mode, epoch=epoch)

        AgentWorkflowEngine.set_training_step = _patched_set_training_step  # type: ignore
    except Exception:
        pass
    workflow_args = {
        "agent_cls": GEMTextAgent,
        "env_cls": GEMEnvAdapter,
        "agent_args": agent_args,
        "env_args": env_args,
        # Global step cap defaults to 3x per-episode max_turns when unspecified.
        "total_step_cap": env_args.get("total_step_cap"),
        "min_episodes": 3,
        "episode_header": "New episode starts; reuse prior knowledge.",
        "training_step_getter": lambda: step_ref.get("step"),
        "step_ref": step_ref,
        "default_local_dir": cfg.trainer.default_local_dir,
    }

    trainer = AgentTrainer(
        workflow_class=MultiEpisodeWorkflow,
        workflow_args=workflow_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=cfg,
        backend="verl",
    )
    trainer.train()


if __name__ == "__main__":
    main()

