#!/usr/bin/env python3
"""Evaluate OpenAI models on multi- or single-episode environments.

This script uses the rLLM AgentExecutionEngine to evaluate OpenAI models
(or OpenAI-compatible APIs) on multi-episode or single-episode game environments.
It collects per-episode metrics and aggregates them by task type.

Example usage:
    python scripts/eval/openai.py \
        --config configs/eval_config.yaml \
        --model gpt-4o-mini \
        --n-parallel 32 \
        --output results/eval_gpt4o_mini.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from transformers import AutoTokenizer
# Add repo root to path for imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.gem_text_agent import GEMTextAgent  # noqa: E402
from envs.multi_episode_env import MultiEpisodeEnv  # noqa: E402
from envs.single_episode_env import SingleEpisodeEnv  # noqa: E402
from trainers.multi_episode_trainer import (  # noqa: E402
    MultiEpisodeAsyncAgentExecutionEngine,
)

from rllm.parser import ChatTemplateParser


class EvalEngine(MultiEpisodeAsyncAgentExecutionEngine):
    """Execution engine for evaluation using Token mode.

    Extends MultiEpisodeAsyncAgentExecutionEngine to use Token mode by default,
    which returns dicts with metrics from env.get_metrics().
    """

    async def execute_tasks(self, tasks: list[dict]) -> list[dict]:
        """Execute tasks using Token mode to get metrics in results.

        Overrides parent to use mode="Token", which returns dicts containing
        metrics extracted via env.get_metrics() by the parent class.

        Args:
            tasks: List of task dictionaries.

        Returns:
            List of result dicts with metrics, chat_completions, etc.
        """
        from concurrent.futures import ThreadPoolExecutor
        import asyncio

        from rllm.agents.agent import BaseAgent
        from rllm.utils import colorful_print

        if not hasattr(self, "executor") or self.executor._shutdown:
            self.executor = ThreadPoolExecutor(max_workers=self.max_env_workers)

        max_concurrent = self.n_parallel_agents
        all_results = {}
        task_queue = list(enumerate(tasks))
        semaphore = asyncio.Semaphore(max_concurrent)
        index_queue: asyncio.Queue[int] = asyncio.Queue(maxsize=max_concurrent)
        for i in range(max_concurrent):
            index_queue.put_nowait(i)

        completed = 0
        total = len(tasks)

        async def sem_wrapper(task_id, task):
            nonlocal completed
            async with semaphore:
                index = await index_queue.get()
                try:
                    self.envs[index] = self.env_class.from_dict({**task, **self.env_args})
                    self.agents[index] = self.agent_class(**self.agent_args)
                    assert isinstance(self.agents[index], BaseAgent)
                    self.agents[index].trajectory.task = task
                    # Use Token mode to get metrics in result dict
                    res = await self.run_agent_trajectory_async(
                        index, application_id=str(task_id), mode="Token"
                    )
                    res["task"] = task
                    completed += 1
                    colorful_print(f"Progress: {completed}/{total} trajectories completed", "cyan")
                    return task_id, res
                finally:
                    await index_queue.put(index)

        results = await asyncio.gather(*[sem_wrapper(tid, t) for tid, t in task_queue])
        all_results = {task_id: result for task_id, result in results}
        ordered_results = [all_results[i] for i in range(len(all_results))]

        self.executor.shutdown(wait=False, cancel_futures=True)
        return ordered_results


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed arguments namespace.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate OpenAI models on multi- or single-episode environments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file with task definitions (val_tasks section).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="OpenAI model name (e.g., gpt-4o-mini, gpt-4o, gpt-3.5-turbo).",
    )
    parser.add_argument(
        "--env-mode",
        type=str,
        default="multi",
        choices=["multi", "single"],
        help="Evaluation mode: multi-episode or single-episode environment wrapper.",
    )

    # Optional arguments
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for results JSON. Defaults to results/eval_{model}_{timestamp}.json.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="https://api.openai.com/v1",
        help="OpenAI API base URL (for Azure or other compatible endpoints).",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key. Defaults to OPENAI_API_KEY environment variable.",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=32,
        help="Number of parallel agent-environment pairs.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for the model.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling parameter.",
    )
    parser.add_argument(
        "--max-response-length",
        type=int,
        default=4096,
        help="Maximum response length in tokens.",
    )
    parser.add_argument(
        "--trajectory-timeout",
        type=int,
        default=600,
        help="Timeout for each trajectory in seconds.",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=None,
        help="Custom system prompt for the agent.",
    )
    parser.add_argument(
        "--log-chat-completions",
        action="store_true",
        default=True,
        help="Log chat completions to JSONL files (default: enabled).",
    )
    parser.add_argument(
        "--no-log-chat-completions",
        action="store_true",
        help="Disable chat completions logging.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for task generation.",
    )
    parser.add_argument(
        "--n-rollouts",
        type=int,
        default=1,
        help="Number of rollouts per task for pass@k computation.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging.",
    )

    args = parser.parse_args()

    # Handle log-chat-completions flag
    if args.no_log_chat_completions:
        args.log_chat_completions = False

    # Set API key from environment if not provided
    if args.api_key is None:
        args.api_key = os.getenv("OPENAI_API_KEY")
        if args.api_key is None:
            parser.error("--api-key not provided and OPENAI_API_KEY not set.")

    # Generate default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_safe = args.model.replace("/", "_").replace(":", "_")
        args.output = f"results/eval_{model_safe}_{timestamp}.json"

    return args


def _get_task_specific_seed(task_cfg: Dict[str, Any], global_seed: int) -> int:
    """Generate a deterministic seed for a task based on its configuration.

    This ensures that identical task configurations will always generate the same
    seeds, regardless of what other tasks are in the config or their order.
    
    NOTE: This must match the implementation in data/prepare_gem_data.py to ensure
    consistency between training/validation and OpenAI evaluation.

    Args:
        task_cfg: Task configuration dictionary.
        global_seed: Global seed for reproducibility.

    Returns:
        A deterministic seed integer for this specific task configuration.
    """
    import hashlib

    # Create a deterministic representation of the task config
    # Exclude size parameters (train_size/test_size) as they don't affect task identity
    task_params = {
        k: v for k, v in task_cfg.items() if k not in ["train_size", "test_size"]
    }

    # Sort items to ensure consistent hashing regardless of dict order
    sorted_items = sorted(task_params.items())

    # Create a hash from the task parameters and global seed
    # Use a stable hash function (SHA256) and convert to int
    param_str = str(sorted_items) + str(global_seed)
    hash_bytes = hashlib.sha256(param_str.encode()).digest()
    # Convert first 8 bytes to int (ensures positive value)
    task_seed = int.from_bytes(hash_bytes[:8], byteorder="big") % (2**31)

    return task_seed


def load_eval_tasks(
    config_path: str,
    seed: int = 42,
    n_rollouts: int = 1,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load evaluation tasks from config file.

    Uses the same deterministic seeding strategy as prepare_multi_task_gem_data()
    to ensure consistency between training/validation and OpenAI evaluation.
    Each task type gets a deterministic seed based on its configuration hash,
    ensuring the same tasks are generated regardless of task order in the config.

    Args:
        config_path: Path to YAML config file.
        seed: Random seed for task generation.
        n_rollouts: Number of rollouts per task.

    Returns:
        Tuple of (list of task dicts, raw config dict).

    Raises:
        FileNotFoundError: If config file doesn't exist.
        ValueError: If val_tasks is missing from config.
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    val_tasks = config.get("val_tasks", [])
    if not val_tasks:
        raise ValueError("val_tasks must be specified in config")

    # Generate tasks with seeds using per-task deterministic seeding
    # This matches the strategy in data/prepare_gem_data.py
    all_tasks = []

    for task_cfg in val_tasks:
        env_id = task_cfg.get("env_id")
        if env_id is None:
            raise ValueError(f"env_id is required in task config: {task_cfg}")

        test_size = task_cfg.get("test_size", 64)

        # Generate a deterministic seed for this specific task configuration
        # This ensures identical task configs always generate the same seeds
        task_specific_seed = _get_task_specific_seed(task_cfg, seed)
        task_rng = np.random.default_rng(task_specific_seed)
        seeds = task_rng.integers(0, 1_000_000, size=test_size).tolist()

        for task_seed in seeds:
            # Create n_rollouts copies of each task
            for rollout_idx in range(n_rollouts):
                task_dict: Dict[str, Any] = {}

                # Copy all parameters from task_cfg except size parameters
                for key, value in task_cfg.items():
                    if key not in ["train_size", "test_size"]:
                        task_dict[key] = value

                # Add generated fields
                task_dict["seed"] = int(task_seed)
                task_dict["uid"] = f"{env_id}-{task_seed}-{rollout_idx}"
                task_dict["data_source"] = env_id
                task_dict["rollout_idx"] = rollout_idx

                all_tasks.append(task_dict)

    logger.info(f"Loaded {len(all_tasks)} tasks from {len(val_tasks)} task types")
    return all_tasks, config


def get_default_system_prompt(env_mode: str) -> str:
    """Return a default system prompt for the selected evaluation mode."""
    if env_mode == "single":
        return (
            "You are solving a task in a single episode. "
            "Analyze the situation carefully and take the best actions to succeed. "
            "Think briefly and respond with actions inside \\boxed{} each turn. Overlong responses will be penalized."
        )
    return (
        "You are solving the same task across multiple episodes with a fixed total step budget. "
        "Each episode resets the environment but keeps the task identical. "
        "Leverage information gathered from earlier episodes to succeed faster. "
        # "Think step by step and respond with actions inside \\boxed{} each turn."
        "Think briefly and respond with actions inside \\boxed{} each turn. "
        "Overlong responses will be penalized."
        
    )


def create_engine(
    args: argparse.Namespace,
    max_steps: int,
    env_args: Dict[str, Any],
    agent_args: Dict[str, Any],
    env_class: type,
) -> EvalEngine:
    """Create the execution engine for evaluation.

    Uses EvalEngine which extracts metrics from environments that implement
    get_metrics(), ensuring metrics are captured even for early termination
    (TRUNCATION, TIMEOUT, MAX_STEPS, etc.).

    Args:
        args: Parsed command-line arguments.
        max_steps: Maximum steps per trajectory.
        env_args: Environment arguments.
        agent_args: Agent arguments.
        env_class: Environment wrapper class (multi- or single-episode).

    Returns:
        Configured EvalEngine.
    """
    rollout_engine_args = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "sampling_params": {
            "temperature": args.temperature,
            "top_p": args.top_p,
        },
    }
    # dummy tokenizer and parser for openai engine
    dummy_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    dummy_parser = ChatTemplateParser.get_parser(dummy_tokenizer)
    
    # Create dummy config with OmegaConf for attribute access (required by AgentExecutionEngine)
    from omegaconf import OmegaConf
    config = OmegaConf.create({
        "rllm": {
            "filter_token_mismatch": False,
            "disable_thinking": False,
        }
    })
    engine = EvalEngine(
        agent_class=GEMTextAgent,
        env_class=env_class,
        config=config,
        agent_args=agent_args,
        env_args=env_args,
        engine_name="openai",
        rollout_engine_args=rollout_engine_args,
        # chat_parser=dummy_parser,
        tokenizer=dummy_tokenizer,
        n_parallel_agents=args.n_parallel,
        max_steps=max_steps,
        max_response_length=args.max_response_length,
        max_prompt_length=1024,  # Generous prompt length for multi-turn
        trajectory_timeout=args.trajectory_timeout,
        sampling_params={
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_completion_tokens": args.max_response_length,
            "reasoning_effort": "low",
        },
    )

    return engine


def aggregate_metrics_by_source(
    trajectory_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Aggregate metrics by data_source (env_id).

    Args:
        trajectory_results: List of trajectory result dicts with metrics.

    Returns:
        Dict mapping data_source to aggregated metrics.
    """
    source_metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for result in trajectory_results:
        data_source = result.get("data_source", "unknown")
        metrics = result.get("metrics", {})
        source_metrics[data_source].append(metrics)

    aggregated: Dict[str, Dict[str, Any]] = {}

    for data_source, metrics_list in source_metrics.items():
        if not metrics_list:
            continue

        # Collect all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())

        # Aggregate each metric
        agg: Dict[str, Any] = {
            "count": len(metrics_list),
        }

        for key in all_keys:
            values = []
            for metrics in metrics_list:
                if key in metrics:
                    value = metrics[key]
                    # Filter out -1 values (missing episodes)
                    if isinstance(value, (int, float)) and value >= 0:
                        values.append(float(value))

            if values:
                agg[f"{key}/mean"] = float(np.mean(values))
                agg[f"{key}/std"] = float(np.std(values))
                agg[f"{key}/min"] = float(np.min(values))
                agg[f"{key}/max"] = float(np.max(values))

        aggregated[data_source] = agg

    return aggregated


def compute_pass_at_k(
    trajectory_results: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    """Compute pass@k metrics grouped by data_source.

    Pass@k measures whether at least one of k rollouts succeeded.

    Args:
        trajectory_results: List of trajectory result dicts.

    Returns:
        Dict mapping data_source to pass@k metrics.
    """
    # Group by (data_source, seed) to find unique tasks
    task_results: Dict[str, Dict[int, List[bool]]] = defaultdict(lambda: defaultdict(list))

    for result in trajectory_results:
        data_source = result.get("data_source", "unknown")
        seed = result.get("seed", 0)
        is_correct = result.get("is_correct", False)
        task_results[data_source][seed].append(is_correct)

    pass_at_k_results: Dict[str, Dict[str, float]] = {}

    for data_source, seed_results in task_results.items():
        total_tasks = len(seed_results)
        if total_tasks == 0:
            continue

        # Compute pass@1 (first rollout success rate)
        pass_at_1_count = sum(
            1 for results in seed_results.values() if results and results[0]
        )
        pass_at_1 = pass_at_1_count / total_tasks

        # Compute pass@k (any rollout succeeded)
        pass_at_k_count = sum(
            1 for results in seed_results.values() if any(results)
        )
        pass_at_k = pass_at_k_count / total_tasks

        # Determine k (max rollouts per task)
        k = max(len(results) for results in seed_results.values())

        pass_at_k_results[data_source] = {
            "pass_at_1": pass_at_1,
            f"pass_at_{k}": pass_at_k,
            "num_tasks": total_tasks,
            "num_rollouts": k,
        }

    return pass_at_k_results


def sanitize_filename(name: str) -> str:
    """Sanitize a string to be used as a filename.

    Args:
        name: Original string.

    Returns:
        Sanitized filename-safe string.
    """
    # Replace problematic characters with underscores
    for char in [":", "/", "\\", " ", "?", "*", "<", ">", "|", '"']:
        name = name.replace(char, "_")
    return name


def log_chat_completions(
    trajectory_results: List[Dict[str, Any]],
    output_path: Path,
    model_name: str,
) -> None:
    """Log chat completions to JSONL files organized by data_source.

    Args:
        trajectory_results: List of trajectory result dicts.
        output_path: Output results file path (used for consistent timestamp).
    """
    chat_dir = output_path.parent / "chat_completions"
    chat_dir.mkdir(parents=True, exist_ok=True)
    stem_parts = output_path.stem.split("_")
    if len(stem_parts) >= 2 and stem_parts[-2].isdigit() and stem_parts[-1].isdigit():
        timestamp_suffix = f"{stem_parts[-2]}_{stem_parts[-1]}"
    else:
        timestamp_suffix = stem_parts[-1]

    # Group by data_source
    source_chats: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for result in trajectory_results:
        data_source = result.get("data_source", "unknown")
        chat_entry = {
            "uid": result.get("uid", ""),
            "task": result.get("task", {}),
            "messages": result.get("chat_completions", []),
            "metrics": result.get("metrics", {}),
            "is_correct": result.get("is_correct", False),
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
        }
        source_chats[data_source].append(chat_entry)

    # Custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    # Write JSONL files
    model_safe = sanitize_filename(model_name)
    for data_source, chats in source_chats.items():
        filename = f"{sanitize_filename(data_source)}_{model_safe}_{timestamp_suffix}.jsonl"
        filepath = chat_dir / filename

        with open(filepath, "w") as f:
            for chat in chats:
                f.write(json.dumps(chat, ensure_ascii=False, cls=NumpyEncoder) + "\n")

        logger.info(f"Saved {len(chats)} chat logs to {filepath}")


async def run_evaluation(
    tasks: List[Dict[str, Any]],
    engine: EvalEngine,
    config: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Run evaluation on all tasks.

    Args:
        tasks: List of task dictionaries.
        engine: Configured EvalEngine.
        config: Raw config dict for env_args construction.

    Returns:
        List of trajectory result dictionaries.
    """
    logger.info(f"Starting evaluation on {len(tasks)} tasks...")

    # Execute tasks (returns dicts in Token mode)
    token_results = await engine.execute_tasks(tasks)

    # Collect results from Token mode output
    results = []
    for idx, token_result in enumerate(token_results):
        task = token_result.get("task", tasks[idx])

        # Extract metrics from Token mode result
        # Metrics keys are flattened (e.g., "episode_success_rate" instead of "episode/success_rate")
        raw_metrics = token_result.get("metrics", {})

        # Convert flattened keys back to original format for consistency
        metrics = {}
        for key, value in raw_metrics.items():
            # Handle different episode metric patterns:
            # - "episode_success_rate" -> "episode/success_rate"
            # - "episode_1_steps" -> "episode_1/steps"
            # - "episode_1_success_rate" -> "episode_1/success_rate"
            if key.startswith("episode_"):
                # Match "episode_N_..." pattern (per-episode metrics)
                match = re.match(r"^(episode_\d+)_(.+)$", key)
                if match:
                    original_key = f"{match.group(1)}/{match.group(2)}"
                else:
                    # Match "episode_..." pattern (aggregate metrics)
                    original_key = key.replace("_", "/", 1)
            else:
                original_key = key
            metrics[original_key] = value

        # Determine correctness from metrics or trajectory reward
        trajectory_reward = token_result.get("trajectory_reward", 0.0)
        is_correct = metrics.get("episode/success_rate", 0.0) > 0 or trajectory_reward > 0

        # Get chat completions
        chat_completions = token_result.get("chat_completions", [])

        result = {
            "uid": task.get("uid", f"task-{idx}"),
            "data_source": task.get("data_source", "unknown"),
            "seed": task.get("seed", 0),
            "rollout_idx": task.get("rollout_idx", 0),
            "task": task,
            "metrics": metrics,
            "is_correct": is_correct,
            "chat_completions": chat_completions,
            "trajectory_reward": trajectory_reward,
        }

        results.append(result)

    return results


def save_results(
    results: List[Dict[str, Any]],
    aggregated_metrics: Dict[str, Dict[str, Any]],
    pass_at_k: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
    output_path: Path,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: List of trajectory result dicts.
        aggregated_metrics: Aggregated metrics by data_source.
        pass_at_k: Pass@k metrics by data_source.
        args: Command-line arguments.
        output_path: Output file path.
    """
    # Compute overall summary
    total_correct = sum(1 for r in results if r.get("is_correct", False))
    total_tasks = len(results)
    overall_success_rate = total_correct / total_tasks if total_tasks > 0 else 0.0

    # Build per-task summary
    per_task_summary = {}
    for data_source in aggregated_metrics.keys():
        agg = aggregated_metrics[data_source]
        pk = pass_at_k.get(data_source, {})

        per_task_summary[data_source] = {
            "count": agg.get("count", 0),
            "success_rate": agg.get("episode/success_rate/mean", 0.0),
            "avg_episodes": agg.get("episode/num_episodes/mean", 0.0),
            "avg_success_count": agg.get("episode/success_count/mean", 0.0),
            "avg_episode_length": agg.get("episode/episode_length/mean", 0.0),
            **pk,
        }

    # Build output dict (exclude chat_completions from main results file)
    trajectory_summaries = []
    for r in results:
        summary = {k: v for k, v in r.items() if k != "chat_completions"}
        trajectory_summaries.append(summary)

    output = {
        "config": {
            "model": args.model,
            "base_url": args.base_url,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "n_parallel": args.n_parallel,
            "n_rollouts": args.n_rollouts,
            "env_mode": args.env_mode,
            "config_file": args.config,
            "timestamp": datetime.now().isoformat(),
        },
        "summary": {
            "overall_success_rate": overall_success_rate,
            "total_tasks": total_tasks,
            "total_correct": total_correct,
            "per_task": per_task_summary,
        },
        "aggregated_metrics": aggregated_metrics,
        "trajectories": trajectory_summaries,
    }

    # Custom encoder to handle numpy types
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)

    logger.info(f"Results saved to {output_path}")


def print_summary(
    aggregated_metrics: Dict[str, Dict[str, Any]],
    pass_at_k: Dict[str, Dict[str, float]],
    overall_success_rate: float,
) -> None:
    """Print evaluation summary to console.

    Args:
        aggregated_metrics: Aggregated metrics by data_source.
        pass_at_k: Pass@k metrics by data_source.
        overall_success_rate: Overall success rate.
    """
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"\nOverall Success Rate: {overall_success_rate:.2%}")
    print("\nPer-Task Results:")
    print("-" * 60)

    for data_source in sorted(aggregated_metrics.keys()):
        agg = aggregated_metrics[data_source]
        pk = pass_at_k.get(data_source, {})

        print(f"\n{data_source}:")
        print(f"  Count: {agg.get('count', 0)}")
        print(f"  Success Rate: {agg.get('episode/success_rate/mean', 0):.2%}")
        if "episode/num_episodes/mean" in agg:
            print(f"  Avg Episodes: {agg.get('episode/num_episodes/mean', 0):.2f}")
        if "episode/success_count/mean" in agg:
            print(f"  Avg Success Count: {agg.get('episode/success_count/mean', 0):.2f}")
        if "episode/episode_length/mean" in agg:
            print(f"  Avg Episode Length: {agg.get('episode/episode_length/mean', 0):.2f}")

        if pk:
            print(f"  Pass@1: {pk.get('pass_at_1', 0):.2%}")
            k = pk.get("num_rollouts", 1)
            if k > 1:
                print(f"  Pass@{k}: {pk.get(f'pass_at_{k}', 0):.2%}")

    print("\n" + "=" * 60)


async def main() -> None:
    """Main entry point for evaluation."""
    args = parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info(f"Evaluating model: {args.model}")
    logger.info(f"Config file: {args.config}")
    logger.info(f"Environment mode: {args.env_mode}")

    # Load tasks
    tasks, config = load_eval_tasks(
        args.config,
        seed=args.seed,
        n_rollouts=args.n_rollouts,
    )

    # Determine max_steps from config based on env_mode
    val_tasks = config.get("val_tasks", [])
    if args.env_mode == "single":
        max_steps = max(
            (task.get("max_turns_per_episode", 30) for task in val_tasks),
            default=30,
        )
        logger.info(f"Max turns per episode: {max_steps}")
        env_class = SingleEpisodeEnv
    else:
        max_steps = max(
            (task.get("total_step_cap", 30) for task in val_tasks),
            default=30,
        )
        logger.info(f"Max total step cap: {max_steps}")
        env_class = MultiEpisodeEnv

    # Build env_args from config (base configuration)
    # Per-task configuration (including inner_env_class) comes from task dict via from_dict
    env_args: Dict[str, Any] = {
        "success_reward": 1.0,
        "episode_header": "New episode begins.",
    }

    # Build agent_args
    system_prompt = args.system_prompt or get_default_system_prompt(args.env_mode)
    agent_args: Dict[str, Any] = {
        "system_prompt": system_prompt,
        "max_steps": max_steps,
    }

    # Create engine
    engine = create_engine(args, max_steps, env_args, agent_args, env_class)

    # Run evaluation
    results = await run_evaluation(tasks, engine, config)

    # Aggregate metrics
    aggregated_metrics = aggregate_metrics_by_source(results)
    pass_at_k = compute_pass_at_k(results)

    # Compute overall success rate
    total_correct = sum(1 for r in results if r.get("is_correct", False))
    overall_success_rate = total_correct / len(results) if results else 0.0

    # Save results
    output_path = Path(args.output)
    save_results(results, aggregated_metrics, pass_at_k, args, output_path)

    # Log chat completions if enabled
    if args.log_chat_completions:
        log_chat_completions(results, output_path, args.model)

    # Print summary
    print_summary(aggregated_metrics, pass_at_k, overall_success_rate)

    # Cleanup
    engine.shutdown()

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    asyncio.run(main())