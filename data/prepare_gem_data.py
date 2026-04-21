"""Prepare and register simple GEM task datasets for train/val.

Mirrors the frozenlake data prep: generates task dictionaries and
registers them via DatasetRegistry so AgentTrainer can load them.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import yaml
from rllm.data.dataset import DatasetRegistry  # type: ignore


def prepare_gem_data(
    train_size: int = 1024,
    test_size: int = 128,
    env_id: str = "game:GuessTheNumber-v0-hard",
    seed: int = 42,
) -> Tuple[object, object]:
    """Create and register train/val splits of GEM task specs.

    Args:
        train_size: number of training tasks.
        test_size: number of validation tasks.
        env_id: GEM environment id to target.

    Returns:
        (train_dataset, test_dataset) as registered DatasetRegistry entries.
    """

    rng = np.random.default_rng(seed)
    train_seeds = rng.integers(0, 1_000_000, size=train_size).tolist()
    test_seeds = rng.integers(0, 1_000_000, size=test_size).tolist()

    def task_fn(idx: int, task_seed: int) -> dict:
        return {
            "env_id": env_id,
            "seed": int(task_seed),
            "uid": f"{env_id}-{task_seed}",
            "data_source": env_id,  # Add data_source for metric grouping
        }

    train_data = [task_fn(i, s) for i, s in enumerate(train_seeds)]
    test_data = [task_fn(i, s) for i, s in enumerate(test_seeds)]

    # Temporarily override verl postprocessing to include data_source at top level
    # (verl extracts top-level fields into non_tensor_batch)
    original_postprocessing = DatasetRegistry.apply_verl_postprocessing

    def gem_verl_postprocessing(data: list[dict]) -> list[dict]:
        """Custom verl postprocessing that preserves data_source at top level."""
        processed_data = []
        for entry in data:
            processed_entry = {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None,
                },
                "extra_info": entry,
            }
            # Preserve data_source at top level so verl can extract it
            if "data_source" in entry:
                processed_entry["data_source"] = entry["data_source"]
            processed_data.append(processed_entry)
        return processed_data

    DatasetRegistry.apply_verl_postprocessing = staticmethod(gem_verl_postprocessing)

    try:
        train_dataset = DatasetRegistry.register_dataset("gem_tasks", train_data, "train")
        test_dataset = DatasetRegistry.register_dataset("gem_tasks", test_data, "test")
    finally:
        # Restore original postprocessing
        DatasetRegistry.apply_verl_postprocessing = original_postprocessing

    return train_dataset, test_dataset


def prepare_multi_task_gem_data(
    tasks_config_path: Optional[str] = None,
    tasks_config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
) -> Tuple[object, object]:
    """Create and register train/val splits for multiple GEM tasks.

    Each task can have its own configuration (e.g., max_turns_per_episode, total_step_cap, etc.).
    All parameters except train_size/test_size are included in the task dictionary.

    Args:
        tasks_config_path: Path to YAML config file containing task configurations.
            Expected format:
                train_tasks:
                  - env_id: "game:TaskName-v0"
                    max_turns_per_episode: 7
                    total_step_cap: 21
                    train_size: 512
                    inner_env_class: 'envs.gem_env_adapter.GEMEnvAdapter'
                    ... any other task-specific parameters ...
                val_tasks:
                  - env_id: "game:TaskName-v0"
                    max_turns_per_episode: 7
                    total_step_cap: 21
                    test_size: 64
                    ... any other task-specific parameters ...
        tasks_config: Optional dict with 'train_tasks' and/or 'val_tasks' keys.
        seed: Random seed for generating task seeds.

    Returns:
        (train_dataset, test_dataset) as registered DatasetRegistry entries.
        Each task dict in the dataset includes all parameters from the config
        (except train_size/test_size), plus:
            - seed: Task seed
            - uid: Unique identifier
            - data_source: env_id (for metric grouping)

    Raises:
        ValueError: If neither tasks_config_path nor tasks_config is provided,
            or if train_tasks/val_tasks are missing.
        FileNotFoundError: If tasks_config_path is provided but file doesn't exist.
    """
    # Load config from file or use provided config
    if tasks_config is None:
        if tasks_config_path is None:
            raise ValueError("Either tasks_config_path or tasks_config must be provided")
        config_path = Path(tasks_config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {tasks_config_path}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = tasks_config

    train_tasks = config.get("train_tasks", [])
    val_tasks = config.get("val_tasks", [])

    if not train_tasks:
        raise ValueError("train_tasks must be specified in config")
    if not val_tasks:
        raise ValueError("val_tasks must be specified in config")

    def get_task_specific_seed(task_cfg: Dict[str, Any], global_seed: int) -> int:
        """Generate a deterministic seed for a task based on its configuration.
        
        This ensures that identical task configurations will always generate the same
        seeds, regardless of what other tasks are in the config or their order.
        
        Args:
            task_cfg: Task configuration dictionary.
            global_seed: Global seed for reproducibility.
            
        Returns:
            A deterministic seed integer for this specific task configuration.
        """
        # Create a deterministic representation of the task config
        # Exclude size parameters (train_size/test_size) as they don't affect task identity
        task_params = {
            k: v for k, v in task_cfg.items() 
            if k not in ["train_size", "test_size"]
        }
        
        # Sort items to ensure consistent hashing regardless of dict order
        sorted_items = sorted(task_params.items())
        
        # Create a hash from the task parameters and global seed
        # Use a stable hash function (SHA256) and convert to int
        param_str = str(sorted_items) + str(global_seed)
        hash_bytes = hashlib.sha256(param_str.encode()).digest()
        # Convert first 8 bytes to int (ensures positive value)
        task_seed = int.from_bytes(hash_bytes[:8], byteorder='big') % (2**31)
        
        return task_seed

    def process_task_list(task_list: List[Dict[str, Any]], is_train: bool) -> List[Dict[str, Any]]:
        """Process a list of task configs and return task dictionaries.
        
        All parameters from task_cfg are copied to the task_dict, except:
        - train_size / test_size (used only to determine number of tasks to generate)
        
        Each task gets its own deterministic RNG based on its configuration, ensuring
        that identical task configs generate the same seeds regardless of other tasks.
        """
        result = []
        for task_cfg in task_list:
            # Extract size parameter (train_size or test_size)
            size_key = "train_size" if is_train else "test_size"
            size = task_cfg.get(size_key, 512 if is_train else 64)
            
            # Get env_id (required field)
            if "env_id" not in task_cfg:
                raise ValueError(f"env_id is required in task config: {task_cfg}")
            env_id = task_cfg["env_id"]

            # Sequential-seed mode: seeds = [0, 1, ..., size-1]. Useful for
            # envs like WebShop where the adapter maps seed -> goal index via
            # `seed % num_goals`, so this selects the first `size` tasks of
            # the split deterministically.
            if task_cfg.get("sequential_seeds", False):
                seeds = list(range(size))
            else:
                # Generate a deterministic seed for this specific task configuration
                task_specific_seed = get_task_specific_seed(task_cfg, seed)
                if is_train:
                    task_specific_seed = task_specific_seed * 2

                # Create a task-specific RNG to ensure deterministic seed generation
                # This ensures identical task configs always generate the same seeds
                task_rng = np.random.default_rng(task_specific_seed)
                seeds = task_rng.integers(0, 1_000_000, size=size).tolist()

            def task_fn(idx: int, task_seed: int) -> dict:
                """Create a task dictionary with all task-specific configuration.
                
                Copies all parameters from task_cfg except train_size/test_size.
                """
                task_dict: Dict[str, Any] = {}
                
                # Copy all parameters from task_cfg except size / seeding-mode parameters
                for key, value in task_cfg.items():
                    if key not in ["train_size", "test_size", "sequential_seeds"]:
                        task_dict[key] = value
                
                # Add generated/required fields
                task_dict["seed"] = int(task_seed)
                task_dict["uid"] = f"{env_id}-{task_seed}"
                task_dict["data_source"] = env_id  # For metric grouping
                
                return task_dict

            result.extend([task_fn(i, s) for i, s in enumerate(seeds)])
        return result

    # Process train and val tasks
    all_train_data = process_task_list(train_tasks, is_train=True)
    all_test_data = process_task_list(val_tasks, is_train=False)

    # Temporarily override verl postprocessing to include data_source at top level
    # (verl extracts top-level fields into non_tensor_batch)
    original_postprocessing = DatasetRegistry.apply_verl_postprocessing

    def gem_verl_postprocessing(data: list[dict]) -> list[dict]:
        """Custom verl postprocessing that preserves data_source at top level."""
        processed_data = []
        for entry in data:
            processed_entry = {
                "prompt": [{"role": "user", "content": "placeholder"}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": None,
                },
                "extra_info": entry,
            }
            # Preserve data_source at top level so verl can extract it
            if "data_source" in entry:
                processed_entry["data_source"] = entry["data_source"]
            processed_data.append(processed_entry)
        return processed_data

    DatasetRegistry.apply_verl_postprocessing = staticmethod(gem_verl_postprocessing)

    try:
        train_dataset = DatasetRegistry.register_dataset("gem_tasks", all_train_data, "train")
        test_dataset = DatasetRegistry.register_dataset("gem_tasks", all_test_data, "test")
    finally:
        # Restore original postprocessing
        DatasetRegistry.apply_verl_postprocessing = original_postprocessing

    return train_dataset, test_dataset


if __name__ == "__main__":
    # Test single-task (backward compatibility)
    train_ds, test_ds = prepare_gem_data()
    print(f"Single-task - Train: {len(train_ds.get_data())} examples")
    print(f"Single-task - Test: {len(test_ds.get_data())} examples")
    print(f"Sample: {train_ds.get_data()[0]}")

    # Test multi-task
    config_path = Path(__file__).parent.parent / "configs" / "multi_task_gem_config.yaml"
    if config_path.exists():
        train_ds_multi, test_ds_multi = prepare_multi_task_gem_data(tasks_config_path=str(config_path))
        print(f"\nMulti-task - Train: {len(train_ds_multi.get_data())} examples")
        print(f"Multi-task - Test: {len(test_ds_multi.get_data())} examples")
        print(f"Sample: {train_ds_multi.get_data()[0]}")

