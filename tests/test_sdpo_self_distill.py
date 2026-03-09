from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib.machinery
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys
import types

import torch
from omegaconf import OmegaConf

_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_ROOT, _ROOT / "third_party" / "rllm", _ROOT / "third_party" / "verl"):
    _path_str = str(_extra_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)
if "polars" not in sys.modules:
    polars_stub = types.ModuleType("polars")
    polars_stub.__spec__ = importlib.machinery.ModuleSpec("polars", loader=None)
    sys.modules["polars"] = polars_stub
elif getattr(sys.modules["polars"], "__spec__", None) is None:
    sys.modules["polars"].__spec__ = importlib.machinery.ModuleSpec("polars", loader=None)

from rllm.agents.utils import convert_messages_to_tokens_and_masks
from trainers.sdpo_self_distill_trainer import (
    collect_complete_attempt_summaries,
    DistillPayload,
    DistillSettings,
    JointSDPOSelfDistillTrainer,
    build_hindsight_prompt_tokens_first_n_complete_attempts,
    extract_attempt_response_sequence,
    extract_initial_messages_before_first_assistant,
    extract_first_attempt_prefix,
    should_skip_denominator_overflow,
)


class _DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del text, add_special_tokens
        return []


class _FakeDataProto:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.meta_info: dict[str, Any] = {}

    def __len__(self) -> int:
        if not self.batch:
            return 0
        first_tensor = next(iter(self.batch.values()))
        return int(first_tensor.shape[0])

    def select_idxs(self, indices: Any) -> "_FakeDataProto":
        idx = torch.as_tensor(indices, dtype=torch.long)
        return _FakeDataProto({k: v[idx] for k, v in self.batch.items()})

    def union(self, other: "_FakeDataProto") -> "_FakeDataProto":
        merged = dict(self.batch)
        merged.update(other.batch)
        out = _FakeDataProto(merged)
        out.meta_info = dict(self.meta_info)
        out.meta_info.update(other.meta_info)
        return out


@dataclass
class _FakeActorOutput:
    meta_info: dict[str, Any]


class _FakeActorRolloutWG:
    def __init__(self) -> None:
        self.world_size = 1
        self.update_calls = 0
        self.last_update_batch: _FakeDataProto | None = None
        self.compute_log_prob_calls = 0
        self.compute_teacher_log_prob_calls = 0
        self.sync_calls: list[dict[str, float | str]] = []

    def compute_log_prob(self, batch: _FakeDataProto) -> _FakeDataProto:
        self.compute_log_prob_calls += 1
        responses = batch.batch["responses"].float()
        return _FakeDataProto({"old_log_probs": responses / 10.0})

    def compute_teacher_log_prob(self, batch: _FakeDataProto) -> _FakeDataProto:
        self.compute_teacher_log_prob_calls += 1
        responses = batch.batch["responses"].float()
        return _FakeDataProto({"old_log_probs": responses / 20.0})

    def sync_teacher_from_actor(self, mode: str, update_rate: float = 0.0) -> dict[str, float]:
        self.sync_calls.append({"mode": mode, "update_rate": float(update_rate)})
        return {"applied": 1.0}

    def update_actor(self, batch: _FakeDataProto) -> _FakeActorOutput:
        self.update_calls += 1
        self.last_update_batch = batch
        return _FakeActorOutput(meta_info={"metrics": {"loss": 1.0}})


def _build_trainer_for_unit_tests() -> JointSDPOSelfDistillTrainer:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.tokenizer = _DummyTokenizer()
    trainer.config = SimpleNamespace(
        data=SimpleNamespace(max_prompt_length=10),
    )
    trainer.distill_settings = DistillSettings(
        enable=True,
        lambda_coef=1.0,
        context_limit=7,
        min_distill_tokens=1,
        teacher_context_attempts=None,
    )
    trainer._latest_token_trajectories = []
    return trainer


def _make_settings_config(distill_cfg: dict[str, Any]) -> Any:
    return OmegaConf.create(
        {
            "data": {
                "max_prompt_length": 10,
                "max_response_length": 5,
            },
            "rllm": {
                "distill": distill_cfg,
            },
            "algorithm": {
                "use_kl_in_reward": False,
            },
            "actor_rollout_ref": {
                "actor": {
                    "use_kl_loss": False,
                }
            },
        }
    )


def test_denominator_overflow_guard_gt_and_eq() -> None:
    assert (
        should_skip_denominator_overflow(
            denominator_prompt_len=5,
            first_attempt_sequence_len=3,
            context_limit=7,
        )
        is True
    )
    assert (
        should_skip_denominator_overflow(
            denominator_prompt_len=4,
            first_attempt_sequence_len=3,
            context_limit=7,
        )
        is False
    )


def test_load_distill_settings_teacher_regularization_valid_modes() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config({"teacher_regularization": "none"})
    settings = trainer._load_distill_settings()
    assert settings.teacher_regularization == "none"
    assert settings.teacher_update_rate == 0.05
    assert settings.teacher_update_interval == 10

    trainer.config = _make_settings_config(
        {
            "teacher_regularization": "ema",
            "teacher_update_rate": 0.2,
        }
    )
    settings = trainer._load_distill_settings()
    assert settings.teacher_regularization == "ema"
    assert settings.teacher_update_rate == 0.2

    trainer.config = _make_settings_config(
        {
            "teacher_regularization": "every_n_steps",
            "teacher_update_interval": 3,
        }
    )
    settings = trainer._load_distill_settings()
    assert settings.teacher_regularization == "every_n_steps"
    assert settings.teacher_update_interval == 3


def test_load_distill_settings_sdpo_loss_controls() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config(
        {
            "loss_variant": "full_logit",
            "alpha": 0.25,
            "is_clip": 2.0,
            "full_logit_topk": 32,
            "full_logit_add_tail": False,
        }
    )
    settings = trainer._load_distill_settings()
    assert settings.loss_variant == "full_logit"
    assert abs(settings.alpha - 0.25) < 1e-9
    assert settings.is_clip == 2.0
    assert settings.full_logit_topk == 32
    assert settings.full_logit_add_tail is False
    assert settings.negate_sdpo_loss is False
    assert settings.lambda_coef == 1.0
    assert settings.mode == "sdpo_self"
    assert settings.use_grpo_loss is True

    trainer.config = _make_settings_config({"loss_variant": "non_full", "alpha": 0.5})
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "alpha" in str(exc)
    else:
        raise AssertionError("Expected ValueError for non_full alpha != 1.0")


def test_load_distill_settings_negate_sdpo_loss() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config({"negate_sdpo_loss": True})
    settings = trainer._load_distill_settings()
    assert settings.negate_sdpo_loss is True


def test_load_distill_settings_allows_full_logit_topk_zero() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config({"loss_variant": "full_logit", "full_logit_topk": 0})
    settings = trainer._load_distill_settings()
    assert settings.full_logit_topk == 0


def test_load_distill_settings_mode_sdpo_pure_disables_grpo_loss() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config({"mode": "sdpo_pure"})
    settings = trainer._load_distill_settings()
    assert settings.mode == "sdpo_pure"
    assert settings.use_grpo_loss is False
    assert settings.lambda_coef == 1.0


def test_load_distill_settings_mode_invalid_raises() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config({"mode": "bad_mode"})
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "distill.mode" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid distill mode")


def test_load_distill_settings_trajectory_selection_validates_modes() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config(
        {
            "trajectory_selection": "selective_retry_success_n2",
            "teacher_context_attempts": 1,
        }
    )
    settings = trainer._load_distill_settings()
    assert settings.trajectory_selection == "selective_retry_success_n2"

    trainer.config = _make_settings_config({"trajectory_selection": "bad_selection"})
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "trajectory_selection" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid trajectory_selection")


def test_load_distill_settings_selective_retry_requires_teacher_context_attempts_one() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config(
        {
            "trajectory_selection": "selective_retry_success_n2",
            "teacher_context_attempts": 2,
        }
    )
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "teacher_context_attempts" in str(exc)
    else:
        raise AssertionError("Expected ValueError for selective_retry_success_n2 without teacher_context_attempts=1")


def test_load_distill_settings_teacher_regularization_invalid_mode() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config({"teacher_regularization": "bad_mode"})
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "teacher_regularization" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid teacher_regularization")


def test_load_distill_settings_teacher_regularization_invalid_ema_rate() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config(
        {
            "teacher_regularization": "ema",
            "teacher_update_rate": 1.2,
        }
    )
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "teacher_update_rate" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid teacher_update_rate")


def test_load_distill_settings_teacher_regularization_invalid_interval() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    trainer.config = _make_settings_config(
        {
            "teacher_regularization": "every_n_steps",
            "teacher_update_interval": 0,
        }
    )
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "teacher_update_interval" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid teacher_update_interval")


def test_load_distill_settings_teacher_regularization_kl_guard() -> None:
    trainer = JointSDPOSelfDistillTrainer.__new__(JointSDPOSelfDistillTrainer)
    cfg = _make_settings_config({"teacher_regularization": "ema"})
    cfg.algorithm.use_kl_in_reward = True
    trainer.config = cfg
    try:
        trainer._load_distill_settings()
    except ValueError as exc:
        assert "incompatible" in str(exc).lower()
    else:
        raise AssertionError("Expected ValueError for KL incompatibility")


def test_denominator_overflow_guard_counts_prompt_in_denominator_context() -> None:
    # Regression: denominator prompt length already includes concat(c_N, prompts).
    # If prompts were omitted from denominator length, this case would not overflow.
    assert (
        should_skip_denominator_overflow(
            denominator_prompt_len=7,
            first_attempt_sequence_len=3,
            context_limit=9,
        )
        is True
    )


def test_extract_first_attempt_prefix_uses_last_first_attempt_index() -> None:
    extracted = extract_first_attempt_prefix(
        response_tokens=torch.tensor([10, 90, 11, 91, 12, 0], dtype=torch.long),
        response_mask=torch.tensor([1, 1, 1, 1, 1, 0], dtype=torch.long),
        first_attempt_response_mask=torch.tensor([1, 0, 0, 1, 0, 0], dtype=torch.float32),
    )
    assert extracted is not None
    prefix_tokens, prefix_response_mask, prefix_distill_mask = extracted
    assert prefix_tokens.tolist() == [10, 90, 11, 91]
    assert prefix_response_mask.tolist() == [1, 1, 1, 1]
    assert prefix_distill_mask.tolist() == [1.0, 0.0, 0.0, 1.0]


def test_complete_trajectory_extracts_first_attempt_and_first_n_attempt_contexts() -> None:
    step_records = [
        {
            "prompt_ids": [1, 2],
            "completion_ids": [11, 12],
            "episode_index": 0,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 11, 12, 21],
            "completion_ids": [13],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 11, 12, 21, 13, 22, 220],
            "completion_ids": [31],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 11, 12, 21, 13, 22, 220, 31, 23],
            "completion_ids": [32],
            "episode_index": 1,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 11, 12, 21, 13, 22, 220, 31, 23, 32, 24, 240],
            "completion_ids": [41],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 11, 12, 21, 13, 22, 220, 31, 23, 32, 24, 240, 41, 25],
            "completion_ids": [42],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
    ]
    # Full assembled response stream across all attempts:
    # completion + prompt delta + completion + ...
    response_tokens = torch.tensor([11, 12, 21, 13, 22, 31, 23, 32, 24, 41, 25, 42], dtype=torch.long)
    response_mask = torch.tensor([1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    first_attempt_response_mask = torch.tensor(
        [1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=torch.float32,
    )

    extracted = extract_first_attempt_prefix(
        response_tokens=response_tokens,
        response_mask=response_mask,
        first_attempt_response_mask=first_attempt_response_mask,
    )
    assert extracted is not None
    first_attempt_tokens, first_attempt_seq_mask, first_attempt_distill_mask = extracted
    assert first_attempt_tokens.tolist() == [11, 12, 21, 13]
    assert first_attempt_seq_mask.tolist() == [1, 1, 0, 1]
    assert first_attempt_distill_mask.tolist() == [1.0, 1.0, 0.0, 1.0]

    first_1 = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=1,
    )
    first_2 = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )
    all_complete = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=None,
    )

    assert first_1 is not None
    assert first_1.tolist() == [1, 2, 11, 12, 21, 13, 22]
    assert first_2 is not None
    assert first_2.tolist() == [1, 2, 11, 12, 21, 13, 22, 220, 31, 23, 32, 24]
    assert all_complete is not None
    assert all_complete.tolist() == [1, 2, 11, 12, 21, 13, 22, 220, 31, 23, 32, 24]


def test_extract_attempt_response_sequence_second_attempt_ignores_attempt_boundary_prompt_delta() -> None:
    step_records = [
        {
            "prompt_ids": [1, 2],
            "completion_ids": [11],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 2,
            "episode_success": False,
        },
        {
            "prompt_ids": [1, 2, 11, 21, 31, 32],
            "completion_ids": [41],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
            "episode_success": True,
        },
        {
            "prompt_ids": [1, 2, 11, 21, 31, 32, 41, 42],
            "completion_ids": [51],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
            "episode_success": True,
            "episode_done": True,
        },
    ]

    complete_attempts = collect_complete_attempt_summaries(step_records)
    assert complete_attempts is not None
    assert complete_attempts[0].context_tokens.tolist() == [1, 2, 11, 21]
    assert complete_attempts[0].success is False

    extracted = extract_attempt_response_sequence(
        step_records,
        target_episode_index=1,
    )
    assert extracted is not None
    assert extracted.response_tokens.tolist() == [41, 42, 51]
    assert extracted.response_mask.tolist() == [1, 0, 1]
    assert extracted.distill_mask.tolist() == [1.0, 0.0, 1.0]


def test_build_hindsight_prompt_tokens_first_n_complete_attempts_success() -> None:
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [2],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 3, 30],
            "completion_ids": [4],
            "episode_index": 1,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 3, 30, 4, 5, 50],
            "completion_ids": [6],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 3, 30, 4, 5, 50, 6, 7],
            "completion_ids": [8],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
    ]
    hindsight = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )
    assert hindsight is not None
    assert hindsight.tolist() == [1, 2, 3, 30, 4, 5]


def test_build_hindsight_prompt_tokens_includes_reflection_turns_for_first_n_attempts() -> None:
    # Tokens are synthetic ids:
    # - 20 represents reflection_prompt_2 tokens in each episode.
    # - 30/31 represent model reflection completions.
    # Transition-confirmed complete attempts must include both prompt and reflection.
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [10],
            "episode_index": 0,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 10, 20],
            "completion_ids": [30],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 10, 20, 30, 40],
            "completion_ids": [11],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 10, 20, 30, 40, 11, 20],
            "completion_ids": [31],
            "episode_index": 1,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 10, 20, 30, 40, 11, 20, 31, 41],
            "completion_ids": [12],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
    ]
    first_1 = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=1,
    )
    first_2 = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )

    assert first_1 is not None
    assert first_1.tolist() == [1, 10, 20, 30]
    assert first_2 is not None
    assert first_2.tolist() == [1, 10, 20, 30, 40, 11, 20, 31]


def test_build_hindsight_prompt_tokens_first_n_complete_attempts_insufficient_returns_none() -> None:
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [2],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 3, 4, 5],
            "completion_ids": [6],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
    ]
    hindsight = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )
    assert hindsight is None


def test_build_hindsight_prompt_tokens_final_completed_attempt_without_next_transition() -> None:
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [2],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
            "episode_done": True,
        },
        {
            "prompt_ids": [1, 2, 30, 31],
            "completion_ids": [4],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
            # Final attempt completes at trajectory end (no third attempt reset).
            "episode_done": True,
        },
    ]
    hindsight = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )
    assert hindsight is not None
    assert hindsight.tolist() == [1, 2, 30, 31, 4]


def test_build_hindsight_prompt_tokens_final_completed_attempt_with_reflection_step() -> None:
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [2],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            # Attempt-1 reflection turn.
            "prompt_ids": [1, 2, 20],
            "completion_ids": [30],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 20, 30, 40],
            "completion_ids": [4],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            # Final reflection turn at trajectory end (outer_done reached).
            "prompt_ids": [1, 2, 20, 30, 40, 4, 21],
            "completion_ids": [31],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
            "episode_done": True,
        },
    ]
    hindsight = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )
    assert hindsight is not None
    assert hindsight.tolist() == [1, 2, 20, 30, 40, 4, 21, 31]


def test_build_hindsight_prompt_tokens_first_n_complete_attempts_null_uses_all_complete_only() -> None:
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [2],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 3, 30],
            "completion_ids": [4],
            "episode_index": 1,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 1,
            "boundary_next_initial_env_token_len": 1,
        },
        {
            "prompt_ids": [1, 2, 3, 30, 4, 5, 50],
            "completion_ids": [6],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 3, 30, 4, 5, 50, 6, 7],
            "completion_ids": [8],
            "episode_index": 2,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
    ]
    hindsight = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=None,
    )
    assert hindsight is not None
    assert hindsight.tolist() == [1, 2, 3, 30, 4, 5]


def test_build_hindsight_prompt_tokens_first_n_complete_attempts_malformed_boundary_returns_none() -> None:
    step_records = [
        {
            "prompt_ids": [1],
            "completion_ids": [2],
            "episode_index": 0,
            "boundary_transition": True,
            "boundary_terminal_env_token_len": 2,
            "boundary_next_initial_env_token_len": 0,
        },
        {
            "prompt_ids": [1, 2, 3],
            "completion_ids": [4],
            "episode_index": 1,
            "boundary_transition": False,
            "boundary_terminal_env_token_len": 0,
            "boundary_next_initial_env_token_len": 0,
        },
    ]
    hindsight = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=1,
    )
    assert hindsight is None


def test_prepare_distill_payload_mixed_batch_partial_skip() -> None:
    class _PromptTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {
                "system:s0": [99],
                "user:u0<assistant>": [1, 2, 3],
                "system:s1": [88],
                "user:v0<assistant>": [4, 5, 6],
            }
            return list(mapping.get(text, []))

    class _PromptParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            out: list[str] = []
            for message in messages:
                role = message["role"]
                content = message.get("content", "")
                if role == "assistant":
                    out.append(f"{self.assistant_token}{content}")
                else:
                    out.append(f"{role}:{content}")
            text = "".join(out)
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = _PromptTokenizer()
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=_PromptParser())
    trainer.distill_settings.context_limit = 13
    trainer.distill_settings.teacher_context_attempts = 1
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [3],
                    "response": "first",
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 2,
                    "boundary_next_initial_env_token_len": 1,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 3, 4, 5, 6],
                    "completion_ids": [7],
                    "response": "retry",
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                },
            ]
        },
        {
            "chat_completions": [
                {"role": "system", "content": "s1"},
                {"role": "user", "content": "v0"},
                {"role": "assistant", "content": "a1"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [8, 9],
                    "completion_ids": [10],
                    "response": "first",
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 4,
                    "boundary_next_initial_env_token_len": 0,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [8, 9, 10, 11, 12, 13, 14],
                    "completion_ids": [15],
                    "response": "retry",
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                },
            ]
        },
        {"step_records": "malformed"},
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor(
                [
                    [0, 0, 99, 1, 2, 3],
                    [0, 0, 88, 4, 5, 6],
                    [0, 0, 77, 7, 8, 9],
                ],
                dtype=torch.long,
            ),
            "responses": torch.tensor(
                [
                    [11, 90, 12, 91, 13, 0],
                    [21, 80, 22, 81, 23, 0],
                    [31, 70, 32, 71, 33, 0],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 0],
                ],
                dtype=torch.long,
            ),
            "first_attempt_response_mask": torch.tensor(
                [
                    # last first-attempt index is 3 (prefix length 4), while sum(mask)=2
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                    [1, 0, 0, 1, 0, 0],
                ],
                dtype=torch.float32,
            ),
            "attention_mask": torch.tensor(
                [
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                ],
                dtype=torch.long,
            ),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert payload.total_samples == 3
    assert payload.kept_samples == 1
    assert payload.skipped_context_overflow == 1
    assert payload.distill_mask.shape[0] == 1
    assert metrics["distill/skipped_context_overflow"] == 1.0
    assert metrics["distill/skipped_hindsight_unavailable"] == 1.0
    assert metrics["distill/skipped_hindsight_step_records_not_list"] == 1.0
    assert metrics["distill/skipped_hindsight_system_strip_unavailable"] == 0.0
    assert metrics["distill/skipped_selective_gate"] == 0.0
    assert metrics["distill/skipped_attempt_incomplete"] == 0.0
    assert abs(metrics["distill/kept_ratio"] - (1.0 / 3.0)) < 1e-6
    assert payload.student_batch.batch["prompts"][0].tolist() == [99, 1, 2, 3]
    assert payload.student_batch.batch["responses"][0].tolist() == [11, 90, 12, 91]
    assert payload.teacher_batch.batch["prompts"][0].tolist() == [1, 2, 3, 4, 5, 1, 2, 3]
    assert payload.teacher_batch.batch["responses"][0].tolist() == [11, 90, 12, 91]
    assert payload.distill_mask[0].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert payload.kept_indices.tolist() == [0]


def test_prepare_distill_payload_filters_when_system_strip_unavailable() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer._latest_token_trajectories = [
        {
            # chat_completions missing -> cannot reconstruct prompt without system.
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [3],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 1,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 3, 4, 5],
                    "completion_ids": [6],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                },
            ]
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[9, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[11, 90, 12, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "first_attempt_response_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is None
    assert metrics["distill/skipped_context_overflow"] == 0.0
    assert metrics["distill/skipped_hindsight_unavailable"] == 1.0
    assert metrics["distill/skipped_hindsight_system_strip_unavailable"] == 1.0
    assert metrics["distill/skipped_hindsight_teacher_context_unavailable"] == 0.0
    assert metrics["distill/skipped_selective_gate"] == 0.0
    assert metrics["distill/skipped_attempt_incomplete"] == 0.0
    assert metrics["distill/kept_ratio"] == 0.0


def test_prepare_distill_payload_reports_teacher_context_unavailable_branch() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 2
    trainer._latest_token_trajectories = [
        {
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [3],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 1,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 3, 4, 5],
                    "completion_ids": [6],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                },
            ]
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[9, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[11, 90, 12, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "first_attempt_response_mask": torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float32),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 1, 0]], dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is None
    assert metrics["distill/skipped_hindsight_unavailable"] == 1.0
    assert metrics["distill/skipped_hindsight_teacher_context_unavailable"] == 1.0
    assert metrics["distill/skipped_hindsight_system_strip_unavailable"] == 0.0


def test_prepare_distill_payload_strips_system_from_student_prompt_in_denominator() -> None:
    class _TemplateTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(ch) + 1 for ch in text]

    class _TemplateParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    tokenizer = _TemplateTokenizer()
    parser = _TemplateParser()
    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = tokenizer
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=parser)
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1

    initial_messages = [
        {"role": "system", "content": "SYS0"},
        {"role": "system", "content": "SYS1"},
        {"role": "user", "content": "OBS0"},
    ]
    student_prompt_ids, _ = convert_messages_to_tokens_and_masks(
        initial_messages,
        tokenizer=tokenizer,
        parser=parser,
        contains_first_msg=True,
        contains_generation_msg=True,
    )
    completion_ids = [701]
    terminal_and_next_initial = [801, 802]

    trajectory = {
        "chat_completions": [
            {"role": "system", "content": "SYS0"},
            {"role": "system", "content": "SYS1"},
            {"role": "user", "content": "OBS0"},
            {"role": "assistant", "content": "A0"},
            {"role": "user", "content": "OBS1"},
        ],
        "step_records": [
            {
                "prompt_ids": student_prompt_ids,
                "completion_ids": completion_ids,
                "episode_index": 0,
                "boundary_transition": True,
                "boundary_terminal_env_token_len": 1,
                "boundary_next_initial_env_token_len": 1,
            },
            {
                "prompt_ids": student_prompt_ids + completion_ids + terminal_and_next_initial,
                "completion_ids": [901],
                "episode_index": 1,
                "boundary_transition": False,
                "boundary_terminal_env_token_len": 0,
                "boundary_next_initial_env_token_len": 0,
            },
        ],
    }
    trainer._latest_token_trajectories = [trajectory]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([student_prompt_ids], dtype=torch.long),
            "responses": torch.tensor([[55, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "first_attempt_response_mask": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "attention_mask": torch.ones((1, len(student_prompt_ids) + 2), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert metrics["distill/skipped_hindsight_unavailable"] == 0.0
    denominator_prompts = payload.teacher_batch.batch["prompts"]
    denominator_responses = payload.teacher_batch.batch["responses"]
    denominator_attention = payload.teacher_batch.batch["attention_mask"]
    denominator_response_len = int(denominator_responses.shape[1])
    denominator_prompt_mask = denominator_attention[0, :-denominator_response_len] > 0
    denominator_prompt_tokens = denominator_prompts[0][denominator_prompt_mask]

    hindsight_context = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=trajectory["step_records"],
        teacher_context_attempts=1,
    )
    assert hindsight_context is not None
    leading_system_messages = []
    for message in initial_messages:
        if message["role"] != "system":
            break
        leading_system_messages.append(message)
    system_prefix_ids, _ = convert_messages_to_tokens_and_masks(
        leading_system_messages,
        tokenizer=tokenizer,
        parser=parser,
        contains_first_msg=True,
        contains_generation_msg=False,
    )
    expected_stripped_prompt = torch.as_tensor(
        student_prompt_ids[len(system_prefix_ids) :],
        dtype=torch.long,
    )
    stripped_prompt = trainer._build_student_prompt_without_system(
        trajectory=trajectory,
        student_prompt_tokens=torch.as_tensor(student_prompt_ids, dtype=torch.long),
    )
    assert stripped_prompt is not None
    assert torch.equal(stripped_prompt, expected_stripped_prompt)
    expected_denominator = torch.cat(
        [hindsight_context.long(), expected_stripped_prompt],
        dim=0,
    )

    assert denominator_prompt_tokens.tolist() == expected_denominator.tolist()
    assert expected_stripped_prompt.numel() < len(student_prompt_ids)
    assert batch.batch["prompts"][0].tolist() == student_prompt_ids


def test_prepare_distill_payload_selective_retry_success_n2_uses_attempt2_sequence() -> None:
    class _PromptTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {
                "system:s0": [99],
                "user:u0<assistant>": [1, 2, 3],
            }
            return list(mapping.get(text, []))

    class _PromptParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = _PromptTokenizer()
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=_PromptParser())
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [10],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 2,
                    "episode_success": False,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 10, 20, 30, 31],
                    "completion_ids": [11],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 10, 20, 30, 31, 11, 40],
                    "completion_ids": [12],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                    "episode_done": True,
                },
            ],
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[99, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[55, 56, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 7), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert metrics["distill/skipped_selective_gate"] == 0.0
    assert metrics["distill/skipped_attempt_incomplete"] == 0.0
    assert payload.student_batch.batch["prompts"][0].tolist() == [99, 1, 2, 3]
    assert payload.student_batch.batch["responses"][0].tolist() == [11, 40, 12]
    assert payload.teacher_batch.batch["prompts"][0].tolist() == [1, 2, 10, 20, 1, 2, 3]
    assert payload.teacher_batch.batch["responses"][0].tolist() == [11, 40, 12]
    assert payload.distill_mask[0].tolist() == [1.0, 0.0, 1.0]


def test_prepare_distill_payload_selective_retry_success_n2_gate_failure_skips_sample() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 1,
                    "episode_success": True,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 3, 4],
                    "completion_ids": [5],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                    "episode_done": True,
                },
            ],
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[9, 1, 2]], dtype=torch.long),
            "responses": torch.tensor([[7, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is None
    assert metrics["distill/skipped_selective_gate"] == 1.0
    assert metrics["distill/skipped_attempt_incomplete"] == 0.0


def test_prepare_distill_payload_selective_retry_success_n2_requires_two_complete_attempts() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1],
                    "completion_ids": [2],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 1,
                    "episode_success": False,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 3, 4],
                    "completion_ids": [5],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                },
            ],
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[9, 1, 2]], dtype=torch.long),
            "responses": torch.tensor([[7, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is None
    assert metrics["distill/skipped_attempt_incomplete"] == 1.0
    assert metrics["distill/skipped_selective_gate"] == 0.0


def test_prepare_distill_payload_selective_retry_success_n2_keeps_reflection_retry_success() -> None:
    class _PromptTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {
                "system:s0": [99],
                "user:u0<assistant>": [1, 2, 3],
            }
            return list(mapping.get(text, []))

    class _PromptParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = _PromptTokenizer()
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=_PromptParser())
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [10],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": False,
                    "episode_done": True,
                },
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2, 10, 20],
                    "completion_ids": [11],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 1,
                    # Reflection boundary step loses the terminal success flag.
                    "episode_success": False,
                    "episode_done": False,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 10, 20, 11, 30],
                    "completion_ids": [12],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                    "episode_done": True,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 10, 20, 11, 30, 12, 40],
                    "completion_ids": [13],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    # Final reflection step also loses the original terminal flag.
                    "episode_success": False,
                    "episode_done": True,
                },
            ],
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[99, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[55, 56, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 7), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert metrics["distill/skipped_selective_gate"] == 0.0
    assert metrics["distill/skipped_attempt_incomplete"] == 0.0
    assert payload.kept_samples == 1


def test_prepare_distill_payload_selective_retry_success_n2_computes_student_old_log_probs_for_clip() -> None:
    class _PromptTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {
                "system:s0": [99],
                "user:u0<assistant>": [1, 2, 3],
            }
            return list(mapping.get(text, []))

    class _PromptParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = _PromptTokenizer()
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=_PromptParser())
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
    trainer.distill_settings.is_clip = 2.0
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [10],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 1,
                    "episode_success": False,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 10, 20, 30],
                    "completion_ids": [11],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                    "episode_done": True,
                },
            ],
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[99, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[55, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, _ = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 1
    assert payload.student_old_log_probs is not None
    assert torch.allclose(
        payload.student_old_log_probs,
        torch.tensor([[1.1]], dtype=torch.float32),
    )


def test_prepare_distill_payload_clip_pads_compact_batch_to_world_size() -> None:
    class _PromptTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {
                "system:s0": [99],
                "user:u0<assistant>": [1, 2, 3],
            }
            return list(mapping.get(text, []))

    class _PromptParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    class _ChunkAwareActorRolloutWG(_FakeActorRolloutWG):
        def __init__(self) -> None:
            super().__init__()
            self.world_size = 4
            self.last_log_prob_batch_size = 0

        def compute_log_prob(self, batch: _FakeDataProto) -> _FakeDataProto:
            self.compute_log_prob_calls += 1
            self.last_log_prob_batch_size = len(batch)
            assert len(batch) % self.world_size == 0
            responses = batch.batch["responses"].float()
            return _FakeDataProto({"old_log_probs": responses / 10.0})

    def _pad_fake_dataproto_to_world_size(*, batch: _FakeDataProto) -> _FakeDataProto:
        pad = (-len(batch)) % 4
        if pad == 0:
            return batch
        padded = {}
        for key, value in batch.batch.items():
            pad_rows = value[:1].repeat((pad,) + (1,) * (value.dim() - 1))
            padded[key] = torch.cat([value, pad_rows], dim=0)
        return _FakeDataProto(padded)

    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = _PromptTokenizer()
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=_PromptParser())
    trainer.actor_rollout_wg = _ChunkAwareActorRolloutWG()
    trainer._pad_dataproto_to_world_size = _pad_fake_dataproto_to_world_size  # type: ignore[method-assign]
    trainer.distill_settings.context_limit = 100000
    trainer.distill_settings.teacher_context_attempts = 1
    trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
    trainer.distill_settings.is_clip = 2.0
    trainer._latest_token_trajectories = [
        {
            "chat_completions": [
                {"role": "system", "content": "s0"},
                {"role": "user", "content": "u0"},
                {"role": "assistant", "content": "a0"},
            ],
            "step_records": [
                {
                    "episode_index": 0,
                    "prompt_ids": [1, 2],
                    "completion_ids": [10],
                    "boundary_transition": True,
                    "boundary_terminal_env_token_len": 1,
                    "boundary_next_initial_env_token_len": 1,
                    "episode_success": False,
                },
                {
                    "episode_index": 1,
                    "prompt_ids": [1, 2, 10, 20, 30],
                    "completion_ids": [11],
                    "boundary_transition": False,
                    "boundary_terminal_env_token_len": 0,
                    "boundary_next_initial_env_token_len": 0,
                    "episode_success": True,
                    "episode_done": True,
                },
            ],
        }
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor([[99, 1, 2, 3]], dtype=torch.long),
            "responses": torch.tensor([[55, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "attention_mask": torch.ones((1, 6), dtype=torch.long),
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, _ = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 1
    assert trainer.actor_rollout_wg.last_log_prob_batch_size == 4
    assert payload.student_old_log_probs is not None
    assert payload.student_old_log_probs.shape == (1, 1)
    assert torch.allclose(
        payload.student_old_log_probs,
        torch.tensor([[1.1]], dtype=torch.float32),
    )


def test_build_student_prompt_without_system_strips_exact_prefix_when_first_message_tokenization_differs() -> None:
    class _PrefixTokenizer:
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            mapping = {
                "system:FIRST:SYS": [10, 11],
                "user:NEXT:OBS<assistant>": [20, 21],
                "user:FIRST:OBS<assistant>": [30, 31],
            }
            if text not in mapping:
                raise KeyError(text)
            return list(mapping[text])

    class _PrefixParser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            out: list[str] = []
            for idx, message in enumerate(messages):
                role = message["role"]
                content = message.get("content", "")
                if role == "assistant":
                    out.append(f"{self.assistant_token}{content}")
                    continue
                position = "FIRST" if (is_first_msg and idx == 0) else "NEXT"
                out.append(f"{role}:{position}:{content}")
            text = "".join(out)
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    tokenizer = _PrefixTokenizer()
    parser = _PrefixParser()
    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = tokenizer
    trainer.agent_execution_engine = SimpleNamespace(chat_parser=parser)

    initial_messages = [
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": "OBS"},
    ]
    student_prompt_ids, _ = convert_messages_to_tokens_and_masks(
        initial_messages,
        tokenizer=tokenizer,
        parser=parser,
        contains_first_msg=True,
        contains_generation_msg=True,
    )
    trajectory = {
        "chat_completions": [
            {"role": "system", "content": "SYS"},
            {"role": "user", "content": "OBS"},
            {"role": "assistant", "content": "A0"},
        ]
    }

    student_prompt_tokens = torch.as_tensor(student_prompt_ids, dtype=torch.long)
    stripped_prompt = trainer._build_student_prompt_without_system(
        trajectory=trajectory,
        student_prompt_tokens=student_prompt_tokens,
    )

    system_prefix_ids, _ = convert_messages_to_tokens_and_masks(
        [initial_messages[0]],
        tokenizer=tokenizer,
        parser=parser,
        contains_first_msg=True,
        contains_generation_msg=False,
    )
    old_suffix_ids = torch.as_tensor(
        tokenizer.encode(
            parser.parse(
                initial_messages[1:],
                add_generation_prompt=True,
                is_first_msg=True,
            ),
            add_special_tokens=False,
        ),
        dtype=torch.long,
    )

    assert stripped_prompt is not None
    assert torch.equal(
        stripped_prompt,
        student_prompt_tokens[len(system_prefix_ids) :],
    )
    assert not torch.equal(stripped_prompt, old_suffix_ids)


def test_attach_distill_payload_to_batch_none_payload() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.min_distill_tokens = 3
    batch = _FakeDataProto(
        {
            "responses": torch.tensor([[1, 2, 3]], dtype=torch.long),
        }
    )
    metrics = trainer._attach_distill_payload_to_batch(batch=batch, payload=None)
    assert metrics["distill/sdpo_loss"] == 0.0
    assert metrics["distill/token_count"] == 0.0
    assert batch.meta_info["distill_enabled"] is False
    assert batch.meta_info["distill_use_grpo_loss"] is True
    assert "distill_mask" not in batch.batch


def test_attach_distill_payload_to_batch_kept_rows_and_mask_alignment() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.min_distill_tokens = 1
    batch = _FakeDataProto(
        {
            "responses": torch.tensor(
                [
                    [11, 90, 12, 91],
                    [21, 80, 22, 81],
                ],
                dtype=torch.long,
            ),
        }
    )
    payload = DistillPayload(
        student_batch=_FakeDataProto(
            {
                "prompts": torch.tensor([[9, 8]], dtype=torch.long),
                "responses": torch.tensor([[11, 90, 12, 91]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2, 3, 4, 5]], dtype=torch.long),
            }
        ),
        teacher_batch=_FakeDataProto(
            {
                "prompts": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
                "responses": torch.tensor([[11, 90, 12, 91]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6]], dtype=torch.long),
            }
        ),
        distill_mask=torch.tensor([[1.0, 0.0, 0.0, 1.0]]),
        kept_indices=torch.tensor([0], dtype=torch.long),
        skipped_context_overflow=0,
        skipped_hindsight_unavailable=0,
        skipped_selective_gate=0,
        skipped_attempt_incomplete=0,
        total_samples=2,
        kept_samples=1,
    )

    metrics = trainer._attach_distill_payload_to_batch(batch=batch, payload=payload)
    assert metrics["distill/token_count"] == 2.0
    assert metrics["distill/sdpo_loss"] == 0.0
    assert batch.meta_info["distill_enabled"] is True
    assert batch.meta_info["distill_use_grpo_loss"] is True
    assert batch.batch["distill_mask"].shape == (2, 4)
    assert batch.batch["distill_mask"][0].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert batch.batch["distill_mask"][1].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert batch.batch["distill_student_responses"][0].tolist() == [11, 90, 12, 91]
    assert batch.batch["distill_student_responses"][1].tolist() == [0, 0, 0, 0]
    assert batch.batch["distill_teacher_responses"][0].tolist() == [11, 90, 12, 91]
    assert batch.batch["distill_teacher_responses"][1].tolist() == [0, 0, 0, 0]


def test_attach_distill_payload_to_batch_respects_min_token_gate() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.min_distill_tokens = 3
    batch = _FakeDataProto({"responses": torch.tensor([[1, 2]], dtype=torch.long)})
    payload = DistillPayload(
        student_batch=_FakeDataProto(
            {
                "prompts": torch.tensor([[7, 8]], dtype=torch.long),
                "responses": torch.tensor([[5, 6]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            }
        ),
        teacher_batch=_FakeDataProto(
            {
                "prompts": torch.tensor([[0, 1]], dtype=torch.long),
                "responses": torch.tensor([[5, 6]], dtype=torch.long),
                "response_mask": torch.tensor([[1, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
                "position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
            }
        ),
        distill_mask=torch.tensor([[1.0, 1.0]]),
        kept_indices=torch.tensor([0], dtype=torch.long),
        skipped_context_overflow=0,
        skipped_hindsight_unavailable=0,
        skipped_selective_gate=0,
        skipped_attempt_incomplete=0,
        total_samples=1,
        kept_samples=1,
    )

    metrics = trainer._attach_distill_payload_to_batch(batch=batch, payload=payload)
    assert metrics["distill/token_count"] == 2.0
    assert batch.meta_info["distill_enabled"] is False
    assert batch.meta_info["distill_use_grpo_loss"] is True
    assert "distill_mask" not in batch.batch


def test_set_distill_meta_reflects_pure_sdpo_mode() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.mode = "sdpo_pure"
    trainer.distill_settings.use_grpo_loss = False
    batch = _FakeDataProto({"responses": torch.tensor([[1, 2]], dtype=torch.long)})
    trainer._set_distill_meta(batch=batch, enabled_for_step=False)
    assert batch.meta_info["distill_use_grpo_loss"] is False


def test_teacher_sync_hooks_init_and_post_update() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer.global_steps = 6
    trainer.distill_settings.teacher_regularization = "ema"
    trainer.distill_settings.teacher_update_rate = 0.25
    trainer.distill_settings.teacher_update_interval = 3

    init_applied = trainer._initialize_teacher_snapshot_if_needed(timing_raw={})
    assert init_applied == 1.0
    assert len(trainer.actor_rollout_wg.sync_calls) == 1
    assert trainer.actor_rollout_wg.sync_calls[0]["mode"] == "hard"

    post_applied = trainer._maybe_sync_teacher_after_actor_update(timing_raw={})
    assert post_applied == 1.0
    assert len(trainer.actor_rollout_wg.sync_calls) == 2
    assert trainer.actor_rollout_wg.sync_calls[1]["mode"] == "ema"
    assert trainer.actor_rollout_wg.sync_calls[1]["update_rate"] == 0.25


def test_teacher_sync_hooks_every_n_steps_interval() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer.distill_settings.teacher_regularization = "every_n_steps"
    trainer.distill_settings.teacher_update_interval = 4

    trainer.global_steps = 5
    not_applied = trainer._maybe_sync_teacher_after_actor_update(timing_raw={})
    assert not_applied == 0.0
    assert len(trainer.actor_rollout_wg.sync_calls) == 0

    trainer.global_steps = 8
    applied = trainer._maybe_sync_teacher_after_actor_update(timing_raw={})
    assert applied == 1.0
    assert len(trainer.actor_rollout_wg.sync_calls) == 1
    assert trainer.actor_rollout_wg.sync_calls[0]["mode"] == "hard"


def test_prepare_distill_payload_reflection_frozenlake_n2_den_num_sanity() -> None:
    import pytest
    from rllm.agents.agent import Action, Step, Trajectory
    from rllm.engine.agent_execution_engine import AgentExecutionEngine
    from rllm.engine.rollout.rollout_engine import ModelOutput

    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter
    from envs.multi_episode_env import MultiEpisodeEnv
    from prompts.prompt import reflection_prompt_2

    pytest.importorskip("gymnasium")

    class _Tokenizer:
        pad_token_id = 0
        name_or_path = "unit-test-frozenlake"
        assistant_token = "<assistant>"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            tokenize: bool = False,
        ) -> str | list[int]:
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            if tokenize:
                return self.encode(text, add_special_tokens=False)
            return text

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [max(1, ord(ch) % 31) for ch in text]

    class _Parser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            out: list[str] = []
            for message in messages:
                role = message["role"]
                content = message.get("content", "")
                if role == "assistant":
                    out.append(f"{self.assistant_token}{content}")
                else:
                    out.append(f"{role}:{content}")
            text = "".join(out)
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    class _BoundaryAwareAgent:
        def __init__(self) -> None:
            self._trajectory = Trajectory()
            self._messages: list[dict[str, str]] = []
            self.reset()

        @property
        def chat_completions(self) -> list[dict[str, str]]:
            return list(self._messages)

        @property
        def trajectory(self) -> Trajectory:
            return self._trajectory

        def reset(self) -> None:
            self._trajectory = Trajectory()
            self._messages = [{"role": "system", "content": "test"}]

        def update_from_env(
            self,
            observation: Any,
            reward: float,
            done: bool,
            info: dict[str, Any],
            **kwargs: Any,
        ) -> None:
            del reward, done, kwargs
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
                if not appended_any:
                    self._messages.append({"role": "user", "content": str(observation)})
            else:
                self._messages.append({"role": "user", "content": str(observation)})

        def update_from_model(self, response: str, **kwargs: Any) -> Action:
            del kwargs
            self._messages.append({"role": "assistant", "content": response})
            step = Step(
                chat_completions=list(self._messages),
                action=Action(action=response),
                model_response=response,
                info={},
            )
            self._trajectory.steps.append(step)
            return Action(action=response)

        def get_current_state(self) -> Step:
            return self._trajectory.steps[-1]

    from concurrent.futures import ThreadPoolExecutor

    env = MultiEpisodeEnv(
        inner_env_class=FrozenLakeEnvAdapter,
        inner_env_kwargs={
            "env_id": "frozenlake",
            "env_kwargs": {
                "desc": ["SF", "FG"],
                "is_slippery": False,
                "max_turns": 2,
                "seed": 21,
            },
        },
        total_step_cap=4,  # 2 * max_turns
        success_reward=1.0,
        episode_header="New episode begins.",
        enable_reflection=True,
        reflection_prompt=reflection_prompt_2,
    )
    engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
    engine.agents = [_BoundaryAwareAgent()]
    engine.envs = [env]
    engine.executor = ThreadPoolExecutor(max_workers=1)
    engine.max_steps = 8
    engine.max_response_length = 4096
    engine.max_prompt_length = 4096
    engine.enforce_max_prompt_length = False
    engine.engine_name = "verl"
    engine.tokenizer = _Tokenizer()
    engine.chat_parser = _Parser()
    engine.config = SimpleNamespace(rllm=SimpleNamespace(filter_token_mismatch=False))
    engine.overlong_filter = False
    engine.trajectory_timeout = 30
    engine.gamma = 0.9
    engine.sampling_params = {}
    engine.disable_thinking = False

    turn = {"k": 0, "action_count": 0}

    async def _fake_get_model_response(
        prompt_messages: list[dict[str, str]],
        application_id: str,
        **kwargs: Any,
    ) -> ModelOutput:
        del application_id
        prompt_ids = kwargs.get("accumulated_prompt_ids")
        if prompt_ids is None:
            prompt_text = engine.chat_parser.parse(
                prompt_messages,
                add_generation_prompt=True,
                is_first_msg=True,
            )
            prompt_ids = engine.tokenizer.encode(prompt_text, add_special_tokens=False)
        else:
            prompt_ids = list(prompt_ids)

        last_user = ""
        for message in reversed(prompt_messages):
            if message.get("role") == "user":
                last_user = str(message.get("content", ""))
                break

        if "[Reflection" in last_user or "reflection" in last_user.lower():
            text = f"reflection-{turn['k']}"
        else:
            text = r"\boxed{right}" if (turn["action_count"] % 2 == 0) else r"\boxed{down}"
            turn["action_count"] += 1

        completion_ids = [700 + turn["k"]]
        turn["k"] += 1
        return ModelOutput(
            text=text,
            prompt_ids=prompt_ids,
            completion_ids=completion_ids,
            logprobs=[-0.1],
        )

    engine.get_model_response = _fake_get_model_response  # type: ignore[method-assign]

    try:
        token_result = asyncio.run(
            engine.run_agent_trajectory_async(
                idx=0,
                application_id="app",
                mode="Token",
                meta_info={},
            )
        )
    finally:
        engine.executor.shutdown(wait=False, cancel_futures=True)
        env.close()

    step_records = token_result["step_records"]
    assert len([step for step in step_records if r"\boxed{" in str(step["response"])]) == 4

    trainer = _build_trainer_for_unit_tests()
    trainer.tokenizer = engine.tokenizer
    trainer.distill_settings.teacher_context_attempts = 2
    trainer.distill_settings.context_limit = 100000
    trainer.agent_execution_engine = engine
    trainer._latest_token_trajectories = [token_result]

    prompt_tokens = torch.as_tensor(token_result["prompt_tokens"], dtype=torch.long)
    response_tokens = torch.as_tensor(token_result["response_tokens"], dtype=torch.long)
    response_mask = torch.as_tensor(token_result["response_masks"], dtype=torch.long)
    first_attempt_response_mask = torch.as_tensor(
        token_result["first_attempt_response_mask"],
        dtype=torch.float32,
    )
    attention_mask = torch.ones(
        (1, prompt_tokens.numel() + response_tokens.numel()),
        dtype=torch.long,
    )
    batch = _FakeDataProto(
        {
            "prompts": prompt_tokens.unsqueeze(0),
            "responses": response_tokens.unsqueeze(0),
            "response_mask": response_mask.unsqueeze(0),
            "first_attempt_response_mask": first_attempt_response_mask.unsqueeze(0),
            "attention_mask": attention_mask,
            "distill_traj_idx": torch.tensor([0], dtype=torch.long),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert metrics["distill/skipped_hindsight_unavailable"] == 0.0
    assert metrics["distill/skipped_context_overflow"] == 0.0
    assert metrics["distill/skipped_selective_gate"] == 0.0
    assert metrics["distill/skipped_attempt_incomplete"] == 0.0
    assert payload.kept_samples == 1

    numerator_prompt_tokens = prompt_tokens
    extracted = extract_first_attempt_prefix(
        response_tokens=response_tokens,
        response_mask=response_mask.float(),
        first_attempt_response_mask=first_attempt_response_mask,
    )
    assert extracted is not None
    numerator_response_tokens, numerator_response_mask, _ = extracted

    denominator_prompts = payload.teacher_batch.batch["prompts"]
    denominator_responses = payload.teacher_batch.batch["responses"]
    denominator_response_mask = payload.teacher_batch.batch["response_mask"]
    denominator_attention = payload.teacher_batch.batch["attention_mask"]
    denominator_response_len = int(denominator_responses.shape[1])
    denominator_prompt_mask = denominator_attention[0, :-denominator_response_len] > 0
    denominator_prompt_tokens = denominator_prompts[0][denominator_prompt_mask]
    denominator_response_tokens = denominator_responses[0]
    denominator_response_mask_tokens = denominator_response_mask[0]

    hindsight_context = build_hindsight_prompt_tokens_first_n_complete_attempts(
        step_records=step_records,
        teacher_context_attempts=2,
    )
    assert hindsight_context is not None
    initial_messages = extract_initial_messages_before_first_assistant(
        token_result["chat_completions"]
    )
    assert initial_messages is not None
    assert len(initial_messages) >= 2
    assert initial_messages[0]["role"] == "system"
    leading_system_messages = []
    for message in initial_messages:
        if message["role"] != "system":
            break
        leading_system_messages.append(message)
    system_prefix_ids, _ = convert_messages_to_tokens_and_masks(
        leading_system_messages,
        tokenizer=engine.tokenizer,
        parser=engine.chat_parser,
        contains_first_msg=True,
        contains_generation_msg=False,
    )
    system_prefix_tokens = torch.as_tensor(
        system_prefix_ids,
        dtype=torch.long,
    )
    assert system_prefix_tokens.numel() < numerator_prompt_tokens.numel()
    assert torch.equal(
        numerator_prompt_tokens[: system_prefix_tokens.numel()],
        system_prefix_tokens,
    )
    prompt_without_system_ids = trainer._build_student_prompt_without_system(
        trajectory=token_result,
        student_prompt_tokens=numerator_prompt_tokens.long(),
    )
    assert prompt_without_system_ids is not None
    assert prompt_without_system_ids.numel() < numerator_prompt_tokens.numel()
    assert torch.equal(
        prompt_without_system_ids,
        numerator_prompt_tokens[system_prefix_tokens.numel() :],
    )
    expected_denominator_prompt = torch.cat(
        [hindsight_context.long(), prompt_without_system_ids.long()],
        dim=0,
    )

    # Distillation sanity:
    # den_prompt = concat(c_N, num_prompt_without_system), den_response = numerator first-attempt prefix.
    assert denominator_prompt_tokens.tolist() == expected_denominator_prompt.tolist()
    assert denominator_response_tokens.tolist() == numerator_response_tokens.tolist()
    assert denominator_response_mask_tokens.tolist() == numerator_response_mask.tolist()

    # c_N should start from the first attempt's initial prompt+response history.
    first_turn_prefix = list(step_records[0]["prompt_ids"]) + list(step_records[0]["completion_ids"])
    assert hindsight_context[: len(first_turn_prefix)].tolist() == first_turn_prefix


def test_selective_retry_success_n2_rollout_consistent_with_and_without_reflection() -> None:
    import pytest
    from concurrent.futures import ThreadPoolExecutor

    from rllm.agents.agent import Action, Step, Trajectory
    from rllm.engine.agent_execution_engine import AgentExecutionEngine
    from rllm.engine.rollout.rollout_engine import ModelOutput

    from envs.frozenlake_env_adapter import FrozenLakeEnvAdapter
    from envs.multi_episode_env import MultiEpisodeEnv
    from prompts.prompt import reflection_prompt_2

    pytest.importorskip("gymnasium")

    class _Tokenizer:
        pad_token_id = 0
        name_or_path = "unit-test-frozenlake"
        assistant_token = "<assistant>"

        def apply_chat_template(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            tokenize: bool = False,
        ) -> str | list[int]:
            text = "".join(
                (
                    f"{message['role']}:{message.get('content', '')}"
                    if message["role"] != "assistant"
                    else f"{self.assistant_token}{message.get('content', '')}"
                )
                for message in messages
            )
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            if tokenize:
                return self.encode(text, add_special_tokens=False)
            return text

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [max(1, ord(ch) % 31) for ch in text]

    class _Parser:
        assistant_token = "<assistant>"

        def parse(
            self,
            messages: list[dict[str, str]],
            add_generation_prompt: bool = False,
            is_first_msg: bool = False,
        ) -> str:
            del is_first_msg
            out: list[str] = []
            for message in messages:
                role = message["role"]
                content = message.get("content", "")
                if role == "assistant":
                    out.append(f"{self.assistant_token}{content}")
                else:
                    out.append(f"{role}:{content}")
            text = "".join(out)
            if add_generation_prompt:
                text = f"{text}{self.assistant_token}"
            return text

    class _BoundaryAwareAgent:
        def __init__(self) -> None:
            self._trajectory = Trajectory()
            self._messages: list[dict[str, str]] = []
            self.reset()

        @property
        def chat_completions(self) -> list[dict[str, str]]:
            return list(self._messages)

        @property
        def trajectory(self) -> Trajectory:
            return self._trajectory

        def reset(self) -> None:
            self._trajectory = Trajectory()
            self._messages = [{"role": "system", "content": "test"}]

        def update_from_env(
            self,
            observation: Any,
            reward: float,
            done: bool,
            info: dict[str, Any],
            **kwargs: Any,
        ) -> None:
            del reward, done, kwargs
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
                if not appended_any:
                    self._messages.append({"role": "user", "content": str(observation)})
            else:
                self._messages.append({"role": "user", "content": str(observation)})

        def update_from_model(self, response: str, **kwargs: Any) -> Action:
            del kwargs
            self._messages.append({"role": "assistant", "content": response})
            step = Step(
                chat_completions=list(self._messages),
                action=Action(action=response),
                model_response=response,
                info={},
            )
            self._trajectory.steps.append(step)
            return Action(action=response)

        def get_current_state(self) -> Step:
            return self._trajectory.steps[-1]

    def _simulate(enable_reflection: bool) -> tuple[dict[str, Any], dict[str, Any], DistillPayload, dict[str, float]]:
        env = MultiEpisodeEnv(
            inner_env_class=FrozenLakeEnvAdapter,
            inner_env_kwargs={
                "env_id": "frozenlake",
                "env_kwargs": {
                    "desc": ["SF", "FG"],
                    "is_slippery": False,
                    "max_turns": 2,
                    "seed": 21,
                },
            },
            total_step_cap=4,
            success_reward=1.0,
            episode_header="New episode begins.",
            enable_reflection=enable_reflection,
            reflection_prompt=reflection_prompt_2,
        )
        engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
        engine.agents = [_BoundaryAwareAgent()]
        engine.envs = [env]
        engine.executor = ThreadPoolExecutor(max_workers=1)
        engine.max_steps = 8
        engine.max_response_length = 4096
        engine.max_prompt_length = 4096
        engine.enforce_max_prompt_length = False
        engine.engine_name = "verl"
        engine.tokenizer = _Tokenizer()
        engine.chat_parser = _Parser()
        engine.config = SimpleNamespace(rllm=SimpleNamespace(filter_token_mismatch=False))
        engine.overlong_filter = False
        engine.trajectory_timeout = 30
        engine.gamma = 0.9
        engine.sampling_params = {}
        engine.disable_thinking = False

        turn = {"k": 0, "action_count": 0}
        scripted_actions = [
            r"\boxed{right}",
            r"\boxed{right}",
            r"\boxed{right}",
            r"\boxed{down}",
        ]

        async def _fake_get_model_response(
            prompt_messages: list[dict[str, str]],
            application_id: str,
            **kwargs: Any,
        ) -> ModelOutput:
            del application_id
            prompt_ids = kwargs.get("accumulated_prompt_ids")
            if prompt_ids is None:
                prompt_text = engine.chat_parser.parse(
                    prompt_messages,
                    add_generation_prompt=True,
                    is_first_msg=True,
                )
                prompt_ids = engine.tokenizer.encode(prompt_text, add_special_tokens=False)
            else:
                prompt_ids = list(prompt_ids)

            last_user = ""
            for message in reversed(prompt_messages):
                if message.get("role") == "user":
                    last_user = str(message.get("content", ""))
                    break

            if "[Reflection" in last_user or "reflection" in last_user.lower():
                text = f"reflection-{turn['k']}"
            else:
                text = scripted_actions[turn["action_count"]]
                turn["action_count"] += 1

            completion_ids = [700 + turn["k"]]
            turn["k"] += 1
            return ModelOutput(
                text=text,
                prompt_ids=prompt_ids,
                completion_ids=completion_ids,
                logprobs=[-0.1],
            )

        engine.get_model_response = _fake_get_model_response  # type: ignore[method-assign]

        try:
            token_result = asyncio.run(
                engine.run_agent_trajectory_async(
                    idx=0,
                    application_id="app",
                    mode="Token",
                    meta_info={},
                )
            )
            env_metrics = env.get_metrics()
        finally:
            engine.executor.shutdown(wait=False, cancel_futures=True)
            env.close()

        trainer = _build_trainer_for_unit_tests()
        trainer.tokenizer = engine.tokenizer
        trainer.distill_settings.context_limit = 100000
        trainer.distill_settings.teacher_context_attempts = 1
        trainer.distill_settings.trajectory_selection = "selective_retry_success_n2"
        trainer.agent_execution_engine = engine
        trainer._latest_token_trajectories = [token_result]

        prompt_tokens = torch.as_tensor(token_result["prompt_tokens"], dtype=torch.long)
        response_tokens = torch.as_tensor(token_result["response_tokens"], dtype=torch.long)
        response_mask = torch.as_tensor(token_result["response_masks"], dtype=torch.long)
        attention_mask = torch.ones(
            (1, prompt_tokens.numel() + response_tokens.numel()),
            dtype=torch.long,
        )
        batch = _FakeDataProto(
            {
                "prompts": prompt_tokens.unsqueeze(0),
                "responses": response_tokens.unsqueeze(0),
                "response_mask": response_mask.unsqueeze(0),
                "attention_mask": attention_mask,
                "distill_traj_idx": torch.tensor([0], dtype=torch.long),
            }
        )

        payload, metrics = trainer._prepare_distill_payload(batch)
        assert payload is not None
        return token_result, env_metrics, payload, metrics

    for enable_reflection in (False, True):
        token_result, env_metrics, payload, metrics = _simulate(enable_reflection)
        complete_attempts = collect_complete_attempt_summaries(token_result["step_records"])
        assert complete_attempts is not None
        assert [attempt.success for attempt in complete_attempts[:2]] == [False, True]
        assert env_metrics["episode_1/success_rate"] == 0.0
        assert env_metrics["episode_2/success_rate"] == 1.0
        assert metrics["distill/skipped_selective_gate"] == 0.0
        assert metrics["distill/skipped_attempt_incomplete"] == 0.0
        assert payload.kept_samples == 1
