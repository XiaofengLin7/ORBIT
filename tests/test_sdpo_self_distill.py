from __future__ import annotations

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

from trainers.sdpo_self_distill_trainer import (
    DistillPayload,
    DistillSettings,
    JointSDPOSelfDistillTrainer,
    build_hindsight_prompt_tokens_first_n_complete_attempts,
    compute_sdpo_advantages,
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
        lambda_coef=0.1,
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


def test_sdpo_advantages_are_detached_and_masked() -> None:
    numerator = torch.tensor([[0.4, 0.9, -0.1]], requires_grad=True)
    denominator = torch.tensor([[0.1, 0.2, 0.3]], requires_grad=True)
    mask = torch.tensor([[1.0, 0.0, 1.0]])

    advantages, stats = compute_sdpo_advantages(
        numerator_log_probs=numerator,
        denominator_log_probs=denominator,
        distill_mask=mask,
        lambda_coef=0.5,
    )

    expected = 0.5 * (numerator.detach() - denominator.detach()) * mask
    assert torch.allclose(advantages, expected)
    assert advantages.requires_grad is False
    assert stats["distill/token_count"] == 2.0


def test_prepare_distill_payload_mixed_batch_partial_skip() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.distill_settings.context_limit = 13
    trainer.distill_settings.teacher_context_attempts = 1
    trainer._latest_token_trajectories = [
        {
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
                    [0, 0, 0, 1, 2, 3],
                    [0, 0, 0, 4, 5, 6],
                    [0, 0, 0, 7, 8, 9],
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
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0],
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
    assert abs(metrics["distill/kept_ratio"] - (1.0 / 3.0)) < 1e-6
    assert payload.denominator_batch.batch["prompts"][0].tolist() == [1, 2, 3, 4, 5, 1, 2, 3]
    assert payload.denominator_batch.batch["responses"][0].tolist() == [11, 90, 12, 91]
    assert payload.distill_mask[0].tolist() == [1.0, 0.0, 0.0, 1.0]
    assert payload.kept_indices.tolist() == [0]


def test_compute_distill_bonus_requires_nonempty_payload() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer._pad_distill_inputs_to_world_size = (
        lambda denominator_batch, distill_mask, world_size: (
            denominator_batch,
            distill_mask,
        )
    )
    batch = _FakeDataProto(
        {
            "advantages": torch.zeros((1, 2)),
            "old_log_probs": torch.tensor([[1.0, 2.0]]),
        }
    )

    skipped_bonus, skipped = trainer._compute_distill_bonus(batch=batch, payload=None, timing_raw={})
    assert torch.allclose(skipped_bonus, torch.zeros((1, 2)))
    assert skipped["distill/skipped_batches"] == 1.0

    payload = DistillPayload(
        denominator_batch=_FakeDataProto({"responses": torch.tensor([[8.0, 15.0]])}),
        distill_mask=torch.tensor([[1.0, 1.0]]),
        kept_indices=torch.tensor([0], dtype=torch.long),
        skipped_context_overflow=0,
        total_samples=1,
        kept_samples=1,
    )
    kept_bonus, kept = trainer._compute_distill_bonus(batch=batch, payload=payload, timing_raw={})
    assert torch.allclose(kept_bonus, torch.tensor([[0.02, 0.05]]), atol=1e-6)
    assert kept["distill/skipped_batches"] == 0.0
    assert kept["distill/token_count"] == 2.0
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 1
    assert trainer.actor_rollout_wg.compute_teacher_log_prob_calls == 0


def test_compute_distill_bonus_uses_teacher_logprob_when_regularized() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer.distill_settings.teacher_regularization = "ema"
    trainer._pad_distill_inputs_to_world_size = (
        lambda denominator_batch, distill_mask, world_size: (
            denominator_batch,
            distill_mask,
        )
    )
    batch = _FakeDataProto(
        {
            "advantages": torch.zeros((1, 2)),
            "old_log_probs": torch.tensor([[1.0, 2.0]]),
        }
    )
    payload = DistillPayload(
        denominator_batch=_FakeDataProto({"responses": torch.tensor([[8.0, 15.0]])}),
        distill_mask=torch.tensor([[1.0, 1.0]]),
        kept_indices=torch.tensor([0], dtype=torch.long),
        skipped_context_overflow=0,
        total_samples=1,
        kept_samples=1,
    )
    bonus, metrics = trainer._compute_distill_bonus(batch=batch, payload=payload, timing_raw={})
    assert torch.allclose(bonus, torch.tensor([[0.06, 0.125]]), atol=1e-6)
    assert metrics["distill/skipped_batches"] == 0.0
    assert trainer.actor_rollout_wg.compute_log_prob_calls == 0
    assert trainer.actor_rollout_wg.compute_teacher_log_prob_calls == 1


def test_compute_distill_bonus_zero_masks_padded_rows() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer._pad_distill_inputs_to_world_size = (
        lambda denominator_batch, distill_mask, world_size: (
            _FakeDataProto(
                {
                    "responses": torch.tensor(
                        [
                            [10.0, 20.0],
                            [100.0, 200.0],
                        ]
                    )
                }
            ),
            torch.tensor(
                [
                    [1.0, 1.0],
                    [0.0, 0.0],
                ]
            ),
        )
    )
    batch = _FakeDataProto(
        {
            "advantages": torch.zeros((2, 2)),
            "old_log_probs": torch.tensor(
                [
                    [1.0, 2.0],
                    [10.0, 20.0],
                ]
            ),
        }
    )

    payload = DistillPayload(
        denominator_batch=_FakeDataProto({"responses": torch.tensor([[8.0, 15.0]])}),
        distill_mask=torch.tensor([[1.0, 1.0]]),
        kept_indices=torch.tensor([0], dtype=torch.long),
        skipped_context_overflow=0,
        total_samples=1,
        kept_samples=1,
    )
    bonus, metrics = trainer._compute_distill_bonus(batch=batch, payload=payload, timing_raw={})

    assert metrics["distill/skipped_batches"] == 0.0
    assert metrics["distill/token_count"] == 2.0
    assert torch.allclose(bonus[1], torch.zeros(2))


def test_compute_distill_bonus_respects_min_token_gate() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer.distill_settings.min_distill_tokens = 3
    trainer._pad_distill_inputs_to_world_size = (
        lambda denominator_batch, distill_mask, world_size: (
            denominator_batch,
            distill_mask,
        )
    )
    batch = _FakeDataProto(
        {
            "advantages": torch.zeros((1, 2)),
            "old_log_probs": torch.tensor([[1.0, 2.0]]),
        }
    )
    payload = DistillPayload(
        denominator_batch=_FakeDataProto({"responses": torch.tensor([[8.0, 15.0]])}),
        distill_mask=torch.tensor([[1.0, 1.0]]),
        kept_indices=torch.tensor([0], dtype=torch.long),
        skipped_context_overflow=0,
        total_samples=1,
        kept_samples=1,
    )

    bonus, metrics = trainer._compute_distill_bonus(batch=batch, payload=payload, timing_raw={})
    assert torch.allclose(bonus, torch.zeros((1, 2)))
    assert metrics["distill/skipped_batches"] == 1.0
    assert metrics["distill/token_count"] == 2.0


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
