from __future__ import annotations

from dataclasses import dataclass
import importlib.machinery
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys
import types

import torch

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
    compute_sdpo_advantages,
    should_skip_context_overflow,
)


class _DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text:
            return []
        if "fb_over" in text:
            return [1, 2]
        if "fb_eq" in text:
            return [1]
        if "fb_keep" in text:
            return [1]
        return [1]


class _FakeDataProto:
    def __init__(self, batch: dict[str, torch.Tensor]) -> None:
        self.batch = batch
        self.meta_info: dict[str, Any] = {}

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

    def compute_log_prob(self, batch: _FakeDataProto) -> _FakeDataProto:
        responses = batch.batch["responses"].float()
        return _FakeDataProto({"old_log_probs": responses / 10.0})

    def update_actor(self, batch: _FakeDataProto) -> _FakeActorOutput:
        self.update_calls += 1
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
    )
    trainer._latest_token_trajectories = []
    return trainer


def test_context_overflow_guard_gt_and_eq() -> None:
    assert should_skip_context_overflow(3, 5, 7) is True
    assert should_skip_context_overflow(3, 4, 7) is False


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
    trainer._latest_token_trajectories = [
        {
            "step_records": [
                {"episode_index": 0, "prompt_ids": [1, 2, 3, 4], "response": "first"},
                {"episode_index": 1, "prompt_ids": [1], "response": "fb_over"},
            ]
        },
        {
            "step_records": [
                {"episode_index": 0, "prompt_ids": [1, 2, 3], "response": "first"},
                {"episode_index": 1, "prompt_ids": [1], "response": "fb_eq"},
            ]
        },
        {
            "step_records": [
                {"episode_index": 0, "prompt_ids": [1, 2], "response": "first"},
                {"episode_index": 1, "prompt_ids": [1], "response": "fb_keep"},
            ]
        },
    ]

    batch = _FakeDataProto(
        {
            "prompts": torch.tensor(
                [
                    [0, 0, 0, 0, 1, 2],
                    [0, 0, 0, 0, 2, 3],
                    [0, 0, 0, 0, 3, 4],
                ],
                dtype=torch.long,
            ),
            "responses": torch.tensor(
                [
                    [11, 12, 0],
                    [21, 22, 0],
                    [31, 32, 0],
                ],
                dtype=torch.long,
            ),
            "response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                dtype=torch.long,
            ),
            "first_attempt_response_mask": torch.tensor(
                [
                    [1, 1, 0],
                    [1, 1, 0],
                    [1, 1, 0],
                ],
                dtype=torch.float32,
            ),
            "attention_mask": torch.tensor(
                [
                    [0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0],
                    [0, 0, 0, 0, 1, 1, 1, 1, 0],
                ],
                dtype=torch.long,
            ),
        }
    )

    payload, metrics = trainer._prepare_distill_payload(batch)

    assert payload is not None
    assert payload.total_samples == 3
    assert payload.kept_samples == 2
    assert payload.skipped_context_overflow == 1
    assert payload.distill_mask.shape[0] == 2
    assert metrics["distill/skipped_context_overflow"] == 1.0
    assert abs(metrics["distill/kept_ratio"] - (2.0 / 3.0)) < 1e-6


def test_run_distill_update_only_runs_on_nonempty_payload() -> None:
    trainer = _build_trainer_for_unit_tests()
    trainer.actor_rollout_wg = _FakeActorRolloutWG()
    trainer.use_reference_policy = False
    trainer.config = SimpleNamespace(
        actor_rollout_ref=SimpleNamespace(actor=SimpleNamespace(use_kl_loss=False)),
    )
    trainer._pad_to_world_size = lambda batch, world_size: batch

    skipped = trainer._run_distill_update(payload=None, timing_raw={})
    assert skipped["distill/skipped_batches"] == 1.0
    assert trainer.actor_rollout_wg.update_calls == 0

    payload = DistillPayload(
        numerator_batch=_FakeDataProto({"responses": torch.tensor([[10.0, 20.0]])}),
        denominator_batch=_FakeDataProto({"responses": torch.tensor([[8.0, 15.0]])}),
        distill_mask=torch.tensor([[1.0, 1.0]]),
        skipped_context_overflow=0,
        total_samples=1,
        kept_samples=1,
    )
    kept = trainer._run_distill_update(payload=payload, timing_raw={})
    assert kept["distill/skipped_batches"] == 0.0
    assert trainer.actor_rollout_wg.update_calls == 1
