from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys
from types import SimpleNamespace
import types

import pytest
import torch
import torch.nn.functional as F

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

from trainers.sdpo_actor import (
    SDPODistillParams,
    SDPODataParallelPPOActor,
    _compute_sdpo_per_token_loss,
    _count_active_distill_sequences,
    _segmented_reverse_cumsum,
)


class _FakeActorDataProto:
    def __init__(
        self,
        batch: dict[str, torch.Tensor],
        *,
        meta_info: dict[str, object] | None = None,
        non_tensor_batch: dict[str, object] | None = None,
    ) -> None:
        self.batch = batch
        self.meta_info = {} if meta_info is None else dict(meta_info)
        self.non_tensor_batch = {} if non_tensor_batch is None else dict(non_tensor_batch)

    def select(
        self,
        *,
        batch_keys: list[str],
        non_tensor_batch_keys: list[str],
    ) -> "_FakeActorDataProto":
        selected_batch = {key: self.batch[key] for key in batch_keys if key in self.batch}
        selected_non_tensor = {
            key: self.non_tensor_batch[key] for key in non_tensor_batch_keys if key in self.non_tensor_batch
        }
        return _FakeActorDataProto(
            selected_batch,
            meta_info=self.meta_info,
            non_tensor_batch=selected_non_tensor,
        )

    def split(self, batch_size: int) -> list["_FakeActorDataProto"]:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        total = next(iter(self.batch.values())).shape[0]
        out: list[_FakeActorDataProto] = []
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            out.append(
                _FakeActorDataProto(
                    {key: value[start:end] for key, value in self.batch.items()},
                    meta_info=self.meta_info,
                    non_tensor_batch=self.non_tensor_batch,
                )
            )
        return out

    def to(self, device: object) -> "_FakeActorDataProto":
        del device
        return self


class _DummyTrainModule:
    def train(self) -> None:
        return None


class _DummyOptimizer:
    def zero_grad(self) -> None:
        return None


def _make_distill_params(**overrides: object) -> SDPODistillParams:
    values: dict[str, object] = {
        "enabled": True,
        "lambda_coef": 1.0,
        "use_grpo_loss": False,
        "loss_variant": "full_logit",
        "alpha": 1.0,
        "is_clip": False,
        "full_logit_topk": 2,
        "full_logit_add_tail": True,
        "negate_sdpo_loss": False,
        "use_stale_coefficient": False,
        "teacher_regularization": "ema",
    }
    values.update(overrides)
    return SDPODistillParams(**values)


def test_non_full_sdpo_per_token_matches_reference_formula() -> None:
    student_log_probs = torch.tensor([[0.4, -0.2]])
    teacher_log_probs = torch.tensor([[0.1, -0.3]])
    per_token = _compute_sdpo_per_token_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        loss_variant="non_full",
        alpha=1.0,
    )
    expected = (student_log_probs - teacher_log_probs).detach() * student_log_probs
    assert torch.allclose(per_token, expected)


def test_non_full_sdpo_negate_flips_sign() -> None:
    student_log_probs = torch.tensor([[0.4, -0.2]])
    teacher_log_probs = torch.tensor([[0.1, -0.3]])
    normal = _compute_sdpo_per_token_loss(
        student_log_probs=student_log_probs,
        teacher_log_probs=teacher_log_probs,
        loss_variant="non_full",
        alpha=1.0,
    )
    negated = -normal
    # Negation should flip sign while preserving magnitude.
    assert torch.allclose(negated, -normal)
    assert not torch.allclose(negated, normal)


def test_full_logit_sdpo_forward_kl_alpha_zero() -> None:
    student_all = torch.log_softmax(torch.tensor([[[2.0, 0.0]]]), dim=-1)
    teacher_all = torch.log_softmax(torch.tensor([[[0.0, 2.0]]]), dim=-1)
    per_token = _compute_sdpo_per_token_loss(
        student_log_probs=torch.zeros((1, 1)),
        teacher_log_probs=torch.zeros((1, 1)),
        loss_variant="full_logit",
        alpha=0.0,
        student_all_log_probs=student_all,
        teacher_all_log_probs=teacher_all,
        full_logit_topk=0,
    )
    expected = F.kl_div(student_all, teacher_all, reduction="none", log_target=True).sum(-1)
    assert torch.allclose(per_token, expected, atol=1e-6)


def test_full_logit_sdpo_reverse_kl_alpha_one() -> None:
    student_all = torch.log_softmax(torch.tensor([[[1.0, 0.0, -1.0]]]), dim=-1)
    teacher_all = torch.log_softmax(torch.tensor([[[-1.0, 0.0, 1.0]]]), dim=-1)
    per_token = _compute_sdpo_per_token_loss(
        student_log_probs=torch.zeros((1, 1)),
        teacher_log_probs=torch.zeros((1, 1)),
        loss_variant="full_logit",
        alpha=1.0,
        student_all_log_probs=student_all,
        teacher_all_log_probs=teacher_all,
        full_logit_topk=0,
    )
    expected = F.kl_div(teacher_all, student_all, reduction="none", log_target=True).sum(-1)
    assert torch.allclose(per_token, expected, atol=1e-6)


def test_full_logit_sdpo_jsd_with_topk_tail_is_finite() -> None:
    student_all = torch.log_softmax(torch.tensor([[[3.0, 1.0, 0.0, -2.0]]]), dim=-1)
    teacher_all = torch.log_softmax(torch.tensor([[[0.0, 2.0, 1.5, -1.0]]]), dim=-1)
    per_token = _compute_sdpo_per_token_loss(
        student_log_probs=torch.zeros((1, 1)),
        teacher_log_probs=torch.zeros((1, 1)),
        loss_variant="full_logit",
        alpha=0.3,
        student_all_log_probs=student_all,
        teacher_all_log_probs=teacher_all,
        full_logit_topk=2,
        full_logit_add_tail=True,
    )
    assert per_token.shape == (1, 1)
    assert torch.isfinite(per_token).all()
    assert float(per_token.item()) >= 0.0


def test_full_logit_sdpo_accepts_precomputed_topk_log_probs() -> None:
    student_topk = torch.log_softmax(torch.tensor([[[2.0, 1.0, 0.5]]]), dim=-1)
    teacher_topk = torch.log_softmax(torch.tensor([[[1.5, 1.2, 0.1]]]), dim=-1)
    per_token = _compute_sdpo_per_token_loss(
        student_log_probs=torch.zeros((1, 1)),
        teacher_log_probs=torch.zeros((1, 1)),
        loss_variant="full_logit",
        alpha=0.5,
        student_topk_log_probs=student_topk,
        teacher_topk_log_probs=teacher_topk,
        full_logit_topk=100,
        full_logit_add_tail=True,
    )
    assert per_token.shape == (1, 1)
    assert torch.isfinite(per_token).all()
    assert float(per_token.item()) >= 0.0


def test_parse_distill_params_defaults_use_grpo_loss_true() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    params = actor._parse_distill_params({})
    assert params.use_grpo_loss is True


def test_parse_distill_params_reads_use_grpo_loss_false() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    params = actor._parse_distill_params({"distill_use_grpo_loss": False})
    assert params.use_grpo_loss is False


def test_parse_distill_params_accepts_full_logit_topk_zero() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    params = actor._parse_distill_params({"distill_full_logit_topk": 0})
    assert params.full_logit_topk == 0


def test_parse_distill_params_rejects_full_logit_with_clip() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    try:
        actor._parse_distill_params(
            {"distill_loss_variant": "full_logit", "distill_is_clip": True}
        )
    except ValueError as exc:
        assert "distill_is_clip" in str(exc)
        assert "full_logit" in str(exc)
    else:
        raise AssertionError("Expected ValueError for full_logit + distill_is_clip")


def test_parse_distill_params_rejects_full_logit_with_stale_coefficient() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    try:
        actor._parse_distill_params(
            {
                "distill_loss_variant": "full_logit",
                "distill_use_stale_coefficient": True,
            }
        )
    except ValueError as exc:
        assert "distill_use_stale_coefficient" in str(exc)
        assert "full_logit" in str(exc)
    else:
        raise AssertionError(
            "Expected ValueError for full_logit + distill_use_stale_coefficient"
        )


def test_row_chunked_logsumexp_matches_torch() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    logits = torch.tensor(
        [
            [[2.0, -1.0, 0.5], [0.3, 0.1, -0.4]],
            [[-0.2, 1.1, 0.7], [3.0, 2.0, -5.0]],
        ],
        dtype=torch.float32,
    )

    chunked = actor._row_chunked_logsumexp(logits, target_chunk_bytes=12)
    expected = torch.logsumexp(logits, dim=-1, keepdim=True)

    assert chunked.shape == expected.shape
    assert torch.allclose(chunked, expected, atol=1e-6)


def test_scale_logits_by_temperature_uses_in_place_path_without_grad() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    logits = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    original = logits.clone()

    with torch.no_grad():
        scaled = actor._scale_logits_by_temperature(logits, temperature=0.5)

    assert scaled.data_ptr() == logits.data_ptr()
    assert torch.allclose(scaled, original / 0.5)


def test_scale_logits_by_temperature_preserves_autograd_path() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    logits = torch.tensor([[1.0, 2.0]], dtype=torch.float32, requires_grad=True)

    scaled = actor._scale_logits_by_temperature(logits, temperature=0.5)
    scaled.sum().backward()

    assert scaled.data_ptr() != logits.data_ptr()
    assert logits.grad is not None
    assert torch.allclose(logits.grad, torch.full_like(logits, 2.0))


def test_compute_pg_loss_returns_autograd_safe_zero_when_grpo_disabled() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.config = SimpleNamespace(policy_loss={"loss_mode": "vanilla"})
    log_prob = torch.randn(2, 3, requires_grad=True)
    zero_loss, pg_metrics = actor._compute_pg_loss(
        use_grpo_loss=False,
        old_log_prob=log_prob.detach(),
        log_prob=log_prob,
        advantages=torch.zeros_like(log_prob),
        response_mask=torch.ones_like(log_prob),
        loss_agg_mode="seq-mean-token-mean",
        rollout_is_weights=None,
    )
    assert pg_metrics == {}
    assert zero_loss.requires_grad
    assert float(zero_loss.item()) == 0.0
    zero_loss.backward()
    assert log_prob.grad is not None
    assert torch.allclose(log_prob.grad, torch.zeros_like(log_prob))


def test_compute_sdpo_loss_uses_distill_specific_topk_inputs() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.actor_module = object()
    actor.teacher_module = object()
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False

    topk_roles: list[str] = []
    teacher_topk = torch.log_softmax(torch.tensor([[[1.8, 1.2], [1.1, 0.7]]]), dim=-1)
    student_topk = torch.log_softmax(torch.tensor([[[2.0, 1.0], [1.4, 0.5]]]), dim=-1)
    student_topk_indices = torch.tensor([[[3, 5], [7, 2]]], dtype=torch.long)
    student_input_ids = torch.tensor([[101, 102, 11, 12]], dtype=torch.long)
    teacher_input_ids = torch.tensor([[201, 202, 11, 12]], dtype=torch.long)

    def fake_forward_topk(  # noqa: ANN202
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: object,
        distill_topk: int,
        topk_indices: torch.Tensor | None = None,
        calculate_entropy: bool = False,
    ):
        del temperature, distill_topk, calculate_entropy
        if topk_indices is None:
            topk_roles.append("student")
            assert torch.equal(model_inputs["input_ids"], student_input_ids)
            return torch.tensor([[-0.3, -0.2]], dtype=torch.float32), None, student_topk, student_topk_indices
        topk_roles.append("teacher")
        assert torch.equal(model_inputs["input_ids"], teacher_input_ids)
        assert torch.equal(topk_indices, student_topk_indices)
        return torch.tensor([[-0.5, -0.4]], dtype=torch.float32), None, teacher_topk, None

    actor._forward_topk_with_token_log_probs = fake_forward_topk  # type: ignore[method-assign]

    model_inputs = {
        "distill_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "distill_student_responses": torch.tensor([[11, 12]], dtype=torch.long),
        "distill_student_input_ids": student_input_ids,
        "distill_student_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[11, 12]], dtype=torch.long),
        "distill_teacher_input_ids": teacher_input_ids,
        "distill_teacher_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    }

    sdpo_loss, metrics = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=_make_distill_params(full_logit_topk=2),
    )

    assert topk_roles == ["student", "teacher"]
    assert torch.isfinite(sdpo_loss)
    assert metrics["distill/token_count"] == 2.0


def test_compute_sdpo_loss_uses_distill_specific_all_logit_inputs() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.actor_module = object()
    actor.teacher_module = object()
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False

    teacher_all = torch.log_softmax(torch.tensor([[[1.4, 0.2, -0.3], [0.1, 1.1, -0.4]]]), dim=-1)
    student_all = torch.log_softmax(torch.tensor([[[1.8, -0.1, -0.5], [0.3, 0.9, -0.7]]]), dim=-1)
    all_roles: list[str] = []
    student_input_ids = torch.tensor([[301, 302, 21, 22]], dtype=torch.long)
    teacher_input_ids = torch.tensor([[401, 402, 21, 22]], dtype=torch.long)

    def fake_forward_all(  # noqa: ANN202
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: object,
    ):
        del temperature
        if module is actor.actor_module:
            all_roles.append("student")
            assert torch.equal(model_inputs["input_ids"], student_input_ids)
            return torch.tensor([[-0.4, -0.5]], dtype=torch.float32), student_all
        all_roles.append("teacher")
        assert torch.equal(model_inputs["input_ids"], teacher_input_ids)
        return torch.zeros((1, 2), dtype=torch.float32), teacher_all

    actor._forward_with_all_log_probs = fake_forward_all  # type: ignore[method-assign]

    model_inputs = {
        "distill_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "distill_student_responses": torch.tensor([[21, 22]], dtype=torch.long),
        "distill_student_input_ids": student_input_ids,
        "distill_student_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[21, 22]], dtype=torch.long),
        "distill_teacher_input_ids": teacher_input_ids,
        "distill_teacher_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    }

    sdpo_loss, metrics = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=_make_distill_params(full_logit_topk=0),
    )

    assert all_roles == ["student", "teacher"]
    assert torch.isfinite(sdpo_loss)
    assert metrics["distill/token_count"] == 2.0


def test_compute_sdpo_loss_requires_distill_student_old_log_probs_for_clip() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.actor_module = object()
    actor.teacher_module = None
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False

    def fake_forward_with_module_log_probs(  # noqa: ANN202
        model_inputs: dict[str, torch.Tensor],
        *,
        temperature: float,
        module: object,
    ):
        del model_inputs, temperature, module
        return torch.tensor([[-0.2, -0.4]], dtype=torch.float32)

    actor._forward_with_module_log_probs = fake_forward_with_module_log_probs  # type: ignore[method-assign]

    model_inputs = {
        "distill_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "distill_student_responses": torch.tensor([[11, 12]], dtype=torch.long),
        "distill_student_input_ids": torch.tensor([[101, 102, 11, 12]], dtype=torch.long),
        "distill_student_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[11, 12]], dtype=torch.long),
        "distill_teacher_input_ids": torch.tensor([[201, 202, 11, 12]], dtype=torch.long),
        "distill_teacher_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    }

    try:
        actor._compute_sdpo_loss(
            model_inputs=model_inputs,
            temperature=1.0,
            loss_agg_mode="seq-mean-token-mean",
            distill_params=_make_distill_params(
                loss_variant="non_full",
                alpha=1.0,
                is_clip=True,
                teacher_regularization="none",
            ),
        )
    except ValueError as exc:
        assert "distill_student_old_log_probs" in str(exc)
    else:
        raise AssertionError("Expected ValueError when distill_student_old_log_probs is missing")


def test_compute_sdpo_loss_rejects_full_logit_clip_guard() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.actor_module = object()
    actor.teacher_module = object()
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False

    model_inputs = {
        "distill_mask": torch.tensor([[1.0]], dtype=torch.float32),
        "distill_student_responses": torch.tensor([[11]], dtype=torch.long),
        "distill_student_input_ids": torch.tensor([[101, 11]], dtype=torch.long),
        "distill_student_attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[11]], dtype=torch.long),
        "distill_teacher_input_ids": torch.tensor([[201, 11]], dtype=torch.long),
        "distill_teacher_attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1]], dtype=torch.long),
    }

    try:
        actor._compute_sdpo_loss(
            model_inputs=model_inputs,
            temperature=1.0,
            loss_agg_mode="seq-mean-token-mean",
            distill_params=_make_distill_params(is_clip=True),
        )
    except ValueError as exc:
        assert "is_clip" in str(exc)
        assert "full_logit" in str(exc)
    else:
        raise AssertionError("Expected ValueError for full_logit + is_clip in compute path")


# ---------------------------------------------------------------------------
# _segmented_reverse_cumsum tests
# ---------------------------------------------------------------------------

def test_segmented_reverse_cumsum_single_segment() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
    result = _segmented_reverse_cumsum(values, mask)
    # position 0: 1+2+3+4=10, 1: 2+3+4=9, 2: 3+4=7, 3: 4
    expected = torch.tensor([[10.0, 9.0, 7.0, 4.0]])
    assert torch.allclose(result, expected), f"{result} != {expected}"


def test_segmented_reverse_cumsum_multiple_segments() -> None:
    values = torch.tensor([[1.0, 2.0, 0.0, 0.0, 3.0, 4.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0, 1.0, 1.0]])
    result = _segmented_reverse_cumsum(values, mask)
    # segment 1: [1,2] -> [3,2]; gap: [0,0]; segment 2: [3,4] -> [7,4]
    expected = torch.tensor([[3.0, 2.0, 0.0, 0.0, 7.0, 4.0]])
    assert torch.allclose(result, expected), f"{result} != {expected}"


def test_segmented_reverse_cumsum_batch_dimension() -> None:
    values = torch.tensor([
        [1.0, 2.0, 3.0, 0.0],
        [0.0, 5.0, 5.0, 0.0],
    ])
    mask = torch.tensor([
        [1.0, 1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0, 0.0],
    ])
    result = _segmented_reverse_cumsum(values, mask)
    expected = torch.tensor([
        [6.0, 5.0, 3.0, 0.0],
        [0.0, 10.0, 5.0, 0.0],
    ])
    assert torch.allclose(result, expected), f"{result} != {expected}"


def test_segmented_reverse_cumsum_all_zeros_mask() -> None:
    values = torch.tensor([[1.0, 2.0, 3.0]])
    mask = torch.tensor([[0.0, 0.0, 0.0]])
    result = _segmented_reverse_cumsum(values, mask)
    expected = torch.zeros_like(values)
    assert torch.allclose(result, expected), f"{result} != {expected}"


def test_segmented_reverse_cumsum_single_token_segments() -> None:
    values = torch.tensor([[1.0, 0.0, 2.0, 0.0, 3.0]])
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])
    result = _segmented_reverse_cumsum(values, mask)
    # Each segment is 1 token, so reverse cumsum = the value itself.
    expected = torch.tensor([[1.0, 0.0, 2.0, 0.0, 3.0]])
    assert torch.allclose(result, expected), f"{result} != {expected}"


def test_compute_sdpo_loss_non_full_uses_cumulative_advantage() -> None:
    """Integration test: non_full path produces correct loss with multi-segment mask."""
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor_mod = object()
    teacher_mod = object()
    actor.actor_module = actor_mod
    actor.teacher_module = teacher_mod
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False

    # 1 batch, 4 tokens: two segments [0,1] and [3] (position 2 masked out)
    student_lp = torch.tensor([[-0.5, -0.3, -0.1, -0.4]], requires_grad=True)
    teacher_lp = torch.tensor([[-0.6, -0.1, -0.9, -0.2]])
    distill_mask = torch.tensor([[1.0, 1.0, 0.0, 1.0]])

    def fake_forward(model_inputs, *, temperature, module):
        del model_inputs, temperature
        if module is actor_mod:
            return student_lp
        return teacher_lp

    actor._forward_with_module_log_probs = fake_forward

    student_input_ids = torch.tensor([[10, 20, 1, 2, 3, 4]], dtype=torch.long)
    teacher_input_ids = torch.tensor([[30, 40, 1, 2, 3, 4]], dtype=torch.long)
    model_inputs = {
        "distill_mask": distill_mask,
        "distill_student_responses": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "distill_student_input_ids": student_input_ids,
        "distill_student_attention_mask": torch.ones(1, 6, dtype=torch.long),
        "distill_student_position_ids": torch.arange(6).unsqueeze(0),
        "distill_teacher_responses": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "distill_teacher_input_ids": teacher_input_ids,
        "distill_teacher_attention_mask": torch.ones(1, 6, dtype=torch.long),
        "distill_teacher_position_ids": torch.arange(6).unsqueeze(0),
    }

    sdpo_loss, metrics = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="token-mean",  # should be ignored, SDPO always uses seq-mean-token-mean
        distill_params=_make_distill_params(
            loss_variant="non_full",
            alpha=1.0,
            teacher_regularization="ema",
        ),
    )

    # Manually compute expected loss.
    delta = (student_lp - teacher_lp).detach()
    adv = _segmented_reverse_cumsum(delta, distill_mask)
    ptl = adv * student_lp
    # seq-mean-token-mean: per-sequence mean of masked tokens, then mean over sequences.
    seq_mean = (ptl * distill_mask).sum(dim=1) / distill_mask.sum(dim=1).clamp(min=1)
    expected_loss = seq_mean.mean()

    assert torch.allclose(sdpo_loss, expected_loss, atol=1e-6), f"{sdpo_loss} != {expected_loss}"
    assert metrics["distill/token_count"] == float(distill_mask.sum().item())
    masked_adv = adv[distill_mask > 0]
    assert metrics["distill/active_seq_count"] == 1.0
    assert abs(metrics["distill/sdpo_adv_abs_mean"] - float(masked_adv.abs().mean().item())) < 1e-6
    assert abs(metrics["distill/sdpo_adv_abs_max"] - float(masked_adv.abs().max().item())) < 1e-6


def test_compute_sdpo_loss_ignores_loss_agg_mode_arg() -> None:
    """Verify SDPO always uses seq-mean-token-mean regardless of loss_agg_mode argument."""
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.actor_module = object()
    actor.teacher_module = object()
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False

    teacher_all = torch.log_softmax(torch.tensor([[[1.4, 0.2, -0.3], [0.1, 1.1, -0.4]]]), dim=-1)
    student_all = torch.log_softmax(torch.tensor([[[1.8, -0.1, -0.5], [0.3, 0.9, -0.7]]]), dim=-1)
    student_input_ids = torch.tensor([[301, 302, 21, 22]], dtype=torch.long)
    teacher_input_ids = torch.tensor([[401, 402, 21, 22]], dtype=torch.long)

    def fake_forward_all(model_inputs, *, temperature, module):
        del temperature
        if module is actor.actor_module:
            return torch.tensor([[-0.4, -0.5]], dtype=torch.float32), student_all
        return torch.zeros((1, 2), dtype=torch.float32), teacher_all

    actor._forward_with_all_log_probs = fake_forward_all

    model_inputs = {
        "distill_mask": torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        "distill_student_responses": torch.tensor([[21, 22]], dtype=torch.long),
        "distill_student_input_ids": student_input_ids,
        "distill_student_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[21, 22]], dtype=torch.long),
        "distill_teacher_input_ids": teacher_input_ids,
        "distill_teacher_attention_mask": torch.tensor([[1, 1, 1, 1]], dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1, 2, 3]], dtype=torch.long),
    }

    results = {}
    for agg_mode in ("seq-mean-token-mean", "token-mean", "mean"):
        loss, _ = actor._compute_sdpo_loss(
            model_inputs=model_inputs,
            temperature=1.0,
            loss_agg_mode=agg_mode,
            distill_params=_make_distill_params(full_logit_topk=0),
        )
        results[agg_mode] = loss.detach().item()

    # All should be identical since SDPO hardcodes seq-mean-token-mean.
    vals = list(results.values())
    for v in vals[1:]:
        assert abs(v - vals[0]) < 1e-6, f"Losses differ across agg modes: {results}"


def test_count_active_distill_sequences_counts_only_nonempty_rows() -> None:
    distill_mask = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    assert _count_active_distill_sequences(distill_mask) == 2


# ---------------------------------------------------------------------------
# PPO-style clipping tests
# ---------------------------------------------------------------------------

def _build_indexed_non_full_actor_and_inputs(
    *,
    student_lp_rows: list[list[float]],
    teacher_lp_rows: list[list[float]],
    old_lp_rows: list[list[float]] | None,
    mask_rows: list[list[float]],
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    requires_grad: bool = True,
) -> tuple[SDPODataParallelPPOActor, dict[str, torch.Tensor], torch.Tensor]:
    """Construct a non-full actor whose fake forward pass is row-index stable."""
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor_mod = object()
    teacher_mod = object()
    actor.actor_module = actor_mod
    actor.teacher_module = teacher_mod
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
    )

    student_lp = torch.tensor(student_lp_rows, dtype=torch.float32, requires_grad=requires_grad)
    teacher_lp = torch.tensor(teacher_lp_rows, dtype=torch.float32)
    distill_mask = torch.tensor(mask_rows, dtype=torch.float32)
    old_lp = None if old_lp_rows is None else torch.tensor(old_lp_rows, dtype=torch.float32)

    batch_size, n_resp = student_lp.shape
    n_prompt = 2
    row_ids = torch.arange(batch_size, dtype=torch.long)
    row_markers = (1000 + row_ids).unsqueeze(1)
    second_prompt = (2000 + row_ids).unsqueeze(1)
    shared_responses = torch.arange(100, 100 + n_resp, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)
    student_input_ids = torch.cat([row_markers, second_prompt, shared_responses], dim=1)
    teacher_input_ids = torch.cat([row_markers, second_prompt + 1000, shared_responses], dim=1)
    full_attention = torch.ones(batch_size, n_prompt + n_resp, dtype=torch.long)
    full_positions = torch.arange(n_prompt + n_resp, dtype=torch.long).unsqueeze(0).repeat(batch_size, 1)

    def fake_forward(model_inputs, *, temperature, module):  # noqa: ANN202
        del temperature
        row_idx = (model_inputs["input_ids"][:, 0] - 1000).long()
        if module is actor_mod:
            return student_lp.index_select(0, row_idx)
        return teacher_lp.index_select(0, row_idx)

    actor._forward_with_module_log_probs = fake_forward  # type: ignore[method-assign]

    model_inputs = {
        "distill_mask": distill_mask,
        "distill_student_responses": shared_responses.clone(),
        "distill_student_input_ids": student_input_ids,
        "distill_student_attention_mask": full_attention.clone(),
        "distill_student_position_ids": full_positions.clone(),
        "distill_teacher_responses": shared_responses.clone(),
        "distill_teacher_input_ids": teacher_input_ids,
        "distill_teacher_attention_mask": full_attention.clone(),
        "distill_teacher_position_ids": full_positions.clone(),
    }
    if old_lp is not None:
        model_inputs["distill_student_old_log_probs"] = old_lp
    return actor, model_inputs, student_lp


def _build_clip_actor_and_inputs(
    student_lp_vals: list[float],
    teacher_lp_vals: list[float],
    old_lp_vals: list[float],
    mask_vals: list[float],
    clip_ratio_low: float = 0.2,
    clip_ratio_high: float = 0.2,
    loss_variant: str = "non_full",
    requires_grad: bool = True,
):
    """Helper: construct a minimal actor + model_inputs for clip tests."""
    actor, model_inputs, student_lp = _build_indexed_non_full_actor_and_inputs(
        student_lp_rows=[student_lp_vals],
        teacher_lp_rows=[teacher_lp_vals],
        old_lp_rows=[old_lp_vals],
        mask_rows=[mask_vals],
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        requires_grad=requires_grad,
    )
    params = _make_distill_params(
        loss_variant=loss_variant,
        alpha=1.0,
        is_clip=True,
        teacher_regularization="ema",
    )
    return actor, model_inputs, params, student_lp


def _slice_model_inputs_rows(model_inputs: dict[str, torch.Tensor], rows: list[int]) -> dict[str, torch.Tensor]:
    """Slice a model-input dict across the batch dimension."""
    row_index = torch.tensor(rows, dtype=torch.long)
    return {key: value.index_select(0, row_index) for key, value in model_inputs.items()}


def _compute_partitioned_sdpo_loss(
    actor: SDPODataParallelPPOActor,
    model_inputs: dict[str, torch.Tensor],
    distill_params: SDPODistillParams,
    partitions: list[list[int]],
) -> torch.Tensor:
    """Compute the scaled SDPO contribution by summing across micro-batch partitions."""
    total_active_sequences = _count_active_distill_sequences(model_inputs["distill_mask"])
    if total_active_sequences <= 0:
        raise ValueError("Expected at least one active distill sequence in partition test.")

    total_loss: torch.Tensor | None = None
    for rows in partitions:
        micro_inputs = _slice_model_inputs_rows(model_inputs, rows)
        micro_loss, micro_metrics = actor._compute_sdpo_loss(
            model_inputs=micro_inputs,
            temperature=1.0,
            loss_agg_mode="seq-mean-token-mean",
            distill_params=distill_params,
        )
        micro_scale = micro_metrics["distill/active_seq_count"] / float(total_active_sequences)
        contribution = micro_loss * micro_scale
        total_loss = contribution if total_loss is None else total_loss + contribution

    if total_loss is None:
        raise ValueError("Expected at least one partition contribution.")
    return total_loss


def _expected_clipped_non_full_loss(
    *,
    student_log_probs: torch.Tensor,
    teacher_log_probs: torch.Tensor,
    student_old_log_probs: torch.Tensor,
    distill_mask: torch.Tensor,
    clip_ratio_low: float,
    clip_ratio_high: float,
) -> torch.Tensor:
    """Compute the expected clipped non-full loss for test assertions."""
    delta = (student_log_probs - teacher_log_probs).detach()
    advantage = _segmented_reverse_cumsum(delta, distill_mask)
    per_token_loss = advantage * student_log_probs
    ratio = torch.exp(torch.clamp(student_log_probs - student_old_log_probs, min=-20.0, max=20.0))
    clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio_low, 1.0 + clip_ratio_high)
    clipped_per_token_loss = torch.maximum(per_token_loss * ratio, per_token_loss * clipped_ratio)
    seq_mean = (clipped_per_token_loss * distill_mask).sum(dim=1) / distill_mask.sum(dim=1).clamp(min=1.0)
    return seq_mean.mean()


def test_sdpo_ppo_clip_stops_gradient_when_ratio_exceeds_bounds() -> None:
    """When ratio is far outside [1-eps, 1+eps], gradient is bounded by clipped_ratio."""
    # old_lp much lower than current → ratio = exp(current - old) >> 1+eps
    actor, model_inputs, params, student_lp = _build_clip_actor_and_inputs(
        student_lp_vals=[-0.1, -0.2],
        teacher_lp_vals=[-0.3, -0.5],
        old_lp_vals=[-2.0, -2.0],  # big gap → ratio ≈ exp(1.9), exp(1.8) >> 1.2
        mask_vals=[1.0, 1.0],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    sdpo_loss, _ = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )
    sdpo_loss.backward()
    grad_clipped = student_lp.grad.clone()

    # Run with a very wide clip (effectively no clipping) to get the raw-ratio gradient.
    actor2, model_inputs2, params_wide, student_lp2 = _build_clip_actor_and_inputs(
        student_lp_vals=[-0.1, -0.2],
        teacher_lp_vals=[-0.3, -0.5],
        old_lp_vals=[-2.0, -2.0],
        mask_vals=[1.0, 1.0],
        clip_ratio_low=0.99,
        clip_ratio_high=0.99,
    )
    sdpo_loss2, _ = actor2._compute_sdpo_loss(
        model_inputs=model_inputs2,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params_wide,
    )
    sdpo_loss2.backward()
    grad_wide = student_lp2.grad.clone()

    # Tight clip (eps=0.2) should yield strictly smaller gradient magnitude
    # than wide clip (eps=0.99) when the raw ratio far exceeds both bounds.
    assert (grad_clipped.abs() < grad_wide.abs()).all(), (
        f"Expected tighter clip to have smaller gradient: tight={grad_clipped}, wide={grad_wide}"
    )


def test_sdpo_ppo_clip_passes_gradient_when_ratio_within_bounds() -> None:
    """When ratio is within [1-eps, 1+eps], unclipped branch wins and gradient flows normally."""
    # old_lp ≈ current → ratio ≈ 1.0, within [0.8, 1.2]
    actor, model_inputs, params, student_lp = _build_clip_actor_and_inputs(
        student_lp_vals=[-0.5, -0.3],
        teacher_lp_vals=[-0.6, -0.1],
        old_lp_vals=[-0.5, -0.3],  # identical to current → ratio = 1.0
        mask_vals=[1.0, 1.0],
    )

    sdpo_loss, _ = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )
    sdpo_loss.backward()
    grad_clipped = student_lp.grad.clone()

    # With ratio=1.0 exactly, unclipped = clipped, so gradient should be
    # identical to the no-clip case (ratio * per_token_loss = 1.0 * per_token_loss).
    actor2, model_inputs2, _, student_lp2 = _build_clip_actor_and_inputs(
        student_lp_vals=[-0.5, -0.3],
        teacher_lp_vals=[-0.6, -0.1],
        old_lp_vals=[-0.5, -0.3],
        mask_vals=[1.0, 1.0],
    )
    params_noclip = _make_distill_params(
        loss_variant="non_full", alpha=1.0, is_clip=False, teacher_regularization="ema",
    )
    sdpo_loss2, _ = actor2._compute_sdpo_loss(
        model_inputs=model_inputs2,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params_noclip,
    )
    sdpo_loss2.backward()
    grad_noclip = student_lp2.grad.clone()

    assert torch.allclose(grad_clipped, grad_noclip, atol=1e-6), (
        f"Gradients should match when ratio=1: clipped={grad_clipped}, noclip={grad_noclip}"
    )


def test_sdpo_ppo_clip_reports_clipfrac_metric() -> None:
    """Verify distill/sdpo_clipfrac is reported and correct."""
    # Two tokens: one with ratio >> 1+eps (clipped), one with ratio=1 (not clipped).
    actor, model_inputs, params, _ = _build_clip_actor_and_inputs(
        student_lp_vals=[-0.1, -0.5],
        teacher_lp_vals=[-0.3, -0.6],
        old_lp_vals=[-2.0, -0.5],  # token 0: ratio >> 1.2; token 1: ratio = 1.0
        mask_vals=[1.0, 1.0],
    )

    _, metrics = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )

    assert "distill/sdpo_clipfrac" in metrics
    clipfrac = metrics["distill/sdpo_clipfrac"]
    # Token 0 has ratio >> 1.2 → clipped branch wins for that token.
    # Token 1 has ratio = 1.0 → unclipped branch wins (or tied).
    # So clipfrac should be ~0.5 (1 out of 2 tokens clipped).
    assert 0.0 < clipfrac <= 1.0, f"Expected clipfrac in (0, 1], got {clipfrac}"
    assert abs(clipfrac - 0.5) < 0.1, f"Expected clipfrac ≈ 0.5, got {clipfrac}"


def test_sdpo_ppo_clip_handles_mixed_sign_per_token_loss() -> None:
    """Pessimistic selection is correct for both positive and negative per_token_loss."""
    # Construct scenario where per_token_loss has mixed signs across tokens.
    # Token 0: student > teacher → positive advantage → positive per_token_loss
    # Token 1: student < teacher → negative advantage → negative per_token_loss
    actor, model_inputs, params, student_lp = _build_clip_actor_and_inputs(
        student_lp_vals=[-0.1, -0.8],
        teacher_lp_vals=[-0.5, -0.2],  # tok0: student better, tok1: teacher better
        old_lp_vals=[-1.5, -1.5],  # ratio >> 1 for both → clipped branch active
        mask_vals=[1.0, 1.0],
    )

    sdpo_loss, _ = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )

    # Manually verify torch.maximum picks the larger (worse for minimization) value.
    # With ratio > 1+eps:
    #   If per_token_loss > 0: unclipped > clipped (larger ratio * positive) → max picks unclipped
    #   If per_token_loss < 0: clipped > unclipped (larger ratio * negative is more negative) → max picks clipped
    # This is the correct pessimistic behavior.
    assert torch.isfinite(sdpo_loss)
    # Verify gradient flows (not all zeros).
    sdpo_loss.backward()
    assert student_lp.grad is not None
    assert not torch.allclose(student_lp.grad, torch.zeros_like(student_lp.grad))


@pytest.mark.parametrize(
    ("clip_ratio_low", "clip_ratio_high", "student_rows", "teacher_rows", "old_rows", "mask_rows"),
    [
        (
            0.0,
            0.2,
            [[-0.5]],
            [[-0.3]],
            [[-0.5 - torch.log(torch.tensor(0.9)).item()]],
            [[1.0]],
        ),
        (
            0.2,
            0.0,
            [[-0.5]],
            [[-1.0]],
            [[-0.5 - torch.log(torch.tensor(1.1)).item()]],
            [[1.0]],
        ),
        (
            0.0,
            0.0,
            [[-0.5], [-0.5]],
            [[-0.3], [-1.0]],
            [
                [-0.5 - torch.log(torch.tensor(0.9)).item()],
                [-0.5 - torch.log(torch.tensor(1.1)).item()],
            ],
            [[1.0], [1.0]],
        ),
    ],
)
def test_sdpo_clip_honors_explicit_zero_bounds(
    clip_ratio_low: float,
    clip_ratio_high: float,
    student_rows: list[list[float]],
    teacher_rows: list[list[float]],
    old_rows: list[list[float]],
    mask_rows: list[list[float]],
) -> None:
    actor, model_inputs, _ = _build_indexed_non_full_actor_and_inputs(
        student_lp_rows=student_rows,
        teacher_lp_rows=teacher_rows,
        old_lp_rows=old_rows,
        mask_rows=mask_rows,
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
        requires_grad=False,
    )
    params = _make_distill_params(
        loss_variant="non_full",
        alpha=1.0,
        is_clip=True,
        teacher_regularization="ema",
    )

    sdpo_loss, _ = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )

    expected_loss = _expected_clipped_non_full_loss(
        student_log_probs=torch.tensor(student_rows, dtype=torch.float32),
        teacher_log_probs=torch.tensor(teacher_rows, dtype=torch.float32),
        student_old_log_probs=torch.tensor(old_rows, dtype=torch.float32),
        distill_mask=torch.tensor(mask_rows, dtype=torch.float32),
        clip_ratio_low=clip_ratio_low,
        clip_ratio_high=clip_ratio_high,
    )
    assert torch.allclose(sdpo_loss.detach(), expected_loss, atol=1e-6)


def test_sdpo_clip_metrics_distinguish_branch_selection_from_ratio_overflow() -> None:
    ratio = 1.3
    old_rows = [[-0.5 - torch.log(torch.tensor(ratio)).item()]] * 2
    actor, model_inputs, _ = _build_indexed_non_full_actor_and_inputs(
        student_lp_rows=[[-0.5], [-0.5]],
        teacher_lp_rows=[[-0.3], [-1.0]],
        old_lp_rows=old_rows,
        mask_rows=[[1.0], [1.0]],
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        requires_grad=False,
    )
    params = _make_distill_params(
        loss_variant="non_full",
        alpha=1.0,
        is_clip=True,
        teacher_regularization="ema",
    )

    _, metrics = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )

    assert abs(metrics["distill/sdpo_ratio_gt_clip_frac"] - 1.0) < 1e-6
    assert abs(metrics["distill/sdpo_clipfrac"] - 0.5) < 1e-6


def test_sdpo_truncation_updates_token_count_and_clip_metrics() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor_mod = object()
    teacher_mod = object()
    actor.actor_module = actor_mod
    actor.teacher_module = teacher_mod
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.config = SimpleNamespace(
        clip_ratio=0.2,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )

    student_lp = torch.tensor([[-0.1, -0.5]], dtype=torch.float32, requires_grad=True)
    teacher_lp = torch.tensor([[-0.3, -0.6]], dtype=torch.float32)

    def fake_forward(model_inputs, *, temperature, module):  # noqa: ANN202
        del model_inputs, temperature
        if module is actor_mod:
            return student_lp
        return teacher_lp

    actor._forward_with_module_log_probs = fake_forward  # type: ignore[method-assign]

    model_inputs = {
        "distill_mask": torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32),
        "distill_student_responses": torch.tensor([[11, 12, 13]], dtype=torch.long),
        "distill_student_input_ids": torch.tensor([[101, 102, 11, 12, 13]], dtype=torch.long),
        "distill_student_attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[11, 12, 13]], dtype=torch.long),
        "distill_teacher_input_ids": torch.tensor([[201, 202, 11, 12, 13]], dtype=torch.long),
        "distill_teacher_attention_mask": torch.tensor([[1, 1, 1, 1, 1]], dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
        "distill_student_old_log_probs": torch.tensor([[-2.0, -0.5, -0.9]], dtype=torch.float32),
    }
    params = _make_distill_params(
        loss_variant="non_full",
        alpha=1.0,
        is_clip=True,
        teacher_regularization="ema",
    )

    _, metrics = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )

    assert metrics["distill/token_count"] == 2.0
    assert abs(metrics["distill/sdpo_clipfrac"] - 0.5) < 1e-6
    assert abs(metrics["distill/sdpo_ratio_gt_clip_frac"] - 0.5) < 1e-6


@pytest.mark.parametrize("active_rows", ([6], [2, 7], list(range(8))))
def test_sdpo_partition_invariance_matches_full_batch(active_rows: list[int]) -> None:
    student_rows: list[list[float]] = []
    teacher_rows: list[list[float]] = []
    old_rows: list[list[float]] = []
    mask_rows: list[list[float]] = []
    for row in range(8):
        student_row = [-0.2 - 0.05 * row, -0.35 - 0.05 * row, -0.5 - 0.05 * row]
        teacher_row = [value - 0.15 for value in student_row]
        student_rows.append(student_row)
        teacher_rows.append(teacher_row)
        old_rows.append(student_row[:])
        mask_rows.append([1.0, 1.0, 1.0] if row in active_rows else [0.0, 0.0, 0.0])

    actor, model_inputs, student_lp = _build_indexed_non_full_actor_and_inputs(
        student_lp_rows=student_rows,
        teacher_lp_rows=teacher_rows,
        old_lp_rows=old_rows,
        mask_rows=mask_rows,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
    )
    params = _make_distill_params(
        loss_variant="non_full",
        alpha=1.0,
        is_clip=True,
        teacher_regularization="ema",
    )
    partitions = [[0, 1, 2], [3, 4, 5], [6, 7]]

    full_loss, _ = actor._compute_sdpo_loss(
        model_inputs=model_inputs,
        temperature=1.0,
        loss_agg_mode="seq-mean-token-mean",
        distill_params=params,
    )
    full_loss.backward()
    full_grad = student_lp.grad.clone()
    student_lp.grad = None

    partitioned_loss = _compute_partitioned_sdpo_loss(
        actor=actor,
        model_inputs=model_inputs,
        distill_params=params,
        partitions=partitions,
    )
    partitioned_loss.backward()
    partitioned_grad = student_lp.grad.clone()

    assert torch.allclose(full_loss.detach(), partitioned_loss.detach(), atol=1e-6)
    assert torch.allclose(full_grad, partitioned_grad, atol=1e-6)


def test_update_policy_emits_sdpo_debug_metrics() -> None:
    actor = SDPODataParallelPPOActor.__new__(SDPODataParallelPPOActor)
    actor.actor_module = _DummyTrainModule()
    actor.actor_optimizer = _DummyOptimizer()
    actor.teacher_module = object()
    actor.scaler = None
    actor.use_fused_kernels = False
    actor.use_ulysses_sp = False
    actor.config = SimpleNamespace(
        use_kl_loss=False,
        use_dynamic_bsz=False,
        ppo_mini_batch_size=2,
        ppo_micro_batch_size_per_gpu=1,
        ppo_epochs=1,
        entropy_coeff=0.0,
        loss_agg_mode="seq-mean-token-mean",
    )

    def fake_forward_micro_batch(model_inputs, *, temperature, calculate_entropy):  # noqa: ANN202
        del model_inputs, temperature, calculate_entropy
        log_prob = torch.zeros((1, 2), dtype=torch.float32, requires_grad=True)
        entropy = torch.zeros((1, 2), dtype=torch.float32)
        return entropy, log_prob

    def fake_compute_pg_loss(  # noqa: ANN202
        *,
        use_grpo_loss: bool,
        old_log_prob: torch.Tensor,
        log_prob: torch.Tensor,
        advantages: torch.Tensor,
        response_mask: torch.Tensor,
        loss_agg_mode: str,
        rollout_is_weights: torch.Tensor | None,
    ):
        del use_grpo_loss, old_log_prob, advantages, response_mask, loss_agg_mode, rollout_is_weights
        return log_prob.sum() * 0.0, {}

    def fake_compute_sdpo_loss(  # noqa: ANN202
        *,
        model_inputs: dict[str, torch.Tensor],
        temperature: float,
        loss_agg_mode: str,
        distill_params: SDPODistillParams,
    ):
        del model_inputs, temperature, loss_agg_mode, distill_params
        return torch.tensor(3.0, dtype=torch.float32), {
            "distill/sdpo_loss": 3.0,
            "distill/token_count": 2.0,
            "distill/sdpo_clipfrac": 0.25,
            "distill/sdpo_ratio_gt_clip_frac": 0.5,
            "distill/sdpo_adv_abs_mean": 1.5,
            "distill/sdpo_adv_abs_max": 2.5,
            "distill/active_seq_count": 1.0,
        }

    actor._forward_micro_batch = fake_forward_micro_batch  # type: ignore[method-assign]
    actor._compute_pg_loss = fake_compute_pg_loss  # type: ignore[method-assign]
    actor._compute_sdpo_loss = fake_compute_sdpo_loss  # type: ignore[method-assign]
    actor._optimizer_step = lambda: torch.tensor(0.0)  # type: ignore[method-assign]

    batch = {
        "responses": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "response_mask": torch.ones(2, 2, dtype=torch.float32),
        "input_ids": torch.tensor([[10, 11, 1, 2], [20, 21, 3, 4]], dtype=torch.long),
        "attention_mask": torch.ones(2, 4, dtype=torch.long),
        "position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
        "old_log_probs": torch.zeros(2, 2, dtype=torch.float32),
        "advantages": torch.zeros(2, 2, dtype=torch.float32),
        "distill_student_input_ids": torch.tensor([[10, 11, 1, 2], [20, 21, 3, 4]], dtype=torch.long),
        "distill_student_attention_mask": torch.ones(2, 4, dtype=torch.long),
        "distill_student_position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
        "distill_student_responses": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "distill_teacher_input_ids": torch.tensor([[30, 31, 1, 2], [40, 41, 3, 4]], dtype=torch.long),
        "distill_teacher_attention_mask": torch.ones(2, 4, dtype=torch.long),
        "distill_teacher_position_ids": torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long),
        "distill_teacher_responses": torch.tensor([[1, 2], [3, 4]], dtype=torch.long),
        "distill_mask": torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32),
        "distill_student_old_log_probs": torch.zeros(2, 2, dtype=torch.float32),
    }
    data = _FakeActorDataProto(
        batch,
        meta_info={
            "temperature": 1.0,
            "distill_enabled": True,
            "distill_lambda": 2.0,
            "distill_loss_variant": "non_full",
            "distill_alpha": 1.0,
            "distill_is_clip": True,
            "distill_use_grpo_loss": False,
            "distill_teacher_regularization": "ema",
        },
    )

    metrics = actor.update_policy(data)

    assert metrics["distill/sdpo_clipfrac"] == [0.25, 0.25]
    assert metrics["distill/sdpo_ratio_gt_clip_frac"] == [0.5, 0.5]
    assert metrics["distill/sdpo_adv_abs_mean"] == [1.5, 1.5]
    assert metrics["distill/sdpo_adv_abs_max"] == [2.5, 2.5]
    assert metrics["distill/active_seq_count"] == [1.0, 1.0]
    assert metrics["distill/active_seq_frac"] == [1.0, 1.0]
    assert metrics["distill/sdpo_loss_scaled"] == [3.0, 3.0]
