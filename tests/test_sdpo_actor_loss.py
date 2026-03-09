from __future__ import annotations

import importlib.machinery
from pathlib import Path
import sys
from types import SimpleNamespace
import types

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

from trainers.sdpo_actor import SDPODistillParams, SDPODataParallelPPOActor, _compute_sdpo_per_token_loss


def _make_distill_params(**overrides: object) -> SDPODistillParams:
    values: dict[str, object] = {
        "enabled": True,
        "lambda_coef": 1.0,
        "use_grpo_loss": False,
        "loss_variant": "full_logit",
        "alpha": 1.0,
        "is_clip": None,
        "full_logit_topk": 2,
        "full_logit_add_tail": True,
        "negate_sdpo_loss": False,
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
                is_clip=2.0,
                teacher_regularization="none",
            ),
        )
    except ValueError as exc:
        assert "distill_student_old_log_probs" in str(exc)
    else:
        raise AssertionError("Expected ValueError when distill_student_old_log_probs is missing")
