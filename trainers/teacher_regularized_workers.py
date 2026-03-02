"""Local worker extensions for teacher-regularized SDPO distillation.

This module adds teacher-logprob and teacher-sync APIs without modifying
third-party worker implementations.
"""

from __future__ import annotations

from typing import Any

import torch
from omegaconf import DictConfig
from verl import DataProto  # type: ignore
from verl.single_controller.base.decorator import Dispatch, make_nd_compute_dataproto_dispatch_fn, register
from verl.utils.fsdp_utils import load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker


class _TeacherRegularizationMixin:
    """Mixin that adds teacher log-prob scoring and teacher parameter sync."""

    def __init__(self, config: DictConfig, role: str, **kwargs: Any) -> None:
        # This class is selected only when teacher-regularized distillation is enabled.
        # We remap actor_rollout -> actor_rollout_ref so one worker owns both actor
        # and teacher(ref) modules.
        if role == "actor_rollout":
            role = "actor_rollout_ref"
        super().__init__(config=config, role=role, **kwargs)

    @register(dispatch_mode=make_nd_compute_dataproto_dispatch_fn(mesh_name="actor"))
    def compute_teacher_log_prob(self, data: DataProto) -> DataProto:
        """Compute denominator log-probs using the teacher(ref) snapshot."""
        ref_out = self.compute_ref_log_prob(data)
        ref_log_prob = ref_out.batch["ref_log_prob"]
        return DataProto.from_dict(tensors={"old_log_probs": ref_log_prob})

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def sync_teacher_from_actor(self, mode: str, update_rate: float = 0.0) -> dict[str, float]:
        """Synchronize teacher(ref) parameters from actor parameters.

        Args:
            mode: ``"ema"`` or ``"hard"``.
            update_rate: EMA coefficient used when ``mode=="ema"``.
        """
        if not self._is_actor or not self._is_ref:
            raise ValueError("Teacher sync requires actor+ref worker role (actor_rollout_ref).")
        if not hasattr(self, "actor_module_fsdp") or not hasattr(self, "ref_module_fsdp"):
            raise ValueError("Teacher sync requires initialized actor_module_fsdp and ref_module_fsdp.")
        if mode not in {"ema", "hard"}:
            raise ValueError(f"Unsupported teacher sync mode: {mode}. Expected one of ['ema', 'hard'].")
        if mode == "ema" and not (0.0 <= float(update_rate) <= 1.0):
            raise ValueError(f"EMA update_rate must be in [0, 1], got {update_rate}.")

        actor_offload = bool(getattr(self, "_is_offload_param", False))
        ref_offload = bool(self.config.ref.fsdp_config.get("param_offload", False))
        if actor_offload:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if ref_offload:
            load_fsdp_model_to_gpu(self.ref_module_fsdp)

        actor_params = dict(self.actor_module_fsdp.named_parameters())
        teacher_params = dict(self.ref_module_fsdp.named_parameters())
        if actor_params.keys() != teacher_params.keys():
            raise ValueError("Actor/teacher parameter structure mismatch during teacher sync.")

        with torch.no_grad():
            if mode == "ema":
                rate = float(update_rate)
                for name, teacher_param in teacher_params.items():
                    actor_param = actor_params[name]
                    actor_data = actor_param.data.to(device=teacher_param.device, dtype=teacher_param.dtype)
                    teacher_param.data.mul_(1.0 - rate).add_(actor_data, alpha=rate)
            else:
                for name, teacher_param in teacher_params.items():
                    actor_param = actor_params[name]
                    actor_data = actor_param.data.to(device=teacher_param.device, dtype=teacher_param.dtype)
                    teacher_param.data.copy_(actor_data)

                actor_buffers = dict(self.actor_module_fsdp.named_buffers())
                teacher_buffers = dict(self.ref_module_fsdp.named_buffers())
                if actor_buffers.keys() != teacher_buffers.keys():
                    raise ValueError("Actor/teacher buffer structure mismatch during hard teacher sync.")
                for name, teacher_buffer in teacher_buffers.items():
                    actor_buffer = actor_buffers[name]
                    actor_data = actor_buffer.data.to(device=teacher_buffer.device, dtype=teacher_buffer.dtype)
                    teacher_buffer.data.copy_(actor_data)

        if ref_offload:
            offload_fsdp_model_to_cpu(self.ref_module_fsdp)
        if actor_offload:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
        return {"applied": 1.0}


class TeacherRegularizedActorRolloutRefWorker(_TeacherRegularizationMixin, ActorRolloutRefWorker):
    """Sync worker with teacher-regularization support."""


class TeacherRegularizedAsyncActorRolloutRefWorker(_TeacherRegularizationMixin, AsyncActorRolloutRefWorker):
    """Async worker with teacher-regularization support."""

