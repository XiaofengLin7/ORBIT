"""Standalone patch module loaded by the .pth file at Python startup.

This file is intentionally a TOP-LEVEL module (not inside the ``trainers/``
package) so that importing it does not trigger ``trainers/__init__.py``,
which transitively imports torch / verl / rllm and would initialize CUDA
in every Python process the .pth file fires in (including Ray workers
before they get their per-worker GPU assignment — that path was empirically
verified to break ``init_device_mesh`` with
"CUDA-capable device(s) is/are busy or unavailable" on the SCC cluster).

This module imports ONLY ``wrapt`` at top level. The wrapt callback below
fires lazily, only when verl's ``DataParallelPPOActor`` is first imported
inside ``ActorRolloutRefWorker.init_model``. By that point Ray's GPU setup
is complete and the heavier imports inside the callback (which touch verl
and torch) are safe.

Loaded automatically by the .pth file
``<conda_env>/site-packages/orbit_segtrain.pth`` written on every launch by
``scripts/train_multi_task_multi_episode.sh``.
"""

from __future__ import annotations

import wrapt


@wrapt.when_imported("verl.workers.actor.dp_actor")
def _patch(mod):
    """Replace verl's stock ``DataParallelPPOActor`` with the trajectory-uniform variant.

    Runs once, the first time any process imports the verl module.
    """
    # Heavy imports deferred to callback fire time. By the time this runs:
    #  - We're inside the actor's own import chain (verl.utils.device has
    #    already been imported by the dp_actor module we're patching, so
    #    CUDA is already initialized in this process).
    #  - Ray has finished its per-worker GPU setup.
    #
    # Importing trainers.trajectory_uniform_actor here also pulls in
    # trainers/__init__.py (which we keep deliberately empty so this doesn't
    # cascade into rllm).
    from trainers.trajectory_uniform_actor import TrajectoryUniformPPOActor

    mod.DataParallelPPOActor = TrajectoryUniformPPOActor
