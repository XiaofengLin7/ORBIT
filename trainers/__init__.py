"""Custom trainers for multi-episode training.

This package's ``__init__.py`` is intentionally empty (no re-exports). Eager
imports here would cascade into rllm/verl on package load, which:
  - Initializes CUDA in every Python process the package is imported in
    (verl.utils.device runs ``torch.cuda.is_available()`` at module top).
  - Calls ``signal.signal(SIGALRM, ...)`` at module top of
    ``rllm.rewards.code_utils.taco``, which raises
    ``ValueError: signal only works in main thread`` when imported from
    a non-main thread (e.g. from inside a wrapt.when_imported callback
    fired during Ray's worker init chain).

Both side effects break our `.pth`-loaded patch in ``orbit_segtrain_patch``,
which has to import ``trainers.trajectory_uniform_actor`` from inside the
wrapt callback. With an empty package init, only the requested submodule
is loaded — no heavy cascade.

Import submodules directly, e.g.::

    from trainers.multi_episode_trainer import MultiEpisodeAgentPPOTrainer
    from trainers.trajectory_uniform_actor import TrajectoryUniformPPOActor
"""
