"""Tests for the truncated-trajectory filter in ``trainers.multi_episode_trainer``.

The filter drops trajectories whose engine-side ``termination_reason``
indicates a truncation-class outcome before they reach the parent PPO
transform, so the trajectory-uniform actor's ``N_G`` and GRPO group
normalization both see only kept rollouts.

These tests target ``filter_truncated_trajectories`` directly — a pure
function so we don't need to bring up Hydra / Ray / verl.
"""

from __future__ import annotations

from trainers.multi_episode_trainer import (
    DEFAULT_TRUNCATION_REASONS,
    _is_truncated_trajectory,
    compute_kept_trajectory_indices,
    filter_truncated_trajectories,
)


def _traj(reason: str | None) -> dict:
    """Minimal trajectory dict — only the key the filter reads."""
    d: dict = {"prompt_tokens": [], "response_tokens": []}
    if reason is not None:
        d["termination_reason"] = reason
    return d


# --------------------------- helper ------------------------------------

def test_is_truncated_matches_engine_reasons():
    reasons = set(DEFAULT_TRUNCATION_REASONS)
    for r in DEFAULT_TRUNCATION_REASONS:
        assert _is_truncated_trajectory(_traj(r), reasons)
    assert not _is_truncated_trajectory(_traj("DONE"), reasons)
    assert not _is_truncated_trajectory(_traj("MAX_STEPS"), reasons)
    assert not _is_truncated_trajectory(_traj(None), reasons)


# --------------------------- defaults ----------------------------------

def test_disabled_filter_is_identity():
    trajs = [_traj("TRUNCATION"), _traj("DONE"), _traj(None)]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=False, is_validation=False,
    )
    assert out == trajs
    assert fallback is False


def test_validation_mode_short_circuits_filter():
    trajs = [_traj("TRUNCATION"), _traj("DONE")]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=True, is_validation=True,
    )
    assert out == trajs
    assert fallback is False


def test_empty_input_is_no_op():
    out, fallback = filter_truncated_trajectories(
        [], enable=True, is_validation=False,
    )
    assert out == []
    assert fallback is False


# --------------------------- standard filtering ------------------------

def test_drops_truncated_keeps_rest_with_defaults():
    trajs = [
        _traj("DONE"),
        _traj("TRUNCATION"),
        _traj("DONE"),
        _traj("PROMPT_TRUNCATION"),
        _traj(None),  # missing key defaults to "kept"
        _traj("SUMMARIZATION_BUDGET_EXCEEDED"),
        _traj("SUMMARIZATION_FAILED"),
        _traj("MAX_STEPS"),  # NOT in default set → kept
    ]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=True, is_validation=False,
    )
    kept_reasons = [t.get("termination_reason") for t in out]
    assert kept_reasons == ["DONE", "DONE", None, "MAX_STEPS"]
    assert fallback is False


def test_missing_termination_reason_is_kept():
    trajs = [_traj(None), _traj(None)]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=True, is_validation=False,
    )
    assert out == trajs
    assert fallback is False


def test_custom_reasons_set_includes_max_steps():
    trajs = [
        _traj("DONE"),
        _traj("MAX_STEPS"),
        _traj("TRUNCATION"),
    ]
    out, fallback = filter_truncated_trajectories(
        trajs,
        enable=True,
        is_validation=False,
        reasons={"MAX_STEPS"},  # only filter MAX_STEPS
    )
    kept_reasons = [t.get("termination_reason") for t in out]
    assert kept_reasons == ["DONE", "TRUNCATION"]
    assert fallback is False


def test_custom_reasons_can_be_empty():
    """Empty reasons set ⇒ nothing is dropped (filter effectively off)."""
    trajs = [_traj("TRUNCATION"), _traj("DONE")]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=True, is_validation=False, reasons=set(),
    )
    assert out == trajs
    assert fallback is False


# --------------------------- fallback ----------------------------------

def test_all_dropped_triggers_fallback():
    trajs = [
        _traj("TRUNCATION"),
        _traj("PROMPT_TRUNCATION"),
        _traj("SUMMARIZATION_FAILED"),
    ]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=True, is_validation=False,
    )
    # All would be dropped → fallback returns the full unfiltered list.
    assert out == trajs
    assert fallback is True


def test_fallback_not_triggered_when_any_kept():
    trajs = [_traj("TRUNCATION"), _traj("DONE")]
    out, fallback = filter_truncated_trajectories(
        trajs, enable=True, is_validation=False,
    )
    assert [t.get("termination_reason") for t in out] == ["DONE"]
    assert fallback is False


# --------------------------- isolation ---------------------------------

def test_returned_list_is_a_copy():
    """Caller mutating the result must not bleed back into the input."""
    trajs = [_traj("DONE")]
    out, _ = filter_truncated_trajectories(
        trajs, enable=False, is_validation=False,
    )
    out.append(_traj("EXTRA"))
    assert len(trajs) == 1


# --------------------------- kept-indices helper -----------------------

def test_kept_indices_are_sorted_ascending():
    """kept_idxs must be sorted so that slicing a parallel DataProto with
    select_idxs keeps the same trajectory-to-row alignment as slicing
    the python list with the same indices."""
    trajs = [
        _traj("DONE"),         # 0 kept
        _traj("TRUNCATION"),   # 1 dropped
        _traj("DONE"),         # 2 kept
        _traj("TRUNCATION"),   # 3 dropped
        _traj("DONE"),         # 4 kept
    ]
    kept, fallback = compute_kept_trajectory_indices(
        trajs, enable=True, is_validation=False,
    )
    assert kept == [0, 2, 4]
    assert kept == sorted(kept)
    assert fallback is False


def test_kept_indices_no_filter_returns_full_range():
    trajs = [_traj("DONE"), _traj("TRUNCATION")]
    kept, fallback = compute_kept_trajectory_indices(
        trajs, enable=False, is_validation=False,
    )
    assert kept == [0, 1]
    assert fallback is False


def test_kept_indices_validation_returns_full_range():
    trajs = [_traj("TRUNCATION"), _traj("DONE")]
    kept, fallback = compute_kept_trajectory_indices(
        trajs, enable=True, is_validation=True,
    )
    assert kept == [0, 1]
    assert fallback is False


def test_kept_indices_fallback_returns_full_range():
    """When every trajectory would be dropped, return the full range so
    the caller keeps the batch intact rather than crashing on an empty
    one. fallback_applied flag should fire."""
    trajs = [
        _traj("TRUNCATION"),
        _traj("PROMPT_TRUNCATION"),
        _traj("SUMMARIZATION_FAILED"),
    ]
    kept, fallback = compute_kept_trajectory_indices(
        trajs, enable=True, is_validation=False,
    )
    assert kept == [0, 1, 2]
    assert fallback is True


# --------------- credit-assignment alignment -------------------------
#
# These tests simulate the lockstep slicing that happens inside
# `_expanded_update_actor`: the same `kept_idxs` is used to (1) slice
# raw_trajs and (2) slice the DataProto via select_idxs. After the slice,
# raw_trajs[j] must still pair with batch row j so the downstream
# advantage-fill loop (multi_episode_trainer.py:430) writes G_k into the
# correct row's tensor. We mock the DataProto as a parallel list.

def _make_pair(reason: str | None, advantage_marker: float) -> tuple[dict, float]:
    """One trajectory dict + a unique 'advantage marker' representing
    that trajectory's per-row tensor in the batch."""
    return (_traj(reason), advantage_marker)


def test_lockstep_slicing_preserves_traj_batch_alignment():
    """raw_trajs[j] and batch_row[j] must remain paired post-filter."""
    pairs = [
        _make_pair("DONE", 100.0),         # 0 kept
        _make_pair("TRUNCATION", 101.0),   # 1 dropped
        _make_pair("DONE", 102.0),         # 2 kept
        _make_pair("DONE", 103.0),         # 3 kept
        _make_pair("TRUNCATION", 104.0),   # 4 dropped
    ]
    raw_trajs = [p[0] for p in pairs]
    # Each row of the mocked "batch" carries a unique marker so we can
    # detect any row shuffle.
    batch_markers = [p[1] for p in pairs]

    # Embed the marker into the trajectory dict so we can verify identity
    # without needing a real DataProto.
    for t, m in zip(raw_trajs, batch_markers):
        t["__expected_advantage__"] = m

    kept_idxs, _ = compute_kept_trajectory_indices(
        raw_trajs, enable=True, is_validation=False,
    )
    assert kept_idxs == [0, 2, 3]

    # Apply the same slice to both, exactly as _expanded_update_actor does.
    filtered_trajs = [raw_trajs[i] for i in kept_idxs]
    filtered_batch = [batch_markers[i] for i in kept_idxs]

    # Alignment property: filtered_trajs[j]'s stored marker == filtered_batch[j].
    # If this fails, the row-fill loop at multi_episode_trainer.py:430
    # would write the wrong G_k into the wrong row.
    for j, (t, m) in enumerate(zip(filtered_trajs, filtered_batch)):
        assert t["__expected_advantage__"] == m, (
            f"trajectory/batch misaligned at position {j}: "
            f"traj marker {t['__expected_advantage__']} vs batch marker {m}"
        )


def test_chunk_returns_index_matches_filtered_trajectory_order():
    """`compute_chunk_returns_for_batch(raw_trajs, …)` returns a list
    indexed by the *filtered* position. The advantage-fill loop iterates
    `for traj_i in range(n_g)` and dereferences both
    `per_chunk_returns[traj_i]` and `batch.batch['advantages'][traj_i]`.
    Both must point at the same original trajectory."""
    from trainers.chunk_advantage import compute_chunk_returns_for_batch

    def _action(done=False):
        return {
            "is_summarization": False, "trigger": None,
            "episode_done": done, "episode_index": 0,
        }

    raw_trajs = [
        {  # 0 — kept, ep success
            "termination_reason": "DONE",
            "step_metadata": [_action(False), _action(True)],
            "episode_rewards": [1.0],
        },
        {  # 1 — dropped (truncated)
            "termination_reason": "TRUNCATION",
            "step_metadata": [_action(False)],
            "episode_rewards": [0.0],
        },
        {  # 2 — kept, ep failure
            "termination_reason": "DONE",
            "step_metadata": [_action(False), _action(True)],
            "episode_rewards": [0.0],
        },
    ]
    kept_idxs, _ = compute_kept_trajectory_indices(
        raw_trajs, enable=True, is_validation=False,
    )
    assert kept_idxs == [0, 2]

    filtered = [raw_trajs[i] for i in kept_idxs]
    per_chunk = compute_chunk_returns_for_batch(
        filtered, scope="per_episode", gamma=0.9,
    )
    # Filtered position 0 is the SUCCESSFUL trajectory (originally idx 0):
    # its last chunk gets R=1.0 directly.
    assert per_chunk[0][-1][0] == 1.0
    # Filtered position 1 is the FAILED trajectory (originally idx 2):
    # G=0 everywhere.
    assert all(g == 0.0 for g, _ in per_chunk[1])
    # If the slice misaligned, position 0 might receive the failed
    # trajectory's chunks (all zero) — this assertion catches that.


def test_filter_disabled_preserves_index_order():
    """When filter is off, kept_idxs == range(N). The slicing is a no-op
    and downstream sees the original batch order unchanged."""
    trajs = [_traj("TRUNCATION"), _traj("DONE"), _traj("TRUNCATION")]
    kept, fallback = compute_kept_trajectory_indices(
        trajs, enable=False, is_validation=False,
    )
    assert kept == list(range(len(trajs)))
    assert fallback is False
    # Slicing by `range(N)` is identity.
    assert [trajs[i] for i in kept] == trajs


# ---------- engine-side reward zeroing consistency ---------------------
#
# `summarizing_engine.py` zeros TWO reward fields when a trajectory is
# truncated, so that GRPO and chunk_discounted_topr both see zero at the
# source:
#
#   * `step.reward = 0.0` for every step → `trajectory.reward = 0` via
#     `compute_trajectory_reward`. This feeds GRPO's `trajectory_reward`
#     scalar, which becomes `score_batch[i, last_token]` and ultimately
#     `token_level_scores` for GRPO group normalization.
#   * `ep_rewards = [0.0, …]` for the result dict's `episode_rewards`.
#     This feeds chunk_discounted_topr's `compute_chunk_returns_*`,
#     which writes G_k into the per-token advantage tensor.
#
# Both zeroings live in the SAME `if termination_reason in truncation_reasons`
# guard. The two tests below pin the downstream property: given zeroed
# `episode_rewards`, the TOPR chunk-returns helpers return all-zero G_k
# under both `per_episode` and `terminal` scopes — i.e. no spurious
# positive credit can flow through.

def test_topr_per_episode_returns_zero_when_episode_rewards_zeroed():
    """Engine zeros `episode_rewards` for truncated trajectories →
    chunk_discounted_topr with per_episode scope produces zero G_k
    everywhere, matching the zero `trajectory_reward` GRPO sees."""
    from trainers.chunk_advantage import compute_chunk_returns_per_episode

    def _action(done=False):
        return {
            "is_summarization": False, "trigger": None,
            "episode_done": done, "episode_index": 0,
        }

    # Shape: a successful ep1 plus a truncated tail. WITHOUT engine
    # zeroing, ep1 actions would inherit positive credit. With zeroing,
    # all G_k = 0.
    md = [
        _action(False),
        _action(True),    # ep1 ends here (would carry R=1 before zeroing)
        {"is_summarization": True, "trigger": "episode_end",
         "episode_done": False, "episode_index": 0},
        _action(False),
        _action(False),   # ep2 truncated (no episode_done=True)
    ]
    # Simulating engine post-zeroing: [0.0, 0.0] (was [1.0, 0.0]).
    out = compute_chunk_returns_per_episode(md, [0.0, 0.0], gamma=0.9)
    assert all(g == 0.0 for g, _ in out)
    assert all(is_pos is False for _, is_pos in out)


def test_topr_terminal_returns_zero_when_episode_rewards_zeroed():
    """Same property under reward_scope=terminal: G = γ^Δ · sum(R) = 0
    when all episode rewards are zeroed."""
    from trainers.chunk_advantage import compute_chunk_returns_terminal

    md = [{"is_summarization": False, "trigger": None,
           "episode_done": False, "episode_index": 0} for _ in range(5)]
    out = compute_chunk_returns_terminal(md, [0.0, 0.0], gamma=0.9)
    assert all(g == 0.0 for g, _ in out)
    assert all(is_pos is False for _, is_pos in out)
