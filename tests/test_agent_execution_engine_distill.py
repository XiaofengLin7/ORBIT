from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any
import sys

_ROOT = Path(__file__).resolve().parents[1]
for _extra_path in (_ROOT, _ROOT / "third_party" / "rllm"):
    _path_str = str(_extra_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from rllm.agents.agent import Action, Step, Trajectory
from rllm.engine.agent_execution_engine import AgentExecutionEngine
from rllm.engine.rollout.rollout_engine import ModelOutput


class _DummyTokenizer:
    pad_token_id = 0

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [max(1, ord(ch) % 31) for ch in text]


class _DummyParser:
    assistant_token = "<assistant>"

    def parse(
        self,
        messages: list[dict[str, str]],
        add_generation_prompt: bool = False,
        is_first_msg: bool = False,
    ) -> str:
        out = []
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "assistant":
                out.append(f"{self.assistant_token}{content}")
            else:
                out.append(f"{role}:{content}")
        text = "\n".join(out)
        if add_generation_prompt:
            text = f"{text}{self.assistant_token}"
        return text


class _DummyAgent:
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

    def update_from_env(self, observation: Any, reward: float, done: bool, info: dict, **kwargs) -> None:
        if not self._trajectory.steps:
            self._messages.append({"role": "user", "content": str(observation)})

    def update_from_model(self, response: str, **kwargs) -> Action:
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


class _DummyEnv:
    def __init__(self) -> None:
        self.idx = 0
        self._step = 0

    def reset(self) -> tuple[str, dict[str, Any]]:
        self._step = 0
        return "obs0", {"episode_index": 0}

    def step(self, action: Any) -> tuple[str, float, bool, dict[str, Any]]:
        self._step += 1
        if self._step == 1:
            return "obs1", 0.0, False, {"episode_index": 1}
        return "obs2", 1.0, True, {"episode_index": 1}

    def close(self) -> None:
        return


def _make_engine(filter_token_mismatch: bool = False) -> AgentExecutionEngine:
    engine = AgentExecutionEngine.__new__(AgentExecutionEngine)
    engine.agents = [_DummyAgent()]
    engine.envs = [_DummyEnv()]
    engine.executor = ThreadPoolExecutor(max_workers=1)
    engine.max_steps = 2
    engine.max_response_length = 128
    engine.max_prompt_length = 128
    engine.enforce_max_prompt_length = False
    engine.engine_name = "verl"
    engine.tokenizer = _DummyTokenizer()
    engine.chat_parser = _DummyParser()
    engine.config = SimpleNamespace(rllm=SimpleNamespace(filter_token_mismatch=filter_token_mismatch))
    engine.overlong_filter = False
    engine.trajectory_timeout = 30
    engine.gamma = 0.9
    engine.sampling_params = {}
    engine.disable_thinking = False
    return engine


def test_assemble_steps_first_attempt_mask_alignment() -> None:
    engine = _make_engine(filter_token_mismatch=False)
    steps = [
        {"prompt_ids": [1, 2], "completion_ids": [7, 8], "episode_index": 0},
        {"prompt_ids": [1, 2, 7, 8, 9], "completion_ids": [10], "episode_index": 1},
    ]
    try:
        _, response_tokens, response_masks, first_attempt, is_valid = engine.assemble_steps(steps)
    finally:
        engine.executor.shutdown(wait=False, cancel_futures=True)

    assert is_valid is True
    assert response_tokens.tolist() == [7, 8, 9, 10]
    assert response_masks.tolist() == [1, 1, 0, 1]
    assert first_attempt.tolist() == [1, 1, 0, 0]


def test_assemble_steps_mismatch_zeroes_masks_when_enabled() -> None:
    engine = _make_engine(filter_token_mismatch=True)
    steps = [
        {"prompt_ids": [1, 2], "completion_ids": [7, 8], "episode_index": 0},
        {"prompt_ids": [1, 9, 9], "completion_ids": [10], "episode_index": 1},
    ]
    try:
        _, response_tokens, response_masks, first_attempt, is_valid = engine.assemble_steps(steps)
    finally:
        engine.executor.shutdown(wait=False, cancel_futures=True)

    assert is_valid is False
    assert response_tokens.tolist() == [7, 8]
    assert response_masks.tolist() == [0, 0]
    assert first_attempt.tolist() == [0, 0]


def test_token_mode_returns_step_records_with_episode_indices() -> None:
    engine = _make_engine(filter_token_mismatch=False)
    scripted_outputs = [
        ModelOutput(text="a0", prompt_ids=[1, 2], completion_ids=[3], logprobs=[-0.1]),
        ModelOutput(text="a1", prompt_ids=[1, 2, 3], completion_ids=[4], logprobs=[-0.2]),
    ]

    async def _fake_get_model_response(prompt_messages: Any, application_id: str, **kwargs: Any) -> ModelOutput:
        return scripted_outputs.pop(0)

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

    step_records = token_result["step_records"]
    assert len(step_records) == 2
    assert [int(step["episode_index"]) for step in step_records] == [0, 1]
    assert "first_attempt_response_mask" in token_result
    assert token_result["first_attempt_response_mask"].tolist() == [1, 0]
