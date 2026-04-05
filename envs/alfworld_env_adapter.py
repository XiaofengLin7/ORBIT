"""ALFWorld adapter for rLLM multi-episode wrappers.

Wraps ALFWorld's TextWorld-based environment through the ``BaseEnv`` interface
expected by ``MultiEpisodeEnv``.  Follows the ORBIT prompt style: observations
include admissible actions and the model is expected to respond with
``\\boxed{action}``.

ALFWorld does NOT emit explicit success/failure messages, so the adapter adds
them (e.g. "Congratulations!" on task completion).
"""

from __future__ import annotations

import os
import re
import sys
import threading
import yaml
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from rllm.agents.agent import Action  # type: ignore
from rllm.environments.base.base_env import BaseEnv  # type: ignore

# ---------------------------------------------------------------------------
# ALFWorld imports — resolution order:
#   1. ALFWORLD_VENDOR_PATH env var (explicit override)
#   2. pip-installed alfworld  (pip install alfworld)
#   3. LaMer vendored copy     (legacy fallback)
# ---------------------------------------------------------------------------

def _setup_alfworld_imports() -> None:
    """Add alfworld to sys.path if not already importable."""
    # 1. Explicit vendor path override
    vendor = os.environ.get("ALFWORLD_VENDOR_PATH")
    if vendor and os.path.isdir(vendor):
        if vendor not in sys.path:
            sys.path.insert(0, vendor)
        return

    # 2. Already importable (e.g. pip install alfworld)
    try:
        import alfworld.agents.environment.alfred_tw_env  # noqa: F401
        return
    except ImportError:
        pass

    # 3. Legacy LaMer fallback
    lamer = os.path.normpath(os.path.join(
        os.path.dirname(__file__), "..", "..", "LaMer",
        "agent_system", "environments", "alfworld",
    ))
    if os.path.isdir(lamer) and lamer not in sys.path:
        sys.path.insert(0, lamer)

_setup_alfworld_imports()

import textworld  # type: ignore
import textworld.gym  # type: ignore
from alfworld.agents.environment.alfred_tw_env import (  # type: ignore
    AlfredDemangler,
    AlfredInfos,
    AlfredTWEnv,
    TASK_TYPES,
)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

BOXED_PATTERN = re.compile(r"\\boxed\{([^}]+)\}", re.IGNORECASE)

# Guard TextWorld's global gym registration (not thread-safe).
_TW_REGISTER_LOCK = threading.Lock()

# Class-level cache: (config_path, eval_split) → sorted list of game files.
_GAMEFILE_CACHE: Dict[Tuple[str, str], List[str]] = {}
_CACHE_LOCK = threading.Lock()

# ---------------------------------------------------------------------------
# Monkey-patch tatsu parsers to be thread-local.
#
# TextWorld uses three module-level ``_PARSER`` singletons (tatsu parsers)
# whose ``parse()`` method mutates internal stacks.  Concurrent calls from
# different threads corrupt these stacks (``IndexError: pop from empty
# list``).  We replace each module's ``_parse_and_convert`` function with a
# version that lazily creates a *per-thread* parser, making parallel
# ``reset()`` / ``step()`` safe without a global lock.
# ---------------------------------------------------------------------------

def _install_threadlocal_parsers() -> None:
    """Patch TextWorld's _parse_and_convert functions to use per-thread parsers.

    Each module's ``_PARSER`` is created with ``semantics=…, parseinfo=True``.
    We must replicate those constructor args so that fresh per-thread parsers
    produce the correct model nodes (not raw ``tatsu.ast.AST``).
    """
    import textworld.envs.pddl.logic as _pddl_logic
    import textworld.envs.pddl.textgen as _textgen
    import textworld.logic as _tw_logic

    from textworld.envs.pddl.logic.parser import PddlLogicParser
    from textworld.envs.pddl.logic.model import PddlLogicModelBuilderSemantics
    from textworld.envs.pddl.textgen.parser import CSGParser
    from textworld.envs.pddl.textgen.model import CSGModelBuilderSemantics
    from textworld.logic.parser import GameLogicParser
    from textworld.logic.model import GameLogicModelBuilderSemantics

    def _make_threadsafe(module, parser_factory, walker_cls):
        tls = threading.local()

        def _parse_and_convert(*args, **kwargs):
            try:
                parser = tls.parser
            except AttributeError:
                parser = parser_factory()
                tls.parser = parser
            model = parser.parse(*args, **kwargs)
            return walker_cls().walk(model)

        module._parse_and_convert = _parse_and_convert

    _make_threadsafe(
        _pddl_logic,
        lambda: PddlLogicParser(semantics=PddlLogicModelBuilderSemantics(), parseinfo=True),
        _pddl_logic._ModelConverter,
    )
    _make_threadsafe(
        _textgen,
        lambda: CSGParser(semantics=CSGModelBuilderSemantics(), parseinfo=True),
        _textgen._Converter,
    )
    _make_threadsafe(
        _tw_logic,
        lambda: GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True),
        _tw_logic._ModelConverter,
    )

    # textworld.logic also calls _PARSER.parse directly in parse_document().
    # Patch the module attribute to be a per-thread factory as well.
    _tw_logic_tls = threading.local()

    class _TwLogicParserProxy:
        """Proxy that returns a per-thread GameLogicParser for direct .parse() calls."""
        def __getattr__(self, name):
            try:
                p = _tw_logic_tls.parser
            except AttributeError:
                p = GameLogicParser(semantics=GameLogicModelBuilderSemantics(), parseinfo=True)
                _tw_logic_tls.parser = p
            return getattr(p, name)

    _tw_logic._PARSER = _TwLogicParserProxy()

_install_threadlocal_parsers()

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _collect_game_files(config_path: str, eval_split: str) -> List[str]:
    """Collect and cache the sorted list of game files for a split."""
    key = (config_path, eval_split)
    with _CACHE_LOCK:
        if key in _GAMEFILE_CACHE:
            return _GAMEFILE_CACHE[key]

    config = _load_yaml(config_path)
    base_env = AlfredTWEnv(config, train_eval=eval_split)
    game_files = sorted(base_env.game_files)

    with _CACHE_LOCK:
        _GAMEFILE_CACHE[key] = game_files
    return game_files


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class ALFWorldEnvAdapter(BaseEnv):
    """Adapter exposing ALFWorld via the ``BaseEnv`` contract for ORBIT eval.

    Each instance manages a *single* TextWorld game.  On the first ``reset``
    the game file is selected deterministically from the seed; subsequent
    ``reset`` calls with the same seed replay the same game (multi-episode).
    """

    def __init__(
        self,
        env_id: str = "alfworld",
        env_kwargs: Optional[Dict[str, Any]] = None,
        max_turns: int = 30,
        seed: Optional[int] = None,
        eval_split: str = "eval_in_distribution",
        config_path: Optional[str] = None,
        **_: Any,
    ) -> None:
        super().__init__()

        merged = dict(env_kwargs or {})
        self.env_id = str(merged.get("env_id", env_id))
        self.max_turns = int(merged.get("max_turns", max_turns))
        self._seed = merged.get("seed", seed)
        self.eval_split = str(merged.get("eval_split", eval_split))

        # Resolve config path
        config_path = merged.get("config_path", config_path)
        if config_path is None:
            config_path = str(REPO_ROOT / "configs" / "config_alfworld_tw.yaml")
        if not os.path.isabs(config_path):
            config_path = str(REPO_ROOT / config_path)
        self._config_path = config_path
        self._tw_config = _load_yaml(config_path)

        # Collect game files (cached across instances)
        self._sorted_game_files = _collect_game_files(config_path, self.eval_split)
        if not self._sorted_game_files:
            raise RuntimeError(
                f"No game files found for split={self.eval_split}. "
                f"Check ALFWORLD_DATA and config_path={config_path}."
            )

        # Deferred state — created on first reset()
        self._tw_env = None
        self._gamefile: Optional[str] = None
        self._admissible_commands: List[str] = []
        self._task_description: str = ""
        self.turn: int = 0
        self._done: bool = False

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        task: Optional[dict] = None,
    ) -> Tuple[str, dict]:
        config = self._resolve_reset_config(seed=seed, task=task)
        self._seed = int(config["seed"])
        self.max_turns = int(config["max_turns"])

        gamefile = self._select_gamefile(self._seed)

        if self._gamefile != gamefile or self._tw_env is None:
            # Different game or first time — create new TW env
            self._init_tw_env(gamefile)
            self._gamefile = gamefile

        # Reset (replays the same game since register_games has only this file).
        # Lock needed: fast_downward (called during load/reset) is not thread-safe.
        # step() does NOT need this lock — tatsu parsers are already thread-local.
        with _TW_REGISTER_LOCK:
            obs_list, infos = self._tw_env.reset()
        raw_obs = obs_list[0]
        info = self._unpack_info(infos)

        self._admissible_commands = [
            c for c in info.get("admissible_commands", []) if c != "help"
        ]
        self._task_description = self._extract_task(raw_obs)
        self.turn = 0
        self._done = False

        normalized_info = {
            "env_id": self.env_id,
            "turn": 0,
            "max_turns": self.max_turns,
            "terminated": False,
            "truncated": False,
            "raw_reward": 0.0,
            "gamefile": gamefile,
            "task_type": self._extract_task_type(gamefile),
        }
        return self._format_init_obs(raw_obs), normalized_info

    def step(self, action: Any) -> Tuple[str, float, bool, dict]:
        if self._done:
            raise RuntimeError("Environment is done. Call reset() before step().")

        parsed = self._parse_action(action)
        matched = self._match_admissible(parsed, self._admissible_commands)

        obs_list, scores, dones, infos = self._tw_env.step([matched])
        raw_obs = obs_list[0]
        info = self._unpack_info(infos)

        self.turn += 1
        self._admissible_commands = [
            c for c in info.get("admissible_commands", []) if c != "help"
        ]

        won = bool(info.get("won", False))
        inner_done = bool(dones[0]) if hasattr(dones, "__getitem__") else bool(dones)
        terminated = won or inner_done
        truncated = not terminated and self.turn >= self.max_turns
        done = terminated or truncated
        self._done = done

        reward = 1.0 if won else 0.0

        normalized_info = {
            "env_id": self.env_id,
            "turn": self.turn,
            "max_turns": self.max_turns,
            "terminated": bool(terminated),
            "truncated": truncated,
            "raw_reward": reward,
            "is_correct": won,
            "parsed_action": matched,
            "gamefile": self._gamefile,
        }

        obs_text = self._format_step_obs(raw_obs, won, terminated, truncated)
        return obs_text, reward, done, normalized_info

    def close(self) -> None:
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass
            self._tw_env = None

    @staticmethod
    def from_dict(info: dict) -> "ALFWorldEnvAdapter":
        env_kwargs = dict(info.get("env_kwargs", {}) or {})
        passthrough_keys = (
            "max_turns",
            "seed",
            "eval_split",
            "config_path",
        )
        for key in passthrough_keys:
            if key in info and key not in env_kwargs:
                env_kwargs[key] = info[key]
        return ALFWorldEnvAdapter(
            env_id=info.get("env_id", "alfworld"),
            env_kwargs=env_kwargs,
        )

    @staticmethod
    def is_multithread_safe() -> bool:
        return False

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _select_gamefile(self, seed: int) -> str:
        idx = seed % len(self._sorted_game_files)
        return self._sorted_game_files[idx]

    def _init_tw_env(self, gamefile: str) -> None:
        """Create a TextWorld gym env for a single game file."""
        if self._tw_env is not None:
            try:
                self._tw_env.close()
            except Exception:
                pass

        request_infos = textworld.EnvInfos(
            won=True,
            admissible_commands=True,
            extras=["gamefile"],
        )
        wrappers = [AlfredDemangler(shuffle=False), AlfredInfos]
        max_steps = self._tw_config["dagger"]["training"]["max_nb_steps_per_episode"]

        with _TW_REGISTER_LOCK:
            env_id = textworld.gym.register_games(
                [gamefile],
                request_infos,
                batch_size=1,
                asynchronous=False,
                max_episode_steps=max_steps,
                wrappers=wrappers,
            )
            self._tw_env = textworld.gym.make(env_id)

    def _resolve_reset_config(
        self,
        seed: Optional[int],
        task: Optional[dict],
    ) -> Dict[str, Any]:
        config: Dict[str, Any] = {
            "seed": self._seed if self._seed is not None else 0,
            "max_turns": self.max_turns,
        }
        if isinstance(task, dict):
            if "seed" in task and task["seed"] is not None:
                config["seed"] = int(task["seed"])
            if "max_turns" in task and task["max_turns"] is not None:
                config["max_turns"] = int(task["max_turns"])
            elif "max_turns_per_episode" in task and task["max_turns_per_episode"] is not None:
                config["max_turns"] = int(task["max_turns_per_episode"])
        if seed is not None:
            config["seed"] = int(seed)
        return config

    @staticmethod
    def _unpack_info(infos: dict) -> dict:
        """Unpack batch-1 info dict (ALFWorld returns lists of length 1)."""
        out: Dict[str, Any] = {}
        for k, v in infos.items():
            if isinstance(v, (list, tuple)) and len(v) == 1:
                out[k] = v[0]
            else:
                out[k] = v
        return out

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    @classmethod
    def _parse_action(cls, action: Any) -> str:
        """Extract action text from ``\\boxed{...}`` wrapper."""
        if isinstance(action, Action):
            action = action.action
        raw = str(action).strip()
        matches = list(BOXED_PATTERN.finditer(raw))
        if matches:
            return matches[-1].group(1).strip()
        return raw.strip()

    @staticmethod
    def _match_admissible(parsed: str, admissible: List[str]) -> str:
        """Match parsed action against admissible commands.

        Strategy: exact → prefix → substring → fuzzy → fallback.
        """
        if not admissible:
            return "look"

        parsed_lower = parsed.lower().strip()

        # Exact match (case-insensitive)
        for cmd in admissible:
            if cmd.lower().strip() == parsed_lower:
                return cmd

        # Prefix match
        for cmd in admissible:
            if cmd.lower().startswith(parsed_lower):
                return cmd

        # Substring containment
        for cmd in admissible:
            if parsed_lower in cmd.lower():
                return cmd

        # Fuzzy match via SequenceMatcher
        best_ratio, best_cmd = 0.0, None
        for cmd in admissible:
            ratio = SequenceMatcher(None, parsed_lower, cmd.lower()).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_cmd = cmd
        if best_ratio > 0.6 and best_cmd is not None:
            return best_cmd

        # Fallback: "look" if available, else first command
        for cmd in admissible:
            if cmd.lower() == "look":
                return cmd
        return admissible[0]

    # ------------------------------------------------------------------
    # Observation formatting (ORBIT style)
    # ------------------------------------------------------------------

    def _format_init_obs(self, raw_obs: str) -> str:
        admissible_str = ", ".join(
            f"'{cmd}'" for cmd in self._admissible_commands
        )
        return (
            f"You are in an interactive household environment.\n"
            f"Your task is to: {self._task_description}\n\n"
            f"{raw_obs}\n\n"
            f"Admissible actions: [{admissible_str}]\n\n"
            f"Choose one admissible action and output it inside \\boxed{{}}."
        )

    def _format_step_obs(
        self, raw_obs: str, won: bool, terminated: bool, truncated: bool
    ) -> str:
        if won:
            return (
                f"Observation: {raw_obs}\n"
                f"Congratulations! You have completed the task successfully."
            )
        if terminated and not truncated:
            return (
                f"Observation: {raw_obs}\n"
                f"Episode finished. The task was not completed."
            )
        if truncated:
            return (
                f"Observation: {raw_obs}\n"
                f"Episode stopped because max_turns ({self.max_turns}) was reached."
            )
        admissible_str = ", ".join(
            f"'{cmd}'" for cmd in self._admissible_commands
        )
        return (
            f"Observation: {raw_obs}\n\n"
            f"Admissible actions: [{admissible_str}]\n\n"
            f"Choose one admissible action and output it inside \\boxed{{}}."
        )

    # ------------------------------------------------------------------
    # Task extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_task(text_obs: str) -> str:
        """Extract task description from ALFWorld initial observation."""
        marker = "Your task is to: "
        idx = text_obs.find(marker)
        if idx != -1:
            # Task description runs to end of line
            rest = text_obs[idx + len(marker) :]
            newline = rest.find("\n")
            if newline != -1:
                return rest[:newline].strip()
            return rest.strip()
        return text_obs.strip()

    @staticmethod
    def _extract_task_type(gamefile: str) -> str:
        """Extract task type from gamefile path."""
        task_type_names = list(TASK_TYPES.values())
        parts = gamefile.replace("\\", "/").split("/")
        for part in parts:
            for tname in task_type_names:
                if part.startswith(tname):
                    return tname
        return "unknown"
