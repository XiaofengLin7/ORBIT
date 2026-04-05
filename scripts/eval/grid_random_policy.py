#!/usr/bin/env python
"""Evaluate a random policy on the GridEnvAdapter.

Usage:
    python scripts/eval/grid_random_policy.py --n 4 --m 4 --episodes 1000 --max-steps 7
"""

from __future__ import annotations

import argparse
import os
import random
import sys

# Allow running the script directly from the repo root.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from envs.grid_env_adapter import GridEnvAdapter


ACTIONS = ["up", "right", "down", "left"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate random policy on GridEnv.")
    parser.add_argument(
        "--episodes",
        type=int,
        default=20,
        help="Number of episodes to run.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=50,
        help="Max policy steps per episode.",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=100,
        help="Env max turns per episode (truncation).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        help="Number of rows.",
    )
    parser.add_argument(
        "--m",
        type=int,
        default=5,
        help="Number of columns.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Base seed for env resets and random actions.",
    )
    parser.add_argument(
        "--print-trajectory",
        action="store_true",
        help="Print per-step observations and actions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rng = random.Random(args.seed)
    env = GridEnvAdapter(
        n=args.n,
        m=args.m,
        max_turns=args.max_turns,
        seed=args.seed,
    )

    successes = 0
    total_reward = 0.0
    total_steps = 0

    for ep in range(1, args.episodes + 1):
        obs, info = env.reset(seed=args.seed + ep)
        if args.print_trajectory:
            print(f"=== Episode {ep} reset ===")
            print(obs)
            # print(f"info: {info}")
            print()

        ep_reward = 0.0
        done = False
        steps = 0

        for _ in range(args.max_steps):
            steps += 1
            action = rng.choice(ACTIONS)
            obs, reward, done, info = env.step(action)
            ep_reward += reward
            if args.print_trajectory:
                print(f"--- Step {steps} | action='{action}' ---")
                print(obs)
                print(f"reward={reward}, done={done}, info={info}")
                print()
            if done:
                break

        success = ep_reward > 0
        successes += int(success)
        total_reward += ep_reward
        total_steps += steps

        if args.print_trajectory:
            print(f"=== Episode {ep} summary ===")
            print(f"steps={steps}, reward={ep_reward}, success={success}")
            print()

    env.close()

    avg_reward = total_reward / max(args.episodes, 1)
    avg_steps = total_steps / max(args.episodes, 1)
    success_rate = successes / max(args.episodes, 1)

    print("=== Random policy evaluation ===")
    print(f"episodes={args.episodes}")
    print(f"max_steps={args.max_steps}")
    print(f"success_rate={success_rate:.3f}")
    print(f"avg_reward={avg_reward:.3f}")
    print(f"avg_steps={avg_steps:.2f}")


if __name__ == "__main__":
    main()
