"""Shared helpers for the experiment scripts: env creation, episode rollout and metrics."""
import statistics

import gymnasium as gym
import numpy as np
import torch as th

import petrirl  # noqa: F401  (registers petrirl-moo-v0)

DEVICES = ["auto", "cpu", "cuda"]


def resolve_device(device="auto"):
    """'auto' picks CUDA when available; asking for 'cuda' without a CUDA build of PyTorch is an error."""
    if device == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if device == "cuda" and not th.cuda.is_available():
        raise SystemExit("CUDA requested but not available: install a CUDA build of PyTorch "
                         "(see README) or use --device cpu")
    return device


def make_env(instance="sl01", usage_cap=0.1, benchmark="Solax", reward_f="MX",
             scalarization="linear", render_mode=None):
    return gym.make("petrirl-moo-v0",
                    benchmark=benchmark,
                    instance_id=instance,
                    usage_cap=usage_cap,
                    reward_f=reward_f,
                    scalarization=scalarization,
                    render_mode=render_mode).unwrapped


def episode_metrics(env):
    """Metrics reported in the thesis, computed from the simulator logs after an episode."""
    sim = env.sim
    average_energy = statistics.mean(sim.logs["energy_consumption"])
    return {
        "makespan": int(sim.clock),
        "average_energy": float(average_energy),
        # high-percentile power draw: what an energy cap actually limits
        "peak_power_p95": float(np.percentile(sim.logs["energy_consumption"], 95)),
        "total_power": int(sim.machines_powers.sum()),
        "energy_utilization": float(average_energy / sim.machines_powers.sum()),
        "machine_utilization": float(statistics.mean(sim.logs["machine_utilization"])),
        "idleness": float(1 - statistics.mean(sim.logs["machine_utilization"])),
        # relative deviation of consumption from the cap: >0 means above the cap
        "energy_overshoot": float(statistics.mean(sim.logs["energy_overshoot"])),
    }


def run_episode(env, select_action, render=False, max_steps=100_000):
    """Roll out one episode; select_action(obs, mask) -> int."""
    obs, _ = env.reset()
    terminated, steps = False, 0
    while not terminated:
        mask = np.asarray(env.action_masks(), dtype=bool)
        action = select_action(obs, mask)
        obs, _, terminated, _, _ = env.step(action)
        steps += 1
        if steps >= max_steps:
            raise RuntimeError(f"Episode did not finish within {max_steps} steps")
    if render:
        env.render()
    return episode_metrics(env)
