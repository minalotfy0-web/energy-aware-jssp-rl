"""Compare CPU and CUDA speed for training and for MMAS inference.

Example:
    python scripts/benchmark_device.py --devices cpu cuda --timesteps 16384
"""
import argparse
import time

import torch as th
from sb3_contrib import MaskablePPO

from common import make_env, resolve_device, run_episode
from mmas import MixedPolicy


def time_training(device, timesteps, instance):
    env = make_env(instance, usage_cap=0.5)
    model = MaskablePPO("MlpPolicy", env, learning_rate=3e-4, clip_range=0.2, ent_coef=0.1,
                        n_steps=4096, batch_size=512, seed=101, device=device, verbose=0)
    start = time.perf_counter()
    model.learn(total_timesteps=timesteps)
    return time.perf_counter() - start


def time_mmas(device, energy_path, util_path, instance):
    energy = MaskablePPO.load(energy_path, device=device)
    util = MaskablePPO.load(util_path, device=device)
    env = make_env(instance, usage_cap=0.5)
    policy = MixedPolicy(energy, util, weight_util=0.5, level="logits", normalize=True)
    steps = 0

    def counted(obs, mask):
        nonlocal steps
        steps += 1
        return policy(obs, mask)

    start = time.perf_counter()
    run_episode(env, counted)
    return time.perf_counter() - start, steps


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--timesteps", type=int, default=16384)
    parser.add_argument("--instance", default="sl01")
    parser.add_argument("--energy-expert", default="pretrained/experts/energy_expert_cap0.1.zip")
    parser.add_argument("--util-expert", default="pretrained/experts/utilization_expert_cap1.0.zip")
    args = parser.parse_args()

    print(f"PyTorch {th.__version__}, CUDA available: {th.cuda.is_available()}"
          + (f" ({th.cuda.get_device_name(0)})" if th.cuda.is_available() else ""))
    rows = []
    for name in args.devices:
        device = resolve_device(name)
        train_s = time_training(device, args.timesteps, args.instance)
        infer_s, steps = time_mmas(device, args.energy_expert, args.util_expert, args.instance)
        rows.append((device, train_s, args.timesteps / train_s, infer_s, steps / infer_s))

    print(f"\n{'device':8s} {'train (s)':>10s} {'train steps/s':>14s} {'MMAS episode (s)':>17s} {'MMAS steps/s':>13s}")
    for device, train_s, train_rate, infer_s, infer_rate in rows:
        print(f"{device:8s} {train_s:10.1f} {train_rate:14.0f} {infer_s:17.1f} {infer_rate:13.0f}")


if __name__ == "__main__":
    main()
