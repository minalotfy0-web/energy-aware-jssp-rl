"""Evaluate reward-scalarized agents (one trained agent per preference weight).

Example:
    python scripts/evaluate.py --agents pretrained/scalarized/*.zip --instances sl01 sl02 sl03 sl04
"""
import argparse
import glob
import os
import re

import pandas as pd
import torch as th
from sb3_contrib import MaskablePPO

from common import DEVICES, make_env, resolve_device, run_episode


def cap_from_name(path):
    match = re.search(r"cap([0-9.]+)\.zip$", os.path.basename(path))
    if not match:
        raise ValueError(f"Cannot read the energy cap from '{path}', expected '...cap<value>.zip'")
    return float(match.group(1))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--agents", nargs="+", default=["pretrained/scalarized/*.zip"])
    parser.add_argument("--instances", nargs="+", default=["sl01", "sl02", "sl03", "sl04"])
    parser.add_argument("--benchmark", default="Solax")
    parser.add_argument("--deterministic", action="store_true", help="greedy actions instead of sampling")
    parser.add_argument("--render", action="store_true", help="save a Gantt + energy plot per episode")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=DEVICES, help="'auto' uses CUDA when available")
    parser.add_argument("--out", default="results/scalarization.csv")
    args = parser.parse_args()
    th.manual_seed(args.seed)
    device = resolve_device(args.device)

    paths = sorted(p for pattern in args.agents for p in glob.glob(pattern))
    if not paths:
        raise SystemExit(f"No agents found for {args.agents}")

    rows = []
    for path in paths:
        cap = cap_from_name(path)
        model = MaskablePPO.load(path, device=device)
        for instance in args.instances:
            env = make_env(instance, cap, args.benchmark, render_mode="solution" if args.render else None)

            def select_action(obs, mask):
                action, _ = model.predict(obs, action_masks=mask, deterministic=args.deterministic)
                return int(action)

            metrics = run_episode(env, select_action, render=args.render)
            rows.append({"method": "Reward scalarization", "weight_util": cap, "instance": instance, **metrics})
            print(f"cap={cap:.1f} {instance}: makespan={metrics['makespan']}, "
                  f"utilization={metrics['machine_utilization']:.3f}, overshoot={metrics['energy_overshoot']:.3f}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
