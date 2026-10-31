"""Train a MaskablePPO agent with a scalarized energy/utilization reward.

Example (the two MMAS experts from the thesis):
    python scripts/train.py --usage-cap 0.1    # energy expert
    python scripts/train.py --usage-cap 1.0    # utilization expert
"""
import argparse
import os
import time

from sb3_contrib import MaskablePPO

from common import DEVICES, make_env, resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--instance", default="sl01")
    parser.add_argument("--benchmark", default="Solax")
    parser.add_argument("--usage-cap", type=float, default=0.1,
                        help="energy cap as a fraction of total machine power (= utilization weight Wu)")
    parser.add_argument("--reward", default="MX")
    parser.add_argument("--scalarization", default="linear", choices=["linear", "chebyshev"])
    parser.add_argument("--timesteps", type=float, default=2e6)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="auto", choices=DEVICES,
                        help="'auto' uses CUDA when available")
    parser.add_argument("--out", default="agents")
    parser.add_argument("--logdir", default="logs")
    args = parser.parse_args()

    env = make_env(args.instance, args.usage_cap, args.benchmark, args.reward, args.scalarization)
    name = f"{args.instance}_{args.reward}_{args.scalarization}_cap{args.usage_cap}"

    model = MaskablePPO("MlpPolicy", env,
                        learning_rate=3e-4,
                        clip_range=0.2,
                        ent_coef=0.1,
                        n_steps=4096,
                        batch_size=512,
                        seed=args.seed,
                        device=resolve_device(args.device),
                        verbose=1,
                        tensorboard_log=os.path.join(args.logdir, name))

    start = time.time()
    model.learn(total_timesteps=int(args.timesteps))
    elapsed = time.time() - start

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{name}.zip")
    model.save(path)
    print(f"Saved {path} after {elapsed / 60:.1f} min on {model.device}")


if __name__ == "__main__":
    main()
