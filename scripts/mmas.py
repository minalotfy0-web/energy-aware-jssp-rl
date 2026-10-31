"""MMAS - Mixed Multi-Agent Strategies.

Combine two pretrained single-preference experts at inference time instead of training one agent per
preference weight:
    logits level: softmax(We * z_energy + Wu * z_util)        (z = optionally z-score normalized logits)
    probs  level: We * softmax(z_energy) + Wu * softmax(z_util)

Example:
    python scripts/mmas.py --level logits --normalize --weights 0.1 0.3 0.5 0.7 1.0
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch as th
from sb3_contrib import MaskablePPO

from common import DEVICES, make_env, resolve_device, run_episode


def policy_logits(model, obs):
    """Raw (unmasked) action logits of a MaskablePPO MlpPolicy."""
    policy = model.policy
    with th.no_grad():
        obs_t, _ = policy.obs_to_tensor(obs)
        features = policy.extract_features(obs_t, policy.pi_features_extractor)
        latent_pi = policy.mlp_extractor.forward_actor(features)
        return policy.action_net(latent_pi).squeeze(0)


def zscore(logits, mask):
    """Normalize logits using statistics of the valid actions only, so masked actions don't skew them."""
    valid = logits[mask]
    if valid.numel() < 2:
        return logits - valid.mean()
    return (logits - valid.mean()) / (valid.std() + 1e-8)


class MixedPolicy:
    def __init__(self, energy_model, util_model, weight_util, level="logits", normalize=True,
                 sampling="argmax", noop_guard=30, seed=0):
        self.energy_model = energy_model
        self.util_model = util_model
        self.weight_util = weight_util
        self.weight_energy = 1 - weight_util
        self.level = level
        self.normalize = normalize
        self.sampling = sampling
        self.noop_guard = noop_guard
        self.generator = th.Generator().manual_seed(seed)
        self.noop_streak = 0

    def distribution(self, obs, mask):
        logits = [policy_logits(m, obs) for m in (self.energy_model, self.util_model)]
        mask_t = th.as_tensor(mask, dtype=th.bool, device=logits[0].device)
        if self.normalize:
            logits = [zscore(z, mask_t) for z in logits]
        energy_z, util_z = logits

        # Mask after weighting: a zero weight times a -inf masked logit would give NaN
        if self.level == "logits":
            mixed = self.weight_energy * energy_z + self.weight_util * util_z
            return th.softmax(mixed.masked_fill(~mask_t, -th.inf), dim=-1)
        energy_p, util_p = (th.softmax(z.masked_fill(~mask_t, -th.inf), dim=-1) for z in (energy_z, util_z))
        return self.weight_energy * energy_p + self.weight_util * util_p

    def __call__(self, obs, mask):
        probs = self.distribution(obs, mask).cpu()
        if self.sampling == "sample":
            action = int(th.multinomial(probs, 1, generator=self.generator))
        else:
            action = int(th.argmax(probs))

        # The standby ("do nothing") transition is the last action. A deterministic mix can get stuck
        # choosing it forever, so after `noop_guard` consecutive standbys take the best other valid action.
        noop = len(mask) - 1
        self.noop_streak = self.noop_streak + 1 if action == noop else 0
        if self.noop_guard and self.noop_streak >= self.noop_guard and mask[:noop].any():
            probs = probs.clone()
            probs[noop] = 0
            action = int(th.argmax(probs))
            self.noop_streak = 0

        if not mask[action]:
            raise RuntimeError(f"Mixed policy chose invalid action {action} (probabilities: {probs})")
        return action


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--energy-expert", default="pretrained/experts/energy_expert_cap0.1.zip")
    parser.add_argument("--util-expert", default="pretrained/experts/utilization_expert_cap1.0.zip")
    parser.add_argument("--level", default="logits", choices=["logits", "probs"])
    parser.add_argument("--normalize", action="store_true", help="z-score each expert's logits before mixing")
    parser.add_argument("--weights", nargs="+", type=float, default=[0.1, 0.3, 0.5, 0.7, 1.0],
                        help="utilization weights Wu (We = 1 - Wu); also used as the energy cap")
    parser.add_argument("--instances", nargs="+", default=["sl01", "sl02", "sl03", "sl04"])
    parser.add_argument("--benchmark", default="Solax")
    parser.add_argument("--sampling", default="argmax", choices=["argmax", "sample"])
    parser.add_argument("--noop-guard", type=int, default=30, help="0 disables the standby guard")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=DEVICES, help="'auto' uses CUDA when available")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    device = resolve_device(args.device)
    energy_model = MaskablePPO.load(args.energy_expert, device=device)
    util_model = MaskablePPO.load(args.util_expert, device=device)
    label = f"MMAS {args.level}" + (" (normalized)" if args.normalize else "")

    rows = []
    for weight_util in args.weights:
        for instance in args.instances:
            env = make_env(instance, weight_util, args.benchmark, render_mode="solution" if args.render else None)
            policy = MixedPolicy(energy_model, util_model, weight_util, args.level, args.normalize,
                                 args.sampling, args.noop_guard, args.seed)
            metrics = run_episode(env, policy, render=args.render)
            rows.append({"method": label, "weight_util": weight_util, "instance": instance, **metrics})
            print(f"Wu={weight_util:.1f} {instance}: makespan={metrics['makespan']}, "
                  f"utilization={metrics['machine_utilization']:.3f}, overshoot={metrics['energy_overshoot']:.3f}")

    out = args.out or f"results/mmas_{args.level}{'_norm' if args.normalize else ''}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
