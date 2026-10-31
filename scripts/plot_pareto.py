"""Plot Pareto fronts of the evaluated methods and report their hypervolume (both objectives minimized).

Each point is one preference weight, averaged over the evaluated instances.

The default objectives are makespan vs. peak power (95th percentile, as a fraction of total machine power):
the conflict an energy cap creates. Average energy vs. idleness is *not* a useful pair in this environment,
because machines only draw power while busy, so the two are almost perfectly correlated.

Example:
    python scripts/plot_pareto.py results/scalarization.csv results/mmas_logits_norm.csv
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "peak_power_frac": "Peak power, 95th percentile (fraction of total machine power)",
    "makespan": "Makespan (time steps)",
    "average_energy": "Average power draw (power units)",
    "idleness": "Average machine idleness",
}


def non_dominated(points):
    """Pareto-optimal subset of (x, y) points for a minimization problem, sorted by x."""
    front = [p for p in points
             if not any(q[0] <= p[0] and q[1] <= p[1] and q != p for q in points)]
    return sorted(set(front))


def hypervolume(front, reference):
    """2-D hypervolume dominated by a minimization front up to the reference point."""
    volume, prev_y = 0.0, reference[1]
    for x, y in front:
        if x < reference[0] and y < prev_y:
            volume += (reference[0] - x) * (prev_y - y)
            prev_y = y
    return volume


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("csv", nargs="+")
    parser.add_argument("--x", default="peak_power_frac", choices=list(LABELS))
    parser.add_argument("--y", default="makespan", choices=list(LABELS))
    parser.add_argument("--out", default="docs/figures/pareto_front.png")
    args = parser.parse_args()

    df = pd.concat([pd.read_csv(path) for path in args.csv], ignore_index=True)
    df["peak_power_frac"] = df["peak_power_p95"] / df["total_power"]
    points_df = df.groupby(["method", "weight_util"], sort=False)[[args.x, args.y]].mean().reset_index()

    # Hypervolume on min-max normalized objectives, with a shared reference point, so methods are comparable
    lo, hi = points_df[[args.x, args.y]].min(), points_df[[args.x, args.y]].max()
    normalized = (points_df[[args.x, args.y]] - lo) / (hi - lo)
    points_df[["nx", "ny"]] = normalized.to_numpy()
    reference = (1.1, 1.1)

    fig, ax = plt.subplots(figsize=(9, 6))
    print(f"{'method':28s} {'front':>5s} {'hypervolume':>12s}")
    for method, group in points_df.groupby("method", sort=False):
        front = non_dominated(list(group[[args.x, args.y]].itertuples(index=False, name=None)))
        hv = hypervolume(non_dominated(list(group[["nx", "ny"]].itertuples(index=False, name=None))), reference)
        scatter = ax.scatter(group[args.x], group[args.y], s=30, alpha=0.4)
        color = scatter.get_facecolor()[0][:3]
        ax.plot(*zip(*front), "o-", color=color, label=f"{method} (HV={hv:.3f})")
        for _, row in group.iterrows():
            ax.annotate(f"{row['weight_util']:.1f}", (row[args.x], row[args.y]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points", color=color)
        print(f"{method:28s} {len(front):5d} {hv:12.3f}")

    ax.set_xlabel(LABELS[args.x])
    ax.set_ylabel(LABELS[args.y])
    ax.set_title("Pareto fronts (labels: utilization weight Wu, mean over instances)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
