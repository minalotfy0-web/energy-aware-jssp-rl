import numpy as np
import pytest
import torch as th

from common import make_env, run_episode
from mmas import MixedPolicy, zscore


def test_zscore_ignores_masked_actions():
    logits = th.tensor([2.0, 1.0, 0.0, -1e9])
    mask = th.tensor([True, True, True, False])
    z = zscore(logits, mask)
    assert z[mask].mean().abs() < 1e-6
    assert z[mask].std() == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize("level", ["logits", "probs"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("weight_util", [0.0, 0.5, 1.0])
def test_distribution_is_valid(experts, level, normalize, weight_util):
    """Masked actions get probability 0 and nothing turns NaN, even with a zero weight."""
    env = make_env("sl01", usage_cap=max(weight_util, 0.1))
    obs, _ = env.reset()
    mask = np.asarray(env.action_masks(), dtype=bool)
    policy = MixedPolicy(*experts, weight_util, level=level, normalize=normalize)

    probs = policy.distribution(obs, mask)
    assert not th.isnan(probs).any()
    assert probs.sum() == pytest.approx(1.0, abs=1e-5)
    assert (probs[~th.as_tensor(mask)] == 0).all()


def test_mixed_policy_finishes_an_episode(experts):
    # Wu = 1.0 gives the energy expert weight 0: the case that used to produce NaN and hang
    env = make_env("sl01", usage_cap=1.0)
    policy = MixedPolicy(*experts, weight_util=1.0, level="logits", normalize=True)
    metrics = run_episode(env, policy, max_steps=20_000)
    assert metrics["makespan"] > 0


def test_sampling_is_reproducible(experts):
    env = make_env("sl01", usage_cap=0.5)
    obs, _ = env.reset()
    mask = np.asarray(env.action_masks(), dtype=bool)
    draws = []
    for _ in range(2):
        policy = MixedPolicy(*experts, 0.5, sampling="sample", seed=7, noop_guard=0)
        draws.append([policy(obs, mask) for _ in range(20)])
    assert draws[0] == draws[1]
    assert all(mask[a] for a in draws[0])
