import numpy as np
import pytest

from common import make_env, run_episode


@pytest.fixture
def env():
    return make_env("sl01", usage_cap=0.5)


def test_reset_matches_spaces(env):
    obs, _ = env.reset()
    assert obs.shape == env.observation_space.shape
    assert env.action_space.n == 31  # 15 jobs + 15 machines + standby


def test_mask_has_valid_actions(env):
    env.reset()
    mask = np.asarray(env.action_masks(), dtype=bool)
    assert mask.shape == (env.action_space.n,)
    assert mask.any()


def test_random_valid_policy_finishes(env):
    rng = np.random.default_rng(0)
    metrics = run_episode(env, lambda obs, mask: int(rng.choice(np.flatnonzero(mask))), max_steps=50_000)

    assert metrics["makespan"] > 0
    assert 0 <= metrics["machine_utilization"] <= 1
    assert metrics["idleness"] == pytest.approx(1 - metrics["machine_utilization"])
    assert 0 <= metrics["peak_power_p95"] <= metrics["total_power"]


@pytest.mark.parametrize("scalarization", ["linear", "chebyshev"])
def test_mixed_reward_is_finite(scalarization):
    env = make_env("sl01", usage_cap=0.5, scalarization=scalarization)
    obs, _ = env.reset()
    for _ in range(50):
        mask = np.asarray(env.action_masks(), dtype=bool)
        obs, reward, terminated, _, _ = env.step(int(np.flatnonzero(mask)[0]))
        assert np.isfinite(reward)
        if terminated:
            break


def test_unknown_scalarization_is_rejected():
    with pytest.raises(ValueError):
        make_env("sl01", scalarization="nonsense")
