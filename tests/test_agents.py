import torch as th
from conftest import PRETRAINED
from sb3_contrib import MaskablePPO

from common import make_env, run_episode
from evaluate import cap_from_name


def test_cap_is_read_from_file_name():
    assert cap_from_name("pretrained/scalarized/sl01_MX_cap0.7.zip") == 0.7


def test_pretrained_agents_match_the_environment():
    env = make_env("sl01")
    for path in sorted(PRETRAINED.rglob("*.zip")):
        model = MaskablePPO.load(path, device="cpu")
        assert model.observation_space.shape == env.observation_space.shape, path.name
        assert model.action_space.n == env.action_space.n, path.name


def test_scalarized_agent_finishes_an_episode():
    th.manual_seed(0)
    model = MaskablePPO.load(PRETRAINED / "scalarized" / "sl01_MX_cap1.0.zip", device="cpu")
    env = make_env("sl01", usage_cap=1.0)

    def select_action(obs, mask):
        action, _ = model.predict(obs, action_masks=mask, deterministic=False)
        return int(action)

    metrics = run_episode(env, select_action, max_steps=20_000)
    # the utilization expert should keep machines busy and finish fast
    assert metrics["machine_utilization"] > 0.3
    assert metrics["makespan"] < 5_000
