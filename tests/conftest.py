import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

PRETRAINED = ROOT / "pretrained"


@pytest.fixture(scope="session")
def experts():
    from sb3_contrib import MaskablePPO

    energy = MaskablePPO.load(PRETRAINED / "experts" / "energy_expert_cap0.1.zip", device="cpu")
    util = MaskablePPO.load(PRETRAINED / "experts" / "utilization_expert_cap1.0.zip", device="cpu")
    return energy, util
