"""
tests/test_dataset.py
---------------------
Unit tests for Phase 2: sim/dataset.py
"""
import pathlib
import numpy as np
import yaml
import pytest
from sim.env import SwarmEnv
from sim.dataset import rollout, generate_dataset, load_dataset


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    path = pathlib.Path("configs") / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tiny_cfg(cfg):
    """Tiny config for fast tests: N=10, T=5, 0 episodes needed."""
    import copy
    c = copy.deepcopy(cfg)
    c["swarm"]["N"] = 10
    c["swarm"]["rollout_horizon"] = 5
    return c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_rollout_shape(tiny_cfg):
    """rollout() must return (T, N, 4) float32 array."""
    N = tiny_cfg["swarm"]["N"]
    T = tiny_cfg["swarm"]["rollout_horizon"]
    env = SwarmEnv(tiny_cfg, seed=0)
    traj = rollout(env, T)
    assert traj.shape == (T, N, 4), f"Expected ({T},{N},4), got {traj.shape}"
    assert traj.dtype == np.float32


def test_rollout_values_in_range(tiny_cfg):
    """Position columns must stay within [0, world_size]; speed within max_speed."""
    N = tiny_cfg["swarm"]["N"]
    T = tiny_cfg["swarm"]["rollout_horizon"]
    world = tiny_cfg["swarm"]["world_size"]
    max_sp = tiny_cfg["physics"]["max_speed"]
    env = SwarmEnv(tiny_cfg, seed=1)
    traj = rollout(env, T)
    pos = traj[:, :, :2]
    vel = traj[:, :, 2:]
    assert np.all(pos >= 0.0) and np.all(pos < world + 1e-6), "Position out of arena"
    speeds = np.linalg.norm(vel, axis=2)
    assert np.all(speeds <= max_sp + 1e-5), f"Speed {speeds.max():.3f} > {max_sp}"


def test_generate_saves_files(tiny_cfg, tmp_path):
    """generate_dataset() must create train.npz and val.npz."""
    paths = generate_dataset(tiny_cfg, n_episodes=5, val_fraction=0.4,
                             out_dir=str(tmp_path))
    assert paths["train"].exists(), "train.npz not created"
    assert paths["val"].exists(),   "val.npz not created"


def test_dataset_shape(tiny_cfg, tmp_path):
    """Saved arrays must have shape (E, T, N, 4)."""
    N = tiny_cfg["swarm"]["N"]
    T = tiny_cfg["swarm"]["rollout_horizon"]
    n_ep = 10
    val_frac = 0.2
    n_val   = max(1, int(round(n_ep * val_frac)))
    n_train = n_ep - n_val

    generate_dataset(tiny_cfg, n_episodes=n_ep, val_fraction=val_frac,
                     out_dir=str(tmp_path))

    train = load_dataset("train", out_dir=str(tmp_path))
    val   = load_dataset("val",   out_dir=str(tmp_path))

    assert train["episodes"].shape == (n_train, T, N, 4), (
        f"train shape {train['episodes'].shape}"
    )
    assert val["episodes"].shape == (n_val, T, N, 4), (
        f"val shape {val['episodes'].shape}"
    )


def test_reproducibility(tiny_cfg, tmp_path):
    """Same seed_offset must produce identical datasets."""
    kw = dict(n_episodes=4, val_fraction=0.25, out_dir=str(tmp_path), seed_offset=99)
    generate_dataset(tiny_cfg, **kw)
    ep1 = load_dataset("train", out_dir=str(tmp_path))["episodes"].copy()

    import shutil
    shutil.rmtree(tmp_path); tmp_path.mkdir()
    generate_dataset(tiny_cfg, **kw)
    ep2 = load_dataset("train", out_dir=str(tmp_path))["episodes"]

    np.testing.assert_array_equal(ep1, ep2, err_msg="Datasets differ with same seed")


def test_train_val_no_overlap(tiny_cfg, tmp_path):
    """Train and val sets must not share episode content."""
    n_ep = 6
    generate_dataset(tiny_cfg, n_episodes=n_ep, val_fraction=0.33,
                     out_dir=str(tmp_path), seed_offset=0)
    train = load_dataset("train", out_dir=str(tmp_path))["episodes"]
    val   = load_dataset("val",   out_dir=str(tmp_path))["episodes"]
    # Episodes have unique seeds so first frames must differ from all val eps
    for ti in range(len(train)):
        for vi in range(len(val)):
            if np.allclose(train[ti], val[vi]):
                pytest.fail(f"Train ep {ti} == Val ep {vi}: overlap detected")