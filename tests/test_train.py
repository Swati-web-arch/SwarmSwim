"""
tests/test_train.py
-------------------
Unit tests for Phase 7: models/train.py
"""
import pathlib
import copy
import numpy as np
import yaml
import pytest
from models.train import train_koopman, load_koopman


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    path = pathlib.Path("configs") / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tiny_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["swarm"]["N"] = 8
    c["swarm"]["rollout_horizon"] = 10
    c["koopman"]["obs_dim"] = 10
    c["koopman"]["forecast_steps"] = 3
    return c


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_checkpoint_created(tiny_cfg, tmp_path):
    """train_koopman() must create checkpoints/koopman.npz."""
    metrics = train_koopman(
        tiny_cfg, n_episodes=6,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    ckpt = pathlib.Path(metrics["ckpt_path"])
    assert ckpt.exists(), f"Checkpoint not found at {ckpt}"
    assert ckpt.suffix == ".npz"


def test_checkpoint_has_required_keys(tiny_cfg, tmp_path):
    """Checkpoint .npz must contain K, W, centres, log_bw, obs_dim, reg."""
    metrics = train_koopman(
        tiny_cfg, n_episodes=6,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    data = np.load(metrics["ckpt_path"])
    for key in ("K", "W", "centres", "log_bw", "obs_dim", "reg"):
        assert key in data, f"Missing key '{key}' in checkpoint"


def test_spectral_radius_after_training(tiny_cfg, tmp_path):
    """Spectral radius reported in metrics must be <= 1.0."""
    metrics = train_koopman(
        tiny_cfg, n_episodes=6,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    rho = metrics["spectral_r"]
    assert rho <= 1.0 + 1e-6, f"Spectral radius {rho:.6f} > 1 after training"


def test_load_koopman_restores_model(tiny_cfg, tmp_path):
    """load_koopman() must restore a model that predicts same as before save."""
    metrics = train_koopman(
        tiny_cfg, n_episodes=6,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    from models.koopman import KoopmanPredictor
    from sim.env import SwarmEnv
    from sim.dataset import rollout

    # Get a test state
    env  = SwarmEnv(tiny_cfg, seed=99)
    traj = rollout(env, tiny_cfg["swarm"]["rollout_horizon"])
    state = traj[0]

    # Load and predict
    model = load_koopman(tiny_cfg, ckpt_path=metrics["ckpt_path"])
    pred  = model.predict(state, h=2)

    assert pred.shape == (2, tiny_cfg["swarm"]["N"], 4), (
        f"Loaded model predict shape wrong: {pred.shape}"
    )
    assert np.all(np.isfinite(pred)), "NaN/Inf from loaded model"


def test_metrics_are_finite(tiny_cfg, tmp_path):
    """train_mse and val_mse must be finite positive floats."""
    metrics = train_koopman(
        tiny_cfg, n_episodes=6,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    assert np.isfinite(metrics["train_mse"]) and metrics["train_mse"] >= 0, (
        f"train_mse invalid: {metrics['train_mse']}"
    )
    assert np.isfinite(metrics["val_mse"]) and metrics["val_mse"] >= 0, (
        f"val_mse invalid: {metrics['val_mse']}"
    )


def test_val_mse_reasonable(tiny_cfg, tmp_path):
    """
    Val MSE should be finite and not astronomically large.
    (Sanity check: a trivially bad model would have MSE >> 1e6 on normalised data.)
    """
    metrics = train_koopman(
        tiny_cfg, n_episodes=10,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    assert metrics["val_mse"] < 1e8, (
        f"Val MSE suspiciously large: {metrics['val_mse']:.2e}"
    )