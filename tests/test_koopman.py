"""
tests/test_koopman.py
---------------------
Unit tests for Phase 4: models/koopman.py
"""
import pathlib
import numpy as np
import yaml
import pytest
from sim.env import SwarmEnv
from sim.dataset import rollout
from models.koopman import KoopmanPredictor


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    path = pathlib.Path("configs") / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tiny_cfg(cfg):
    import copy
    c = copy.deepcopy(cfg)
    c["swarm"]["N"] = 8
    c["swarm"]["rollout_horizon"] = 20
    c["koopman"]["obs_dim"] = 12
    c["koopman"]["forecast_steps"] = 5
    return c


@pytest.fixture
def fitted_model(tiny_cfg):
    env  = SwarmEnv(tiny_cfg, seed=0)
    traj = rollout(env, tiny_cfg["swarm"]["rollout_horizon"])
    model = KoopmanPredictor(tiny_cfg, seed=0)
    model.fit(traj)
    return model, traj, tiny_cfg


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_lift_shape(tiny_cfg):
    """lift() must return (N, obs_dim) float64."""
    N = tiny_cfg["swarm"]["N"]
    d = tiny_cfg["koopman"]["obs_dim"]
    model = KoopmanPredictor(tiny_cfg, seed=0)
    state = np.random.default_rng(0).uniform(0, 100, size=(N, 4)).astype(np.float32)
    Z = model.lift(state)
    assert Z.shape == (N, d), f"lift shape {Z.shape} != ({N}, {d})"
    assert Z.dtype == np.float64


def test_fit_runs(tiny_cfg):
    """fit() must complete and set K and W matrices."""
    env  = SwarmEnv(tiny_cfg, seed=1)
    traj = rollout(env, tiny_cfg["swarm"]["rollout_horizon"])
    model = KoopmanPredictor(tiny_cfg, seed=0)
    model.fit(traj)
    d = tiny_cfg["koopman"]["obs_dim"]
    assert model.K is not None, "K not set after fit()"
    assert model.W is not None, "W not set after fit()"
    assert model.K.shape == (d, d), f"K shape {model.K.shape}"
    assert model.W.shape == (4, d), f"W shape {model.W.shape}"


def test_predict_shape(fitted_model):
    """predict() must return (h, N, 4) float32."""
    model, traj, cfg = fitted_model
    N = cfg["swarm"]["N"]
    h = cfg["koopman"]["forecast_steps"]
    state = traj[0]   # (N, 4)
    forecast = model.predict(state, h=h)
    assert forecast.shape == (h, N, 4), f"forecast shape {forecast.shape}"
    assert forecast.dtype == np.float32


def test_spectral_radius_bounded(fitted_model):
    """Spectral radius of K must be <= 1 + small epsilon."""
    model, _, _ = fitted_model
    rho = model.spectral_radius
    assert rho <= 1.0 + 1e-6, f"Spectral radius {rho:.6f} > 1"


def test_predict_before_fit_raises(tiny_cfg):
    """predict() before fit() must raise RuntimeError."""
    model = KoopmanPredictor(tiny_cfg, seed=0)
    state = np.zeros((tiny_cfg["swarm"]["N"], 4), dtype=np.float32)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(state)


def test_rls_update_changes_K(fitted_model, tiny_cfg):
    """After an RLS update, K must change."""
    model, traj, cfg = fitted_model
    K_before = model.K.copy()
    # Use one pair of consecutive lifted states
    z0 = model.lift(traj[0]).mean(axis=0)   # (d,) mean across agents
    z1 = model.lift(traj[1]).mean(axis=0)
    model.update(z0, z1)
    assert not np.allclose(model.K, K_before), "K unchanged after RLS update"


def test_multi_step_predict_consistent(fitted_model):
    """h=1 and h=5 predictions must agree on the first step."""
    model, traj, cfg = fitted_model
    state = traj[5]
    f1 = model.predict(state, h=1)   # (1, N, 4)
    f5 = model.predict(state, h=5)   # (5, N, 4)
    # First step of both must be identical (same K application)
    np.testing.assert_allclose(
        f1[0], f5[0], rtol=1e-5,
        err_msg="h=1 and h=5 first step disagree"
    )