"""
tests/test_baselines.py
-----------------------
Unit tests for Phase 6: baselines/constant_velocity.py + baselines/mean_field.py
"""
import pathlib
import numpy as np
import yaml
import pytest
from baselines.constant_velocity import CVPredictor
from baselines.mean_field import MeanFieldPredictor


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
    c["swarm"]["N"] = 10
    return c


def _state(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(20, 80, size=(N, 2))
    vel = rng.uniform(-2, 2,  size=(N, 2))
    return np.hstack([pos, vel]).astype(np.float32)


# ── CV tests ──────────────────────────────────────────────────────────────────

def test_cv_output_shape(tiny_cfg):
    """CVPredictor.predict() must return (h, N, 4) float32."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    model = CVPredictor(tiny_cfg)
    forecast = model.predict(state, h=10)
    assert forecast.shape == (10, N, 4)
    assert forecast.dtype == np.float32


def test_cv_exact_math(tiny_cfg):
    """CV prediction must match hand-computed pos + h*dt*vel exactly."""
    N = 4
    dt = tiny_cfg["swarm"]["dt"]
    h  = 3
    world = tiny_cfg["swarm"]["world_size"]

    rng = np.random.default_rng(99)
    pos = rng.uniform(20, 60, size=(N, 2))
    vel = rng.uniform(0.1, 0.5, size=(N, 2))   # small positive: no wrap needed
    state = np.hstack([pos, vel]).astype(np.float32)

    model = CVPredictor(tiny_cfg)
    forecast = model.predict(state, h=h)

    # Hand compute step-by-step
    expected_pos = pos.copy()
    for step in range(h):
        expected_pos = (expected_pos + dt * vel) % world

    np.testing.assert_allclose(
        forecast[h - 1, :, :2], expected_pos, rtol=1e-5, atol=1e-5,
        err_msg="CV positions don't match expected hand-computed values"
    )


def test_cv_velocities_constant(tiny_cfg):
    """CV baseline must keep velocities unchanged across all steps."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    model = CVPredictor(tiny_cfg)
    forecast = model.predict(state, h=8)
    vel0 = state[:, 2:]
    for step in range(8):
        np.testing.assert_allclose(
            forecast[step, :, 2:], vel0, rtol=1e-5,
            err_msg=f"Velocity changed at step {step}"
        )


# ── Mean-field tests ──────────────────────────────────────────────────────────

def test_mf_output_shape(tiny_cfg):
    """MeanFieldPredictor.predict() must return (h, N, 4) float32."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    model = MeanFieldPredictor(tiny_cfg)
    forecast = model.predict(state, h=10)
    assert forecast.shape == (10, N, 4)
    assert forecast.dtype == np.float32


def test_mf_centroid_pull(tiny_cfg):
    """
    With alpha=1 (pure centroid pull) and a single step, all agents
    should move closer to the centroid than they started.
    """
    N = tiny_cfg["swarm"]["N"]
    state = _state(N, seed=5)
    model = MeanFieldPredictor(tiny_cfg, alpha=1.0)
    forecast = model.predict(state, h=1)

    orig_pos      = state[:, :2].astype(np.float64)
    centroid      = orig_pos.mean(axis=0)
    pred_pos      = forecast[0, :, :2].astype(np.float64)

    orig_dist = np.linalg.norm(orig_pos - centroid, axis=1)
    pred_dist = np.linalg.norm(pred_pos - centroid, axis=1)

    # After pure centroid pull, agents should be no further away
    # (some may be exactly at centroid if they were already there)
    assert np.all(pred_dist <= orig_dist + 1e-3), (
        "Mean-field with alpha=1 did not reduce distance to centroid"
    )


def test_mf_speed_bounded(tiny_cfg):
    """MeanField predictions must respect max_speed at every step."""
    N = tiny_cfg["swarm"]["N"]
    max_sp = tiny_cfg["physics"]["max_speed"]
    state = _state(N)
    model = MeanFieldPredictor(tiny_cfg)
    forecast = model.predict(state, h=15)
    speeds = np.linalg.norm(forecast[:, :, 2:], axis=2)
    assert np.all(speeds <= max_sp + 1e-5), (
        f"Speed exceeded max: {speeds.max():.4f} > {max_sp}"
    )


def test_mf_positions_in_bounds(tiny_cfg):
    """MeanField positions must stay within [0, world_size]."""
    N = tiny_cfg["swarm"]["N"]
    world = tiny_cfg["swarm"]["world_size"]
    state = _state(N)
    model = MeanFieldPredictor(tiny_cfg)
    forecast = model.predict(state, h=20)
    pos = forecast[:, :, :2]
    assert np.all(pos >= 0.0) and np.all(pos <= world + 1e-5), (
        f"Position out of bounds: min={pos.min():.2f}, max={pos.max():.2f}"
    )


def test_baselines_no_nan(tiny_cfg):
    """Neither baseline should produce NaN or Inf."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    cv = CVPredictor(tiny_cfg)
    mf = MeanFieldPredictor(tiny_cfg)
    for forecast in [cv.predict(state, 20), mf.predict(state, 20)]:
        assert np.all(np.isfinite(forecast)), "NaN or Inf in baseline forecast"