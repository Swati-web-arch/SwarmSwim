"""
tests/test_sim.py
-----------------
Unit tests for Phase 1: sim/env.py and sim/formations.py
"""
import pathlib
import numpy as np
import yaml
import pytest
from sim.env import SwarmEnv
from sim.formations import make_formation


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    path = pathlib.Path("configs") / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def env(cfg):
    return SwarmEnv(cfg, seed=42)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_reset_shape(env, cfg):
    """reset() must return (N,2) arrays."""
    N = cfg["swarm"]["N"]
    pos, vel = env.reset()
    assert pos.shape == (N, 2), f"pos shape {pos.shape} != ({N}, 2)"
    assert vel.shape == (N, 2), f"vel shape {vel.shape} != ({N}, 2)"


def test_step_shape(env, cfg):
    """step() must return (N,2) arrays."""
    N = cfg["swarm"]["N"]
    env.reset()
    pos, vel = env.step()
    assert pos.shape == (N, 2)
    assert vel.shape == (N, 2)


def test_speed_bounded(env, cfg):
    """After many steps, no agent should exceed max_speed."""
    max_speed = cfg["physics"]["max_speed"]
    env.reset()
    for _ in range(100):
        _, vel = env.step()
    speeds = np.linalg.norm(vel, axis=1)
    assert np.all(speeds <= max_speed + 1e-6), (
        f"Max speed exceeded: {speeds.max():.4f} > {max_speed}"
    )


def test_boundary_wrap(env, cfg):
    """Positions must always stay within [0, world_size)."""
    world_size = cfg["swarm"]["world_size"]
    env.reset()
    for _ in range(200):
        pos, _ = env.step()
    assert np.all(pos >= 0.0), f"Negative position found: {pos.min()}"
    assert np.all(pos < world_size + 1e-9), f"Position exceeds world: {pos.max()}"


def test_all_formations(cfg):
    """All formation types must produce correctly shaped arrays."""
    N = cfg["swarm"]["N"]
    world = cfg["swarm"]["world_size"]
    for kind in ("random", "circle", "grid", "two_clusters"):
        rng = np.random.default_rng(0)
        pos, vel = make_formation(kind, N, world, rng)
        assert pos.shape == (N, 2), f"{kind}: pos shape {pos.shape}"
        assert vel.shape == (N, 2), f"{kind}: vel shape {vel.shape}"
        assert np.all(pos >= 0.0) and np.all(pos <= world), (
            f"{kind}: positions out of bounds"
        )


def test_deterministic_seed(cfg):
    """Two envs with same seed must produce identical trajectories."""
    env1 = SwarmEnv(cfg, seed=7)
    env2 = SwarmEnv(cfg, seed=7)
    env1.reset(); env2.reset()
    for _ in range(10):
        p1, v1 = env1.step()
        p2, v2 = env2.step()
    np.testing.assert_array_equal(p1, p2, err_msg="Positions differ with same seed")
    np.testing.assert_array_equal(v1, v2, err_msg="Velocities differ with same seed")


def test_different_seeds_differ(cfg):
    """Two envs with different seeds must produce different initial positions."""
    env1 = SwarmEnv(cfg, seed=1)
    env2 = SwarmEnv(cfg, seed=2)
    p1, _ = env1.reset()
    p2, _ = env2.reset()
    assert not np.allclose(p1, p2), "Different seeds produced identical positions"