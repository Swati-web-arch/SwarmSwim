"""
tests/test_gnn.py
-----------------
Unit tests for Phase 5: models/gnn.py
"""
import pathlib
import numpy as np
import yaml
import pytest
from sim.graph import build_graph
from models.gnn import GNNPredictor, MPLayer


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
    c["swarm"]["world_size"] = 100.0
    c["koopman"]["obs_dim"] = 6   # hidden = 12
    return c


@pytest.fixture
def gnn(tiny_cfg):
    return GNNPredictor(tiny_cfg, seed=0)


def _state(N: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pos = rng.uniform(10, 90, size=(N, 2))
    vel = rng.uniform(-2, 2,  size=(N, 2))
    return np.hstack([pos, vel]).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_forward_shape(gnn, tiny_cfg):
    """forward() must return (N, 4) float32."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    graph = build_graph(state, radius=30.0)
    delta = gnn.forward(graph)
    assert delta.shape == (N, 4), f"Expected ({N}, 4), got {delta.shape}"
    assert delta.dtype == np.float32


def test_predict_shape(gnn, tiny_cfg):
    """predict() must return (h, N, 4) float32."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    h = 5
    forecast = gnn.predict(state, h=h, radius=30.0)
    assert forecast.shape == (h, N, 4), f"Expected ({h},{N},4), got {forecast.shape}"
    assert forecast.dtype == np.float32


def test_permutation_equivariance(gnn, tiny_cfg):
    """
    GNN must be permutation-equivariant:
    shuffling input agents should shuffle output in the same order.
    """
    N = tiny_cfg["swarm"]["N"]
    state = _state(N, seed=7)
    radius = 40.0

    # Original
    graph_orig = build_graph(state, radius)
    delta_orig = gnn.forward(graph_orig)

    # Permute agents
    perm = np.random.default_rng(3).permutation(N)
    state_perm = state[perm]
    graph_perm = build_graph(state_perm, radius)
    delta_perm = gnn.forward(graph_perm)

    # delta_perm should equal delta_orig[perm]
    np.testing.assert_allclose(
        delta_perm, delta_orig[perm], rtol=1e-4, atol=1e-5,
        err_msg="GNN is not permutation-equivariant"
    )


def test_zero_edge_graph(gnn, tiny_cfg):
    """forward() must not crash when there are no edges."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N)
    graph = build_graph(state, radius=0.0)   # no connections
    assert graph["num_edges"] == 0
    delta = gnn.forward(graph)
    assert delta.shape == (N, 4)
    assert np.all(np.isfinite(delta)), "NaN/Inf in zero-edge forward pass"


def test_predict_positions_in_bounds(gnn, tiny_cfg):
    """predict() must keep positions within [0, world_size]."""
    N   = tiny_cfg["swarm"]["N"]
    world = tiny_cfg["swarm"]["world_size"]
    state = _state(N)
    forecast = gnn.predict(state, h=10, radius=30.0)
    pos = forecast[:, :, :2]
    assert np.all(pos >= 0.0) and np.all(pos <= world + 1e-5), (
        f"Position out of bounds: min={pos.min():.2f} max={pos.max():.2f}"
    )


def test_mplayer_output_shape():
    """MPLayer.forward() must return (N, out_dim) array."""
    layer = MPLayer(in_dim=4, edge_dim=1, out_dim=8, seed=0)
    N = 6
    node_feat  = np.random.default_rng(0).uniform(size=(N, 4))
    edge_index = np.array([[0, 1, 2], [1, 2, 0]], dtype=np.int64)
    edge_attr  = np.ones((3, 1))
    out = layer.forward(node_feat, edge_index, edge_attr)
    assert out.shape == (N, 8), f"Expected ({N}, 8), got {out.shape}"


def test_no_nan_after_many_steps(gnn, tiny_cfg):
    """No NaN or Inf should appear after 20 autoregressive steps."""
    N = tiny_cfg["swarm"]["N"]
    state = _state(N, seed=42)
    forecast = gnn.predict(state, h=20, radius=30.0)
    assert np.all(np.isfinite(forecast)), "NaN or Inf detected in long forecast"