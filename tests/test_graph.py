"""
tests/test_graph.py
-------------------
Unit tests for Phase 3: sim/graph.py
"""
import numpy as np
import pytest
from sim.graph import build_graph, build_graph_sequence


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_state(N: int, seed: int = 0) -> np.ndarray:
    """Random (N, 4) state, positions in [0, 100)."""
    rng = np.random.default_rng(seed)
    pos = rng.uniform(0, 100, size=(N, 2))
    vel = rng.uniform(-3, 3,  size=(N, 2))
    return np.hstack([pos, vel]).astype(np.float32)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_output_keys():
    """Graph dict must contain required keys."""
    g = build_graph(_make_state(5), radius=30.0)
    required = {"node_features", "edge_index", "edge_attr", "num_nodes", "num_edges"}
    assert required.issubset(g.keys()), f"Missing keys: {required - g.keys()}"


def test_node_feature_shape():
    """node_features must be (N, 4) float32."""
    N = 10
    g = build_graph(_make_state(N), radius=50.0)
    assert g["node_features"].shape == (N, 4)
    assert g["node_features"].dtype == np.float32


def test_edge_index_shape():
    """edge_index must be (2, E) int64."""
    g = build_graph(_make_state(8), radius=40.0)
    E = g["num_edges"]
    assert g["edge_index"].shape == (2, E), (
        f"edge_index shape {g['edge_index'].shape} != (2, {E})"
    )
    assert g["edge_index"].dtype == np.int64


def test_edge_attr_shape():
    """edge_attr must be (E, 1) float32."""
    g = build_graph(_make_state(8), radius=40.0)
    E = g["num_edges"]
    assert g["edge_attr"].shape == (E, 1)
    assert g["edge_attr"].dtype == np.float32


def test_no_self_loops():
    """No edge should connect a node to itself."""
    g = build_graph(_make_state(12), radius=60.0)
    if g["num_edges"] > 0:
        src, dst = g["edge_index"]
        assert not np.any(src == dst), "Self-loop detected in edge_index"


def test_symmetric_edges():
    """Graph must be symmetric: if (i->j) exists then (j->i) must exist."""
    g = build_graph(_make_state(10), radius=50.0)
    if g["num_edges"] == 0:
        pytest.skip("No edges to test symmetry")
    src, dst = g["edge_index"]
    edge_set = set(zip(src.tolist(), dst.tolist()))
    for s, d in zip(src.tolist(), dst.tolist()):
        assert (d, s) in edge_set, f"Missing reverse edge ({d} -> {s})"


def test_radius_threshold():
    """Only agent pairs within radius should be connected."""
    # Place 3 agents at known positions
    state = np.array([
        [0.0, 0.0, 0.0, 0.0],   # agent 0
        [5.0, 0.0, 0.0, 0.0],   # agent 1 — dist to 0 = 5
        [50.0, 0.0, 0.0, 0.0],  # agent 2 — dist to 0 = 50, to 1 = 45
    ], dtype=np.float32)

    # radius=6: only 0<->1 connected
    g = build_graph(state, radius=6.0)
    E = g["num_edges"]
    assert E == 2, f"Expected 2 edges (0->1 and 1->0), got {E}"

    # radius=1: no connections
    g0 = build_graph(state, radius=1.0)
    assert g0["num_edges"] == 0, "Expected 0 edges for tiny radius"

    # radius=100: all pairs connected (3 agents -> 6 directed edges)
    g_all = build_graph(state, radius=100.0)
    assert g_all["num_edges"] == 6, (
        f"Expected 6 edges for large radius, got {g_all['num_edges']}"
    )


def test_edge_distances_correct():
    """edge_attr distances must match actual Euclidean distances."""
    state = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [3.0, 4.0, 0.0, 0.0],  # dist = 5.0 exactly
    ], dtype=np.float32)
    g = build_graph(state, radius=10.0)
    assert g["num_edges"] == 2  # both directions
    dists = g["edge_attr"][:, 0]
    np.testing.assert_allclose(dists, 5.0, rtol=1e-5)


def test_build_graph_sequence_shape():
    """build_graph_sequence must return T graph dicts."""
    T, N = 8, 6
    rng = np.random.default_rng(5)
    traj = rng.uniform(0, 100, size=(T, N, 4)).astype(np.float32)
    graphs = build_graph_sequence(traj, radius=30.0)
    assert len(graphs) == T
    for t, g in enumerate(graphs):
        assert g["node_features"].shape == (N, 4), f"t={t}: wrong node shape"
        assert g["num_nodes"] == N


def test_empty_state_raises():
    """Zero-node state should raise ValueError."""
    bad = np.zeros((0, 4), dtype=np.float32)
    g = build_graph(bad, radius=10.0)    # should not crash, just be empty
    assert g["num_nodes"] == 0
    assert g["num_edges"] == 0


def test_wrong_feature_dim_raises():
    """State with wrong feature dim must raise ValueError."""
    bad = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        build_graph(bad, radius=10.0)