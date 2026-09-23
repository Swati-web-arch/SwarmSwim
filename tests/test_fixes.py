"""
tests/test_fixes.py
-------------------
Regression tests for the Koopman/GNN RMSE fixes:
  1. toroidal proximity graph (wrap-around distances)
  2. Koopman toroidal Fourier basis + decode round-trip + bounded forecasts
  3. GNN zero-init head (zero-delta baseline at init)
  4. GNN checkpoint save/load round-trip
  5. GNN gradient-based training reduces the loss
"""
import numpy as np
import pytest
import yaml

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from sim.graph import build_graph
from sim.env import SwarmEnv
from models.koopman import KoopmanPredictor
from models.gnn import GNNPredictor
from models.train_gnn import _param_list, _batch_grads, _build_pairs
from models.train_gnn import train_gnn, load_gnn


@pytest.fixture(scope="module")
def cfg():
    return yaml.safe_load(open("configs/default.yaml"))


import copy


def _small_cfg(cfg, N: int, T: int) -> dict:
    """Deep-copied config with small swarm size (avoids cross-test pollution
    that shallow dict(cfg) copies cause via shared nested dicts)."""
    c = copy.deepcopy(cfg)
    c["swarm"]["N"] = N
    c["swarm"]["rollout_horizon"] = T
    return c


def _clustered_state(N=12, seed=0):
    """Small swarm state from the real simulator (2 steps), shape (N, 4)."""
    c = yaml.safe_load(open("configs/default.yaml"))
    c["swarm"]["N"] = N
    env = SwarmEnv(c, seed=seed)
    pos, vel = env.reset()
    for _ in range(2):
        pos, vel = env.step()
    return np.hstack([pos, vel]).astype(np.float32), c


# ── 1. Toroidal graph ─────────────────────────────────────────────────────────

def test_build_graph_toroidal_distances():
    """Agents across the periodic boundary must be 'close' on the torus."""
    state = np.array([[1.0, 50.0, 0.5, 0.0],
                      [99.0, 50.0, -0.5, 0.0]], dtype=np.float32)
    eu = build_graph(state, radius=5.0)                     # plain Euclidean
    tor = build_graph(state, radius=5.0, world_size=100.0)  # toroidal
    assert eu["num_edges"] == 0        # 98 m apart in raw coordinates
    assert tor["num_edges"] == 2       # 2 m apart across the wrap
    assert np.isclose(tor["edge_attr"][0, 0], 2.0, atol=1e-5)


def test_build_graph_backward_compatible():
    """world_size=None (default) keeps plain Euclidean behaviour."""
    rng = np.random.default_rng(0)
    state = np.hstack([rng.uniform(0, 100, (10, 2)),
                       rng.uniform(-3, 3, (10, 2))]).astype(np.float32)
    g1 = build_graph(state, radius=20.0)
    g2 = build_graph(state, radius=20.0, world_size=None)
    assert g1["num_edges"] == g2["num_edges"]
    assert np.allclose(g1["edge_attr"], g2["edge_attr"])


# ── 2. Koopman toroidal basis ─────────────────────────────────────────────────

def test_koopman_decode_roundtrip(cfg):
    """lift -> _decode must recover the original state (torus-aware)."""
    model = KoopmanPredictor(cfg, seed=1)
    rng = np.random.default_rng(3)
    state = np.hstack([rng.uniform(0, 100, (20, 2)),
                       rng.uniform(-3, 3, (20, 2))])
    Z = model.lift(state)
    rec = model._decode(Z)
    assert np.allclose(rec[:, :2], state[:, :2], atol=1e-6)
    assert np.allclose(rec[:, 2:], state[:, 2:], atol=1e-6)


def test_koopman_lift_continuous_across_wrap(cfg):
    """
    Lifted features must be CONTINUOUS across the boundary: two points 0.1 m
    apart on opposite sides of the wrap (x=0.05 vs x=99.95) must have nearly
    identical lifts. With the old raw-coordinate basis the difference was O(100).
    """
    model = KoopmanPredictor(cfg, seed=1)
    s1 = np.array([[0.05, 50.0, 1.0, 0.0]])
    s2 = np.array([[99.95, 50.0, 1.0, 0.0]])   # 0.1 m away through the wrap
    diff = np.max(np.abs(model.lift(s1) - model.lift(s2)))
    assert diff < 0.02   # continuous: diff ~ (2*pi/L)*0.1 ~ 0.0063


def test_koopman_forecast_bounded_and_accurate(cfg):
    """After fitting, 1-step forecasts stay in the arena and are accurate."""
    c = _small_cfg(cfg, N=12, T=12)
    episodes = []
    for s in range(4):
        env = SwarmEnv(c, seed=s)
        pos, vel = env.reset()
        frames = [np.hstack([pos, vel])]
        for _ in range(c["swarm"]["rollout_horizon"] - 1):
            pos, vel = env.step()
            frames.append(np.hstack([pos, vel]))
        episodes.append(np.stack(frames))
    data = np.stack(episodes)

    model = KoopmanPredictor(cfg, seed=42)
    model.fit(data.astype(np.float64))

    L = float(cfg["swarm"]["world_size"])
    fc = model.predict(data[0][0], h=5)
    assert fc.shape == (5, 12, 4)
    assert fc[:, :, :2].min() >= 0.0 and fc[:, :, :2].max() <= L

    errs = [np.sqrt(np.mean((model.predict(ep[t], h=1)[0][:, :2]
                             - ep[t + 1][:, :2]) ** 2))
            for ep in data for t in range(6)]
    assert np.mean(errs) < 5.0   # raw coordinate one-step RMSE (metres)


# ── 3. GNN zero-init head ─────────────────────────────────────────────────────

def test_gnn_zero_init_outputs_zero_delta(cfg):
    """Untrained GNN must output the zero-delta predictor (stable baseline)."""
    state, _ = _clustered_state()
    model = GNNPredictor(cfg, seed=0)
    graph = build_graph(state, radius=float(cfg["physics"]["alignment_radius"]))
    delta = model.forward(graph)
    assert np.abs(delta).max() < 1e-8


# ── 4. GNN checkpoint round-trip ──────────────────────────────────────────────

def test_gnn_checkpoint_roundtrip(cfg, tmp_path):
    """train_gnn -> load_gnn must reproduce identical forward outputs."""
    c = _small_cfg(cfg, N=8, T=6)
    m = train_gnn(c, n_episodes=2, steps=5, batch_size=4,
                  out_dir=str(tmp_path), verbose=False)
    model = load_gnn(c, ckpt_path=m["ckpt_path"])

    state, _ = _clustered_state(N=8)
    graph = build_graph(state, radius=float(c["physics"]["alignment_radius"]))
    out1 = model.forward(graph)

    model2 = load_gnn(c, ckpt_path=m["ckpt_path"])
    out2 = model2.forward(graph)
    assert np.allclose(out1, out2)


# ── 5. GNN training reduces loss ──────────────────────────────────────────────

def test_gnn_training_reduces_loss(cfg, tmp_path):
    """A short training run must reduce the one-step loss below its init value."""
    c = _small_cfg(cfg, N=8, T=8)
    m = train_gnn(c, n_episodes=4, steps=60, batch_size=8, lr=3e-3,
                  out_dir=str(tmp_path), verbose=False)
    h = m["history"]
    assert len(h) == 60
    assert np.mean(h[:5]) > np.mean(h[-5:])


def test_gnn_gradients_match_numerical(cfg):
    """Analytic gradients must match finite differences (sampled entries)."""
    rng = np.random.default_rng(7)
    c = _small_cfg(cfg, N=6, T=8)
    eps = rng.uniform(0, 100, size=(1, 4, 6, 4)).astype(np.float32)
    pairs = _build_pairs(eps, c)
    idx = list(range(len(pairs)))

    model = GNNPredictor(c, seed=0)
    # Randomise the output head: with the zero-init head the gradient signal
    # to lower layers is exactly zero, leaving nothing to check.
    rng_p = np.random.default_rng(11)
    model.W_out = 0.05 * rng_p.standard_normal(model.W_out.shape)
    grads, loss = _batch_grads(model, pairs, idx)
    params = _param_list(model)

    checked = 0
    for pi in range(10):
        p = params[pi]
        flat = p.reshape(-1)
        gflat = grads[pi].reshape(-1)
        for k in rng.choice(flat.size, size=4, replace=False):
            k = int(k)
            if abs(gflat[k]) < 1e-4:      # skip near-zero grads (FD noise)
                continue
            old = flat[k]; e = 1e-6
            flat[k] = old + e; _, lp = _batch_grads(model, pairs, idx)
            flat[k] = old - e; _, lm = _batch_grads(model, pairs, idx)
            flat[k] = old
            num = (lp - lm) / (2 * e)
            rel = abs(num - gflat[k]) / max(abs(num), abs(gflat[k]), 1e-8)
            assert rel < 1e-4
            checked += 1
    assert checked >= 10