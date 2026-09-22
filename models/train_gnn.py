"""
models/train_gnn.py
-------------------
Gradient-based training for the NumPy GNN predictor.

Motivation
----------
GNNPredictor weights were previously random-initialised only, so its
predictions were meaningless (RMSE ~10-30 m, worse than the CV baseline).
This module trains them with supervised one-step learning:

    input   : state X_t  (proximity graph built from it)
    target  : delta = X_{t+1} - X_t   (position deltas wrapped on the torus)

trained with MSE on normalised deltas via manual backpropagation through the
two message-passing layers + linear head, optimised with Adam.

Checkpoint format (checkpoints/gnn.npz)
---------------------------------------
    layer1_W_msg, layer1_b_msg, layer1_W_upd, layer1_b_upd,
    layer2_W_msg, layer2_b_msg, layer2_W_upd, layer2_b_upd,
    W_out, b_out

Public API
----------
train_gnn(cfg, ...) -> metrics dict   (also saves checkpoint)
load_gnn(cfg, ckpt_path) -> GNNPredictor
"""
import pathlib
import numpy as np

from sim.graph import build_graph
from models.gnn import GNNPredictor, MPLayer, _tanh


# ── Parameter access helpers ──────────────────────────────────────────────────

def _param_list(model: GNNPredictor) -> list:
    """References to all trainable arrays, in a fixed order."""
    return [
        model.layer1.W_msg, model.layer1.b_msg,
        model.layer1.W_upd, model.layer1.b_upd,
        model.layer2.W_msg, model.layer2.b_msg,
        model.layer2.W_upd, model.layer2.b_upd,
        model.W_out, model.b_out,
    ]


def _grad_template(model: GNNPredictor) -> list:
    """Zero-initialised gradient buffers matching _param_list order."""
    return [np.zeros_like(p) for p in _param_list(model)]


# ── Forward with cache (single graph) ─────────────────────────────────────────

def _forward_cached(model: GNNPredictor, graph: dict) -> dict:
    """
    Forward pass storing intermediates for backprop.

    Returns cache dict with the layer inputs/activations and the head output
    `delta_norm` (normalised units: positions/world, velocities/max_speed).
    """
    node_feat  = graph["node_features"].astype(np.float64)
    edge_index = graph["edge_index"]
    edge_attr  = graph["edge_attr"].astype(np.float64)

    world  = float(model.cfg["swarm"]["world_size"])
    max_sp = float(model.cfg["physics"]["max_speed"])
    norm   = np.array([world, world, max_sp, max_sp])
    h0     = node_feat / norm                      # (N, 4)  normalised input

    cache = {"edge_index": edge_index, "h0": h0, "norm": norm}

    for li, layer in enumerate((model.layer1, model.layer2), start=1):
        N = h0.shape[0]
        E = edge_index.shape[1]
        if E == 0:
            agg = np.zeros((N, layer.out_dim))
            M, A = None, None
        else:
            src, dst = edge_index[0], edge_index[1]
            M = np.concatenate(
                [h0[src], h0[dst], edge_attr], axis=1)      # (E, 2d+edge)
            S = M @ layer.W_msg.T + layer.b_msg             # (E, out)
            A = _tanh(S)
            agg = np.zeros((N, layer.out_dim))
            np.add.at(agg, dst, A)
        U = np.concatenate([h0, agg], axis=1)               # (N, in+out)
        V = U @ layer.W_upd.T + layer.b_upd
        H = _tanh(V)                                        # (N, out)

        cache[f"l{li}"] = {"M": M, "A": A, "U": U, "H": H,
                           "edge_index": edge_index}
        h0 = H

    delta_norm = H @ model.W_out.T + model.b_out            # (N, 4)
    cache["delta_norm"] = delta_norm
    return cache


# ── Backward for one MP layer ─────────────────────────────────────────────────

def _layer_backward(layer: MPLayer, cache_l: dict, dH_out: np.ndarray) -> tuple:
    """
    Backprop through one message-passing layer.

    Returns
    -------
    dH_in : (N, in_dim)  gradient wrt layer input
    grads : dict of gradient arrays for W_msg, b_msg, W_upd, b_upd
    """
    H, U = cache_l["H"], cache_l["U"]

    # update branch: V = U @ W_upd.T + b_upd, H = tanh(V)
    dV     = dH_out * (1.0 - H ** 2)
    dW_upd = dV.T @ U
    db_upd = dV.sum(axis=0)
    dU     = dV @ layer.W_upd

    d_in = layer.in_dim
    dH   = dU[:, :d_in].copy()
    dG   = dU[:, d_in:]

    # message branch
    if cache_l["M"] is not None:
        src, dst = cache_l["edge_index"]
        M, A     = cache_l["M"], cache_l["A"]

        dA = dG[dst]                                   # (E, out)
        dS = dA * (1.0 - A ** 2)
        dW_msg = dS.T @ M
        db_msg = dS.sum(axis=0)
        dM     = dS @ layer.W_msg                      # (E, 2d+edge)

        np.add.at(dH, src, dM[:, :d_in])
        np.add.at(dH, dst, dM[:, d_in:2 * d_in])
    else:
        dW_msg = np.zeros_like(layer.W_msg)
        db_msg = np.zeros_like(layer.b_msg)

    return dH, {"W_msg": dW_msg, "b_msg": db_msg,
                "W_upd": dW_upd, "b_upd": db_upd}


# ── Training pair preparation ─────────────────────────────────────────────────

def _build_pairs(episodes: np.ndarray, cfg: dict) -> list:
    """
    Convert (E, T, N, 4) episodes into one-step (state, target_delta_norm) pairs.

    Position deltas are wrapped to the torus (shortest signed displacement),
    matching how SwarmEnv moves agents across the periodic boundary.
    """
    L      = float(cfg["swarm"]["world_size"])
    max_sp = float(cfg["physics"]["max_speed"])
    norm   = np.array([L, L, max_sp, max_sp])

    pairs = []
    for ep in episodes:
        T = ep.shape[0]
        for t in range(T - 1):
            x0, x1 = ep[t].astype(np.float64), ep[t + 1].astype(np.float64)
            d = x1 - x0
            d[:, :2] = (d[:, :2] + L / 2.0) % L - L / 2.0   # torus wrap
            pairs.append((x0.astype(np.float32), (d / norm).astype(np.float64)))
    return pairs


def _batch_grads(model: GNNPredictor, pairs: list, idx: list) -> tuple:
    """Accumulate gradients and mean loss over a mini-batch of pairs."""
    grads  = _grad_template(model)
    radius = float(model.cfg["physics"]["alignment_radius"])
    world  = float(model.cfg["swarm"]["world_size"])
    loss   = 0.0

    for i in idx:
        state, target = pairs[i]
        graph = build_graph(state, radius, world_size=world)
        cache = _forward_cached(model, graph)
        D     = cache["delta_norm"]                     # (N, 4)
        resid = D - target

        # MSE averaged over elements
        loss += float(np.mean(resid ** 2))

        dD = 2.0 * resid / resid.size                  # (N, 4)

        # head
        H2 = cache["l2"]["H"]
        grads[-2] += dD.T @ H2                          # W_out
        grads[-1] += dD.sum(axis=0)                     # b_out
        dH2 = dD @ model.W_out

        # layer2 then layer1
        dH1, g2  = _layer_backward(model.layer2, cache["l2"], dH2)
        _,   g1  = _layer_backward(model.layer1, cache["l1"], dH1)

        grads[4] += g2["W_msg"]; grads[5] += g2["b_msg"]
        grads[6] += g2["W_upd"]; grads[7] += g2["b_upd"]
        grads[0] += g1["W_msg"]; grads[1] += g1["b_msg"]
        grads[2] += g1["W_upd"]; grads[3] += g1["b_upd"]

    # average over the batch (loss is averaged too)
    n = max(len(idx), 1)
    for g in grads:
        g /= n
    return grads, loss / n


# ── Adam optimizer ────────────────────────────────────────────────────────────

class _Adam:
    def __init__(self, params: list, lr: float = 3e-3,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params: list, grads: list) -> None:
        self.t += 1
        for p, g, m, v in zip(params, grads, self.m, self.v):
            m *= self.b1; m += (1 - self.b1) * g
            v *= self.b2; v += (1 - self.b2) * (g * g)
            m_hat = m / (1 - self.b1 ** self.t)
            v_hat = v / (1 - self.b2 ** self.t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ── Main training entry-point ─────────────────────────────────────────────────

def train_gnn(
    cfg: dict,
    n_episodes: int = 80,
    steps: int = 500,
    batch_size: int = 16,
    lr: float = 3e-3,
    out_dir: str = "checkpoints",
    data_dir: str = "data",
    seed_offset: int = 0,
    verbose: bool = True,
    warm_start: bool = False,
    save_every: int = 0,
) -> dict:
    """
    Train GNNPredictor on one-step delta prediction and save checkpoint.

    Parameters
    ----------
    warm_start : if True and a checkpoint already exists in out_dir, resume
                 training from it instead of a fresh random init (allows
                 running training in several chunks).
    save_every : if > 0, checkpoint intermediate weights every N steps.

    Returns
    -------
    metrics : dict with "train_loss", "val_loss", "history", "ckpt_path"
    """
    from sim.dataset import generate_dataset, load_dataset

    generate_dataset(cfg, n_episodes=n_episodes, val_fraction=0.2,
                     out_dir=data_dir, seed_offset=seed_offset)
    train_eps = load_dataset("train", out_dir=data_dir)["episodes"]
    val_eps   = load_dataset("val",   out_dir=data_dir)["episodes"]

    train_pairs = _build_pairs(train_eps, cfg)
    val_pairs   = _build_pairs(val_eps, cfg)

    model  = GNNPredictor(cfg, seed=int(cfg.get("seed", 42)))
    params = _param_list(model)

    ckpt_dir  = pathlib.Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "gnn.npz"

    if warm_start and ckpt_path.exists():
        data = np.load(ckpt_path)
        for p, key in zip(params, data.files):
            p[...] = data[key]

    opt    = _Adam(params, lr=lr)
    rng    = np.random.default_rng(int(cfg.get("seed", 42)))

    def _val_loss():
        _, vl = _batch_grads(model, val_pairs, list(range(len(val_pairs))))
        return vl

    history = []
    for step in range(1, steps + 1):
        idx = rng.choice(len(train_pairs),
                         size=min(batch_size, len(train_pairs)),
                         replace=False)
        grads, loss = _batch_grads(model, train_pairs, idx)
        opt.step(params, grads)
        history.append(loss)
        if verbose and (step % 50 == 0 or step == 1):
            print(f"  step {step:4d}  loss = {loss:.6f}")
        if save_every and step % save_every == 0:
            names = ["layer1_W_msg", "layer1_b_msg", "layer1_W_upd",
                     "layer1_b_upd", "layer2_W_msg", "layer2_b_msg",
                     "layer2_W_upd", "layer2_b_upd", "W_out", "b_out"]
            np.savez_compressed(ckpt_path, **{n: p for n, p in zip(names, params)})

    final_val = _val_loss()

    # ── Save checkpoint ───────────────────────────────────────────────────────
    names = ["layer1_W_msg", "layer1_b_msg", "layer1_W_upd", "layer1_b_upd",
             "layer2_W_msg", "layer2_b_msg", "layer2_W_upd", "layer2_b_upd",
             "W_out", "b_out"]
    np.savez_compressed(ckpt_path, **{n: p for n, p in zip(names, params)})

    return {
        "train_loss": float(history[-1]),
        "val_loss":   float(final_val),
        "history":    history,
        "ckpt_path":  str(ckpt_path),
    }


def load_gnn(cfg: dict, ckpt_path: str = "checkpoints/gnn.npz") -> GNNPredictor:
    """Restore a trained GNNPredictor from checkpoint (raises if missing)."""
    path = pathlib.Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"GNN checkpoint not found: {path}")

    data = np.load(path)
    model = GNNPredictor(cfg, seed=int(cfg.get("seed", 42)))
    for p, key in zip(_param_list(model), data.files):
        p[...] = data[key]
    return model