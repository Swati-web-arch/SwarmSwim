"""
models/gnn.py
-------------
Graph Neural Network predictor for swarm trajectory forecasting.

Architecture
------------
Two-layer message-passing network (pure numpy, no framework dependency).

Each MP layer:
  1. For each edge (i->j): compute message  m_ij = W_msg * [h_i || h_j || e_ij]
  2. Aggregate incoming messages per node:  a_i  = sum_{j in N(i)} tanh(m_ij)
  3. Update node state:                     h_i' = tanh(W_upd * [h_i || a_i])

Output head:
  delta_x_i = W_out * h_i'   (predict residual delta, not absolute position)

Equivariance guarantee
----------------------
The aggregation step (sum over neighbours) is permutation-equivariant:
relabelling agents permutes outputs in the same way. Tested explicitly.

Public API
----------
GNNPredictor(cfg)
    .forward(graph)          graph: dict from build_graph()  -> (N, 4) delta
    .predict(state, graph, h) -> (h, N, 4) multi-step forecast
"""
import numpy as np
from sim.graph import build_graph


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class MPLayer:
    """
    Single message-passing layer.

    Parameters
    ----------
    in_dim   : node feature dimension
    edge_dim : edge feature dimension (edge_attr columns)
    out_dim  : output node feature dimension
    seed     : rng seed for weight init
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int, seed: int = 0) -> None:
        self.in_dim   = in_dim
        self.edge_dim = edge_dim
        self.out_dim  = out_dim

        rng = np.random.default_rng(seed)
        scale_msg = np.sqrt(2.0 / (2 * in_dim + edge_dim))
        scale_upd = np.sqrt(2.0 / (in_dim + out_dim))

        # Message MLP: [h_i(in_dim) || h_j(in_dim) || e_ij(edge_dim)] -> out_dim
        self.W_msg = rng.standard_normal((out_dim, 2 * in_dim + edge_dim)) * scale_msg
        self.b_msg = np.zeros(out_dim)

        # Update MLP: [h_i(in_dim) || agg(out_dim)] -> out_dim
        self.W_upd = rng.standard_normal((out_dim, in_dim + out_dim)) * scale_upd
        self.b_upd = np.zeros(out_dim)

    def forward(
        self,
        node_feat: np.ndarray,    # (N, in_dim)
        edge_index: np.ndarray,   # (2, E)
        edge_attr: np.ndarray,    # (E, edge_dim)
    ) -> np.ndarray:              # (N, out_dim)
        N = node_feat.shape[0]
        E = edge_index.shape[1]

        if E == 0:
            # No edges: aggregation is zero, update on self only
            agg = np.zeros((N, self.out_dim))
        else:
            src, dst = edge_index[0], edge_index[1]
            h_src = node_feat[src]              # (E, in_dim)
            h_dst = node_feat[dst]              # (E, in_dim)

            # Build message inputs: [h_src || h_dst || e_attr]
            msg_in = np.concatenate([h_src, h_dst, edge_attr], axis=1)  # (E, 2d+edge)
            msgs   = _tanh(msg_in @ self.W_msg.T + self.b_msg)           # (E, out_dim)

            # Scatter-sum: aggregate messages at destination nodes
            agg = np.zeros((N, self.out_dim))
            np.add.at(agg, dst, msgs)           # (N, out_dim)

        # Update
        upd_in = np.concatenate([node_feat, agg], axis=1)   # (N, in+out)
        h_out  = _tanh(upd_in @ self.W_upd.T + self.b_upd)  # (N, out_dim)
        return h_out


class GNNPredictor:
    """
    Two-layer GNN for per-agent delta-state prediction.

    Parameters
    ----------
    cfg  : full config dict
    seed : weight initialisation seed
    """

    STATE_DIM = 4    # [pos_x, pos_y, vel_x, vel_y]
    EDGE_DIM  = 1    # [distance]

    def __init__(self, cfg: dict, seed: int = 0) -> None:
        self.cfg    = cfg
        self._seed  = seed
        self._radius = float(cfg["physics"]["alignment_radius"])

        # Hidden dimension: use 2x obs_dim from koopman config as hidden size
        hidden = int(cfg["koopman"]["obs_dim"]) * 2

        # Two MP layers
        self.layer1 = MPLayer(
            in_dim=self.STATE_DIM, edge_dim=self.EDGE_DIM,
            out_dim=hidden, seed=seed
        )
        self.layer2 = MPLayer(
            in_dim=hidden, edge_dim=self.EDGE_DIM,
            out_dim=hidden, seed=seed + 1
        )

        # Output head: hidden -> STATE_DIM  (predict delta)
        # Zero-initialised: the untrained network therefore outputs the
        # zero-delta predictor (already a decent one-step baseline), and
        # training only has to learn CORRECTIONS on top of it.
        rng = np.random.default_rng(seed + 2)
        self.W_out = np.zeros((self.STATE_DIM, hidden))
        self.b_out = np.zeros(self.STATE_DIM)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, graph: dict) -> np.ndarray:
        """
        Single-step delta prediction from a graph snapshot.

        Parameters
        ----------
        graph : dict from sim.graph.build_graph()

        Returns
        -------
        delta : (N, 4) float32  predicted state residual
        """
        node_feat  = graph["node_features"].astype(np.float64)   # (N, 4)
        edge_index = graph["edge_index"]                           # (2, E)
        edge_attr  = graph["edge_attr"].astype(np.float64)        # (E, 1)

        # Normalize node features: positions by world size, velocities by
        # max speed -> all features O(1). (Previously hardcoded [100,100,3,3],
        # inconsistent with cfg.)
        world  = float(self.cfg["swarm"]["world_size"])
        max_sp = float(self.cfg["physics"]["max_speed"])
        norm = np.array([world, world, max_sp, max_sp])
        h = node_feat / norm                                       # (N, 4)

        # Two MP layers
        h = self.layer1.forward(h, edge_index, edge_attr)         # (N, hidden)
        h = self.layer2.forward(h, edge_index, edge_attr)         # (N, hidden)

        # Output head: delta in NORMALIZED units -> convert back to raw units
        # so callers can do `state + delta` directly.
        delta = h @ self.W_out.T + self.b_out                     # (N, 4)
        delta = delta * norm
        return delta.astype(np.float32)

    # ── Multi-step forecast ───────────────────────────────────────────────────

    def predict(
        self,
        state: np.ndarray,
        h: int,
        radius: float | None = None,
    ) -> np.ndarray:
        """
        h-step autoregressive forecast.

        Parameters
        ----------
        state  : (N, 4) initial state
        h      : number of forecast steps
        radius : graph connection radius (defaults to alignment_radius)

        Returns
        -------
        forecast : (h, N, 4) float32
        """
        if radius is None:
            radius = self._radius

        state = np.asarray(state, dtype=np.float32)
        world = float(self.cfg["swarm"]["world_size"])
        max_sp = float(self.cfg["physics"]["max_speed"])
        forecast = np.empty((h, state.shape[0], 4), dtype=np.float32)

        cur = state.copy()
        for step in range(h):
            # toroidal proximity graph — matches SwarmEnv's wrap-around physics
            graph = build_graph(cur, radius, world_size=world)
            # feature normalization happens inside forward()
            delta = self.forward(graph)             # (N, 4)  raw units
            cur   = cur + delta
            # Wrap positions on the torus (clip ≠ sim physics; clip causes
            # stuck-at-wall artifacts and compounds over the rollout)
            cur[:, :2] = np.mod(cur[:, :2], world)
            speeds = np.linalg.norm(cur[:, 2:], axis=1, keepdims=True)
            scale  = np.where(speeds > max_sp, max_sp / speeds.clip(min=1e-9), 1.0)
            cur[:, 2:] = cur[:, 2:] * scale
            forecast[step] = cur

        return forecast