"""
sim/graph.py
------------
Convert a raw swarm state frame into a graph representation.

Graph format (PyG-compatible, numpy arrays)
-------------------------------------------
node_features : (N, 4)   float32  [pos_x, pos_y, vel_x, vel_y]
edge_index    : (2, E)   int64    [source_indices; target_indices]
edge_attr     : (E, 1)   float32  [Euclidean distance between the pair]

An edge (i -> j) is added whenever dist(i, j) <= radius.
Self-loops are EXCLUDED (i != j enforced).
The graph is directed but symmetric (both i->j and j->i included).

Public API
----------
build_graph(state, radius)           -> dict
build_graph_sequence(traj, radius)   -> list[dict]   length T
"""
import numpy as np


def build_graph(
    state: np.ndarray,
    radius: float,
) -> dict:
    """
    Build a proximity graph from a single state frame.

    Parameters
    ----------
    state  : (N, 4) float array  [pos_x, pos_y, vel_x, vel_y]
    radius : float  connection radius in world units

    Returns
    -------
    dict with keys:
        "node_features" : (N, 4) float32
        "edge_index"    : (2, E) int64   — row0=src, row1=dst
        "edge_attr"     : (E, 1) float32 — Euclidean distance
        "num_nodes"     : int
        "num_edges"     : int  (= E)
    """
    state = np.asarray(state, dtype=np.float32)
    if state.ndim != 2 or state.shape[1] != 4:
        raise ValueError(
            f"state must be shape (N, 4), got {state.shape}"
        )

    N = state.shape[0]
    pos = state[:, :2]   # (N, 2)

    # --- Pairwise distances (vectorised) ---
    # delta[i, j] = pos[j] - pos[i], shape (N, N, 2)
    delta = pos[np.newaxis, :, :] - pos[:, np.newaxis, :]
    dist  = np.linalg.norm(delta, axis=2)              # (N, N)

    # --- Edge mask: within radius, no self-loops ---
    mask = (dist <= radius) & (np.arange(N)[:, None] != np.arange(N)[None, :])

    src, dst = np.where(mask)                          # (E,), (E,)
    edge_index = np.stack([src, dst], axis=0).astype(np.int64)   # (2, E)
    edge_dist  = dist[src, dst].astype(np.float32)[:, None]      # (E, 1)

    return {
        "node_features": state,                        # (N, 4)
        "edge_index":    edge_index,                   # (2, E)
        "edge_attr":     edge_dist,                    # (E, 1)
        "num_nodes":     N,
        "num_edges":     int(src.shape[0]),
    }


def build_graph_sequence(
    traj: np.ndarray,
    radius: float,
) -> list[dict]:
    """
    Build a graph for every timestep in a trajectory.

    Parameters
    ----------
    traj   : (T, N, 4) float array
    radius : float  connection radius

    Returns
    -------
    list of T graph dicts (same format as build_graph output)
    """
    traj = np.asarray(traj, dtype=np.float32)
    if traj.ndim != 3 or traj.shape[2] != 4:
        raise ValueError(
            f"traj must be shape (T, N, 4), got {traj.shape}"
        )
    T = traj.shape[0]
    return [build_graph(traj[t], radius) for t in range(T)]