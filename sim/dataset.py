"""
sim/dataset.py
--------------
Offline dataset generation for the swarm trajectory prediction task.

Core format
-----------
A single episode is a tensor of shape (T, N, 4) where:
    axis-0 : timestep  (0 .. T-1)
    axis-1 : agent     (0 .. N-1)
    axis-2 : features  [pos_x, pos_y, vel_x, vel_y]

Public API
----------
rollout(env, T)                    -> np.ndarray  shape (T, N, 4)
generate_dataset(cfg, n_episodes)  -> saves data/train.npz + data/val.npz
load_dataset(split)                -> dict {"episodes": (E, T, N, 4)}
"""
import pathlib
import numpy as np
from sim.env import SwarmEnv


# ── Core rollout ─────────────────────────────────────────────────────────────

def rollout(env: SwarmEnv, T: int) -> np.ndarray:
    """
    Run the environment for T steps and record the full trajectory.

    Parameters
    ----------
    env : initialised SwarmEnv (already reset externally or will be reset here)
    T   : number of timesteps to record

    Returns
    -------
    traj : np.ndarray, shape (T, N, 4), dtype float32
           columns: [pos_x, pos_y, vel_x, vel_y]
    """
    env.reset()
    N = env.N
    traj = np.empty((T, N, 4), dtype=np.float32)

    for t in range(T):
        pos, vel = env.step()
        traj[t, :, 0:2] = pos
        traj[t, :, 2:4] = vel

    return traj


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset(
    cfg: dict,
    n_episodes: int = 200,
    val_fraction: float = 0.2,
    out_dir: str = "data",
    seed_offset: int = 0,
) -> dict[str, pathlib.Path]:
    """
    Generate train/val episode tensors and save as compressed .npz files.

    Parameters
    ----------
    cfg          : config dict (from configs/default.yaml)
    n_episodes   : total number of episodes to simulate
    val_fraction : fraction of episodes held out for validation
    out_dir      : directory to write train.npz and val.npz
    seed_offset  : added to per-episode seed to avoid collision across runs

    Returns
    -------
    paths : {"train": Path, "val": Path}
    """
    T = int(cfg["swarm"]["rollout_horizon"])
    n_val = max(1, int(round(n_episodes * val_fraction)))
    n_train = n_episodes - n_val

    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    train_eps: list[np.ndarray] = []
    val_eps: list[np.ndarray]   = []

    for ep_idx in range(n_episodes):
        ep_seed = seed_offset + ep_idx
        env = SwarmEnv(cfg, seed=ep_seed)
        traj = rollout(env, T)

        if ep_idx < n_train:
            train_eps.append(traj)
        else:
            val_eps.append(traj)

    # Stack to (E, T, N, 4)
    train_arr = np.stack(train_eps, axis=0)   # (n_train, T, N, 4)
    val_arr   = np.stack(val_eps,   axis=0)   # (n_val,   T, N, 4)

    train_path = out_path / "train.npz"
    val_path   = out_path / "val.npz"

    np.savez_compressed(train_path, episodes=train_arr)
    np.savez_compressed(val_path,   episodes=val_arr)

    return {"train": train_path, "val": val_path}


# ── Dataset loading ───────────────────────────────────────────────────────────

def load_dataset(split: str = "train", out_dir: str = "data") -> dict:
    """
    Load a previously generated dataset split.

    Parameters
    ----------
    split   : "train" or "val"
    out_dir : directory where .npz files live

    Returns
    -------
    dict with key "episodes" -> np.ndarray shape (E, T, N, 4)
    """
    allowed = {"train", "val"}
    if split not in allowed:
        raise ValueError(f"split must be one of {allowed}, got {split!r}")

    path = pathlib.Path(out_dir) / f"{split}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {path}. "
            f"Run generate_dataset() first."
        )

    data = np.load(path)
    return {"episodes": data["episodes"]}