"""
models/train.py
---------------
Training pipeline for the Koopman predictor.

Workflow
--------
1. Generate offline dataset  (or load existing one)
2. Fit KoopmanPredictor via EDMD on train split
3. Evaluate one-step MSE on val split
4. Save checkpoint to  checkpoints/koopman.npz

Checkpoint format (numpy .npz)
-------------------------------
  K        : (d, d)   Koopman operator
  W        : (4, d)   linear readout
  centres  : (d, 4)   RBF centres
  log_bw   : scalar   log bandwidth
  obs_dim  : scalar
  reg      : scalar

Public API
----------
train_koopman(cfg, n_episodes, out_dir, data_dir)  -> metrics dict
load_koopman(cfg, ckpt_path)                       -> KoopmanPredictor
"""
import pathlib
import numpy as np

from sim.env import SwarmEnv
from sim.dataset import generate_dataset, load_dataset, rollout
from models.koopman import KoopmanPredictor


# ── Main training entry-point ─────────────────────────────────────────────────

def train_koopman(
    cfg: dict,
    n_episodes: int = 100,
    out_dir: str = "checkpoints",
    data_dir: str = "data",
    seed_offset: int = 0,
) -> dict:
    """
    Fit KoopmanPredictor on an offline dataset and save checkpoint.

    Parameters
    ----------
    cfg         : config dict
    n_episodes  : total train+val episodes to simulate
    out_dir     : where to write koopman.npz
    data_dir    : where to write/read train.npz + val.npz
    seed_offset : episode seed offset (avoids collision across runs)

    Returns
    -------
    metrics : dict with keys
        "train_mse"  : float  one-step MSE on training episodes
        "val_mse"    : float  one-step MSE on validation episodes
        "spectral_r" : float  spectral radius of fitted K
        "ckpt_path"  : str    path to saved checkpoint
    """
    # ── 1. Generate dataset ───────────────────────────────────────────────────
    generate_dataset(
        cfg,
        n_episodes=n_episodes,
        val_fraction=0.2,
        out_dir=data_dir,
        seed_offset=seed_offset,
    )

    train_data = load_dataset("train", out_dir=data_dir)["episodes"]  # (E_tr, T, N, 4)
    val_data   = load_dataset("val",   out_dir=data_dir)["episodes"]  # (E_val, T, N, 4)

    # ── 2. Fit on all training episodes (stack into one big trajectory) ───────
    model = KoopmanPredictor(cfg, seed=int(cfg.get("seed", 42)))

    # Use first training episode for the primary EDMD fit
    # Then online-update with remaining episodes for diversity
    model.fit(train_data[0].astype(np.float64))

    # RLS mini-updates from remaining train episodes (one pair per episode)
    for ep in train_data[1:]:
        T = ep.shape[0]
        t = np.random.default_rng(42).integers(0, T - 1)
        z0 = model.lift(ep[t]).mean(axis=0)
        z1 = model.lift(ep[t + 1]).mean(axis=0)
        model.update(z0, z1)

    # ── 3. Compute one-step MSE on train and val ──────────────────────────────
    train_mse = _eval_one_step_mse(model, train_data)
    val_mse   = _eval_one_step_mse(model, val_data)

    # ── 4. Save checkpoint ────────────────────────────────────────────────────
    ckpt_dir  = pathlib.Path(out_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "koopman.npz"

    np.savez_compressed(
        ckpt_path,
        K        = model.K,
        W        = model.W,
        centres  = model._centres,
        log_bw   = np.array([model._log_bw]),
        obs_dim  = np.array([model.obs_dim]),
        reg      = np.array([model.reg]),
    )

    return {
        "train_mse":  float(train_mse),
        "val_mse":    float(val_mse),
        "spectral_r": float(model.spectral_radius),
        "ckpt_path":  str(ckpt_path),
    }


# ── Checkpoint loading ────────────────────────────────────────────────────────

def load_koopman(cfg: dict, ckpt_path: str = "checkpoints/koopman.npz") -> KoopmanPredictor:
    """
    Restore a KoopmanPredictor from a saved checkpoint.

    Parameters
    ----------
    cfg       : config dict (for constructor params)
    ckpt_path : path to the .npz checkpoint file

    Returns
    -------
    model : KoopmanPredictor with K, W, centres restored
    """
    path = pathlib.Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    data  = np.load(path)
    model = KoopmanPredictor(cfg, seed=int(cfg.get("seed", 42)))

    model.K         = data["K"]
    model.W         = data["W"]
    model._centres  = data["centres"]
    model._log_bw   = float(data["log_bw"][0])
    model._P        = (1.0 / model.reg) * np.eye(model.obs_dim)
    model._fitted   = True

    return model


# ── Internal helpers ──────────────────────────────────────────────────────────

def _eval_one_step_mse(model: KoopmanPredictor, episodes: np.ndarray) -> float:
    """
    Compute mean one-step prediction MSE over all episodes and timesteps.

    Parameters
    ----------
    model    : fitted KoopmanPredictor
    episodes : (E, T, N, 4)

    Returns
    -------
    mse : float  mean squared error on position+velocity
    """
    E, T, N, _ = episodes.shape
    total_sq  = 0.0
    total_cnt = 0

    for ep in episodes:
        for t in range(T - 1):
            state_t   = ep[t]            # (N, 4)
            state_tp1 = ep[t + 1]        # (N, 4) ground truth

            pred = model.predict(state_t, h=1)[0]   # (N, 4)
            diff = pred - state_tp1.astype(np.float32)
            total_sq  += float(np.sum(diff ** 2))
            total_cnt += N * 4

    return total_sq / max(total_cnt, 1)