"""
experiments/evaluate.py
-----------------------
Full evaluation pipeline: compare Koopman, GNN, CV, and MF predictors
over the validation dataset at multiple forecast horizons.

Outputs
-------
results/rmse_table.csv   -- RMSE at h in {1,5,10,20,30} for each model
results/rmse_curve.png   -- Publication-quality line plot of RMSE vs horizon

Usage (run directly)
--------------------
    python experiments/evaluate.py

Public API
----------
run_evaluation(cfg, ckpt_dir, data_dir, out_dir, horizons) -> pd.DataFrame
compute_rmse(forecast, ground_truth)                        -> float
"""
import pathlib
import sys
import numpy as np

# Allow imports from project root
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import csv
import yaml

from sim.env import SwarmEnv
from sim.dataset import generate_dataset, load_dataset, rollout
from sim.graph import build_graph
from models.train import train_koopman, load_koopman
from models.gnn import GNNPredictor
from baselines.constant_velocity import CVPredictor
from baselines.mean_field import MeanFieldPredictor


# ── RMSE helper ───────────────────────────────────────────────────────────────

def compute_rmse(forecast: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute position RMSE between forecast and ground truth.

    Parameters
    ----------
    forecast     : (h, N, 4) or (N, 4)  predicted states
    ground_truth : same shape as forecast

    Returns
    -------
    rmse : float  root mean squared error on positions only (first 2 cols)
    """
    f = np.asarray(forecast,     dtype=np.float64)
    g = np.asarray(ground_truth, dtype=np.float64)
    pos_err = f[..., :2] - g[..., :2]
    return float(np.sqrt(np.mean(pos_err ** 2)))


# ── Main evaluation ───────────────────────────────────────────────────────────

def run_evaluation(
    cfg: dict,
    ckpt_dir:  str = "checkpoints",
    data_dir:  str = "data",
    out_dir:   str = "results",
    horizons: list = None,
    n_eval_episodes: int = 20,
) -> list[dict]:
    """
    Evaluate all models on the val dataset at multiple horizons.

    Parameters
    ----------
    cfg              : config dict
    ckpt_dir         : directory containing koopman.npz
    data_dir         : directory containing train.npz / val.npz
    out_dir          : where to save rmse_table.csv
    horizons         : list of forecast horizons to evaluate
    n_eval_episodes  : number of episodes to generate for evaluation

    Returns
    -------
    rows : list of dicts  [{"model": str, "horizon": int, "rmse": float}, ...]
    """
    if horizons is None:
        horizons = [1, 5, 10, 20, 30]

    T = cfg["swarm"]["rollout_horizon"]
    max_h = max(horizons)
    if max_h >= T:
        raise ValueError(
            f"Max horizon {max_h} >= rollout_horizon {T}. "
            f"Increase swarm.rollout_horizon in config."
        )

    # ── Generate / load val data ──────────────────────────────────────────────
    generate_dataset(cfg, n_episodes=n_eval_episodes,
                     val_fraction=0.4, out_dir=data_dir, seed_offset=1000)
    val_data = load_dataset("val", out_dir=data_dir)["episodes"]  # (E, T, N, 4)

    # ── Instantiate models ────────────────────────────────────────────────────
    ckpt_path = pathlib.Path(ckpt_dir) / "koopman.npz"
    koopman = load_koopman(cfg, ckpt_path=str(ckpt_path))
    gnn     = GNNPredictor(cfg, seed=int(cfg.get("seed", 42)))
    cv      = CVPredictor(cfg)
    mf      = MeanFieldPredictor(cfg)

    radius = float(cfg["physics"]["alignment_radius"])

    models = {
        "Koopman": koopman,
        "GNN":     gnn,
        "CV":      cv,
        "MeanField": mf,
    }

    rows = []

    for h in horizons:
        rmse_accum = {name: [] for name in models}

        for ep in val_data:
            # Use first frame as seed, evaluate against frame h
            state_0  = ep[0]        # (N, 4)
            truth_h  = ep[h]        # (N, 4) ground truth at horizon h

            for name, model in models.items():
                if name == "GNN":
                    fc = model.predict(state_0, h=h, radius=radius)  # (h, N, 4)
                else:
                    fc = model.predict(state_0, h=h)                  # (h, N, 4)

                rmse = compute_rmse(fc[h - 1], truth_h)
                rmse_accum[name].append(rmse)

        for name in models:
            mean_rmse = float(np.mean(rmse_accum[name]))
            rows.append({"model": name, "horizon": h, "rmse": mean_rmse})
            print(f"  h={h:2d}  {name:<12s}  RMSE = {mean_rmse:.4f} m")

    # ── Save CSV ──────────────────────────────────────────────────────────────
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "rmse_table.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "horizon", "rmse"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {csv_path}")
    return rows


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_results(
    csv_path: str = "results/rmse_table.csv",
    out_path: str = "results/rmse_curve.png",
) -> None:
    """
    Read rmse_table.csv and produce a publication-quality RMSE vs horizon plot.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Parse CSV manually (no pandas needed)
    data: dict[str, dict[int, float]] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model   = row["model"]
            horizon = int(row["horizon"])
            rmse    = float(row["rmse"])
            if model not in data:
                data[model] = {}
            data[model][horizon] = rmse

    styles = {
        "Koopman":   {"color": "#2196F3", "marker": "o", "lw": 2.5, "ls": "-"},
        "GNN":       {"color": "#9C27B0", "marker": "s", "lw": 2.5, "ls": "-"},
        "CV":        {"color": "#FF9800", "marker": "^", "lw": 1.5, "ls": "--"},
        "MeanField": {"color": "#F44336", "marker": "D", "lw": 1.5, "ls": ":"},
    }

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)

    for model, hmap in sorted(data.items()):
        hs   = sorted(hmap.keys())
        rmse = [hmap[h] for h in hs]
        st   = styles.get(model, {"color": "gray", "marker": "x", "lw": 1.5, "ls": "-"})
        ax.plot(hs, rmse, label=model,
                color=st["color"], marker=st["marker"],
                linewidth=st["lw"], linestyle=st["ls"],
                markersize=6, zorder=3)

    ax.set_xlabel("Forecast Horizon (steps)", fontsize=12)
    ax.set_ylabel("Position RMSE (m)",        fontsize=12)
    ax.set_title("Multi-Step Prediction RMSE: Swarm Trajectory Forecasting",
                 fontsize=12, pad=10)
    ax.legend(fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI entry-point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg_path = pathlib.Path("configs") / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    # Make rollout_horizon large enough for h=30
    cfg["swarm"]["rollout_horizon"] = 35

    print("=== Training Koopman on fresh dataset ===")
    metrics = train_koopman(cfg, n_episodes=80,
                             out_dir="checkpoints", data_dir="data")
    print(f"  train_mse={metrics['train_mse']:.4f}  "
          f"val_mse={metrics['val_mse']:.4f}  "
          f"spectral_r={metrics['spectral_r']:.4f}")

    print("\n=== Evaluation ===")
    rows = run_evaluation(cfg, horizons=[1, 5, 10, 20, 30],
                          n_eval_episodes=40)

    print("\n=== Plotting ===")
    plot_results()
    print("\nDone. Open results/rmse_curve.png")