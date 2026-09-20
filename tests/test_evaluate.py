"""
tests/test_evaluate.py
----------------------
Unit tests for Phase 8: experiments/evaluate.py
"""
import pathlib
import copy
import csv
import numpy as np
import yaml
import pytest
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from models.train import train_koopman
from experiments.evaluate import run_evaluation, compute_rmse, plot_results


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    path = pathlib.Path("configs") / "default.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.fixture
def tiny_cfg(cfg):
    c = copy.deepcopy(cfg)
    c["swarm"]["N"] = 8
    c["swarm"]["rollout_horizon"] = 35
    c["koopman"]["obs_dim"] = 10
    c["koopman"]["forecast_steps"] = 5
    return c


@pytest.fixture
def trained_env(tiny_cfg, tmp_path):
    """Train koopman and return paths + cfg."""
    metrics = train_koopman(
        tiny_cfg, n_episodes=8,
        out_dir=str(tmp_path / "ckpt"),
        data_dir=str(tmp_path / "data"),
    )
    return tiny_cfg, tmp_path, metrics


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_compute_rmse_zero():
    """RMSE of identical arrays must be 0."""
    arr = np.random.default_rng(0).uniform(size=(5, 4)).astype(np.float32)
    assert compute_rmse(arr, arr) == pytest.approx(0.0, abs=1e-8)


def test_compute_rmse_known():
    """RMSE should match hand-computed value."""
    pred  = np.array([[3.0, 4.0, 0.0, 0.0]])   # 1 agent, pos error = (1,1)
    truth = np.array([[2.0, 3.0, 0.0, 0.0]])
    # pos errors: [1, 1]  -> sqrt(mean([1,1])) = 1.0
    assert compute_rmse(pred, truth) == pytest.approx(1.0, rel=1e-5)


def test_csv_created(trained_env, tmp_path):
    """run_evaluation() must create results/rmse_table.csv."""
    cfg, tpath, _ = trained_env
    run_evaluation(
        cfg,
        ckpt_dir=str(tpath / "ckpt"),
        data_dir=str(tpath / "data"),
        out_dir=str(tpath / "results"),
        horizons=[1, 5],
        n_eval_episodes=6,
    )
    csv_path = tpath / "results" / "rmse_table.csv"
    assert csv_path.exists(), "rmse_table.csv not created"


def test_csv_has_all_models(trained_env, tmp_path):
    """CSV must have rows for Koopman, GNN, CV, MeanField."""
    cfg, tpath, _ = trained_env
    run_evaluation(
        cfg,
        ckpt_dir=str(tpath / "ckpt"),
        data_dir=str(tpath / "data"),
        out_dir=str(tpath / "results"),
        horizons=[1, 5],
        n_eval_episodes=6,
    )
    csv_path = tpath / "results" / "rmse_table.csv"
    models_found = set()
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            models_found.add(row["model"])
    required = {"Koopman", "GNN", "CV", "MeanField"}
    assert required.issubset(models_found), (
        f"Missing models in CSV: {required - models_found}"
    )


def test_rmse_values_finite(trained_env, tmp_path):
    """All RMSE values in the CSV must be finite non-negative numbers."""
    cfg, tpath, _ = trained_env
    rows = run_evaluation(
        cfg,
        ckpt_dir=str(tpath / "ckpt"),
        data_dir=str(tpath / "data"),
        out_dir=str(tpath / "results"),
        horizons=[1, 5],
        n_eval_episodes=6,
    )
    for row in rows:
        rmse = row["rmse"]
        assert np.isfinite(rmse) and rmse >= 0, (
            f"Invalid RMSE for {row['model']} h={row['horizon']}: {rmse}"
        )


def test_plot_created(trained_env, tmp_path):
    """plot_results() must create a PNG file."""
    cfg, tpath, _ = trained_env
    run_evaluation(
        cfg,
        ckpt_dir=str(tpath / "ckpt"),
        data_dir=str(tpath / "data"),
        out_dir=str(tpath / "results"),
        horizons=[1, 5],
        n_eval_episodes=6,
    )
    csv_path = str(tpath / "results" / "rmse_table.csv")
    png_path = str(tpath / "results" / "rmse_curve.png")
    plot_results(csv_path=csv_path, out_path=png_path)
    assert pathlib.Path(png_path).exists(), "rmse_curve.png not created"
    assert pathlib.Path(png_path).stat().st_size > 1000, "PNG suspiciously small"