"""
Phase 0 scaffolding tests.
Verifies: packages importable, config loads correctly, seed is 42.
"""
import importlib
import pathlib
import yaml
import pytest


def test_sim_importable():
    mod = importlib.import_module("sim")
    assert mod is not None, "sim package not importable"


def test_models_importable():
    mod = importlib.import_module("models")
    assert mod is not None, "models package not importable"


def test_baselines_importable():
    mod = importlib.import_module("baselines")
    assert mod is not None, "baselines package not importable"


def test_default_config_loads():
    cfg_path = pathlib.Path("configs") / "default.yaml"
    assert cfg_path.exists(), f"Config not found at {cfg_path}"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert isinstance(cfg, dict), "Config did not parse to a dict"


def test_seed_is_42():
    cfg_path = pathlib.Path("configs") / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    assert cfg["seed"] == 42, f"Expected seed=42, got {cfg['seed']}"


def test_required_top_level_keys():
    cfg_path = pathlib.Path("configs") / "default.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    required = {"seed", "swarm", "physics", "rssi", "koopman", "noise"}
    missing = required - cfg.keys()
    assert not missing, f"Missing config keys: {missing}"
