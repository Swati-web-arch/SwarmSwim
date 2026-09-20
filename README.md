# Swarm Latent Trajectory Prediction (Review 2 Package)

**Course:** Innovation Design Project (IDP), VIT Chennai  
**Faculty Guide:** Dr. Suneesh Jacob  
**Team Members:**  
- Swati Yadav (25BRS1088)  
- Thati V S Sram (25BRS1159)  
- Ayush Choudhary (25BRS1209)  

---

## 📂 Project Architecture

```
swarm_review2/
├── sim/
│   ├── env.py                  # Swarm physics engine (Reynolds Boids)
│   ├── formations.py           # Swarm formations (random, circle, grid, two_clusters)
│   ├── dataset.py              # Offline trajectory dataset generator & loader
│   └── graph.py                # Dynamic proximity graph builder
├── models/
│   ├── koopman.py              # Extended Dynamic Mode Decomposition (EDMD) + RLS
│   ├── gnn.py                  # Permutation-equivariant Message-Passing GNN
│   └── train.py                # Offline training & checkpointing
├── baselines/
│   ├── constant_velocity.py    # Kinematic constant-velocity baseline
│   └── mean_field.py           # Centroid-pull cohesion baseline
├── configs/
│   └── default.yaml            # Centralized hyperparameter configuration
├── experiments/
│   └── evaluate.py             # Multi-horizon benchmark pipeline & plot generator
├── tests/
│   ├── test_scaffolding.py     # Environment & config validation
│   ├── test_sim.py             # Physics & boundary constraint tests
│   ├── test_dataset.py         # Offline dataset shape & integrity tests
│   ├── test_graph.py           # Proximity graph topology & distance tests
│   ├── test_koopman.py         # Koopman lifting, fitting & stability tests
│   ├── test_gnn.py             # GNN equivariance & residual prediction tests
│   ├── test_baselines.py       # CV and MF baseline mathematical tests
│   ├── test_train.py           # Training checkpoint save/load tests
│   └── test_evaluate.py        # Benchmark evaluation & plotting tests
├── docs/
│   └── results_report.md       # Comprehensive Review 2 academic report
├── data/                       # Generated .npz datasets
├── checkpoints/                # Saved model weights (.npz)
├── results/                    # Output CSV tables and PNG curves
├── dashboard.html              # Standalone interactive 60-FPS HUD Visualizer
├── run_pipeline.py             # One-command full test + experiment runner
├── run.bat                     # Windows 1-click launcher
└── pyproject.toml              # Project dependencies and config
```

---

## 🚀 Quick Start (Running Everything)

Open PowerShell inside this directory:
```powershell
cd C:\Users\Lenovo\Downloads\swarm_review2
```

### 1. Run All Automated Tests (64 unit tests)
```powershell
python -m pytest tests/ -v
```

### 2. Run the Full Pipeline (Training + Multi-Horizon Evaluation + Plotting)
```powershell
python run_pipeline.py
```
Or run the experiment script directly:
```powershell
python experiments/evaluate.py
```

### 3. Generated Deliverables
- **Academic Technical Report:** `docs/results_report.md`
- **Numerical RMSE Benchmark Table:** `results/rmse_table.csv`
- **Publication-ready Plot:** `results/rmse_curve.png`

### 4. Interactive Live Visualizer
- Double-click **`dashboard.html`** in File Explorer to launch the 60-FPS Cyber HUD in your browser.
- Real-time features: swarm boids dynamics, obstacle deflection, Koopman forecast ribbons, cluster hulls, and RSSI link telemetry.