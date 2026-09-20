"""
baselines/constant_velocity.py
-------------------------------
Constant-Velocity (CV) baseline predictor.

Assumption: each agent maintains its current velocity indefinitely.

    pos_{t+h} = pos_t + h * dt * vel_t
    vel_{t+h} = vel_t   (unchanged)

This is the simplest possible kinematic predictor and serves as the
lower-bound baseline. Any useful model must beat CV RMSE.

Public API
----------
CVPredictor(cfg)
    .predict(state, h) -> (h, N, 4)
"""
import numpy as np


class CVPredictor:
    """
    Constant-velocity baseline.

    Parameters
    ----------
    cfg : full config dict (uses swarm.dt and swarm.world_size)
    """

    def __init__(self, cfg: dict) -> None:
        self.dt         = float(cfg["swarm"]["dt"])
        self.world_size = float(cfg["swarm"]["world_size"])

    def predict(self, state: np.ndarray, h: int) -> np.ndarray:
        """
        Predict h steps ahead assuming constant velocity.

        Parameters
        ----------
        state : (N, 4)  [pos_x, pos_y, vel_x, vel_y]
        h     : forecast horizon (steps)

        Returns
        -------
        forecast : (h, N, 4) float32
        """
        state = np.asarray(state, dtype=np.float64)
        N = state.shape[0]
        pos = state[:, :2].copy()   # (N, 2)
        vel = state[:, 2:].copy()   # (N, 2)

        forecast = np.empty((h, N, 4), dtype=np.float32)
        for step in range(h):
            pos = pos + self.dt * vel
            # Toroidal wrap to stay in bounds
            pos = pos % self.world_size
            forecast[step, :, :2] = pos
            forecast[step, :, 2:] = vel

        return forecast