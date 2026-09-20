"""
baselines/mean_field.py
-----------------------
Mean-Field (MF) baseline predictor.

Assumption: the swarm is a single coherent mass. Each agent is predicted
to move toward the global centroid at a rate proportional to its distance.

    centroid = mean(pos_t)
    vel_mf_i = alpha * (centroid - pos_i)   +  (1-alpha) * vel_i
    pos_{t+h} = pos_t + h * dt * vel_mf_i

This captures global cohesion but ignores individual agent interactions,
neighbour structure, and local separation/alignment dynamics.

Public API
----------
MeanFieldPredictor(cfg, alpha=0.1)
    .predict(state, h) -> (h, N, 4)
"""
import numpy as np


class MeanFieldPredictor:
    """
    Mean-field baseline: agents drift toward swarm centroid.

    Parameters
    ----------
    cfg   : full config dict
    alpha : cohesion strength in [0, 1]
            alpha=0 -> pure constant velocity
            alpha=1 -> pure centroid pull
    """

    def __init__(self, cfg: dict, alpha: float = 0.15) -> None:
        self.dt         = float(cfg["swarm"]["dt"])
        self.world_size = float(cfg["swarm"]["world_size"])
        self.max_speed  = float(cfg["physics"]["max_speed"])
        self.alpha      = float(alpha)

    def predict(self, state: np.ndarray, h: int) -> np.ndarray:
        """
        Predict h steps using mean-field centroid-pull dynamics.

        Parameters
        ----------
        state : (N, 4) [pos_x, pos_y, vel_x, vel_y]
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
            centroid = pos.mean(axis=0, keepdims=True)          # (1, 2)
            pull     = centroid - pos                            # (N, 2) toward centre
            vel_mf   = (1.0 - self.alpha) * vel + self.alpha * pull  # (N, 2)

            # Cap speed
            speeds = np.linalg.norm(vel_mf, axis=1, keepdims=True)
            scale  = np.where(speeds > self.max_speed,
                              self.max_speed / speeds.clip(min=1e-9), 1.0)
            vel_mf = vel_mf * scale

            pos = pos + self.dt * vel_mf
            pos = np.clip(pos, 0.0, self.world_size)

            forecast[step, :, :2] = pos
            forecast[step, :, 2:] = vel_mf
            vel = vel_mf   # carry forward

        return forecast