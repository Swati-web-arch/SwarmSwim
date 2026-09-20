"""
sim/env.py
----------
SwarmEnv: vectorised Reynolds boids simulation.

Physics rules (all vectorised, no Python loops over agents):
  - Separation : steer away from neighbours within separation_radius
  - Alignment  : match average velocity of neighbours within alignment_radius
  - Cohesion   : steer toward centroid of neighbours within cohesion_radius
  - Speed cap  : clip velocity magnitude to max_speed
  - Boundary   : toroidal wrap (positions wrap at world_size)
"""
import numpy as np
from sim.formations import make_formation


class SwarmEnv:
    """
    Vectorised Reynolds boids swarm environment.

    Parameters
    ----------
    cfg  : dict loaded from configs/default.yaml
    seed : random seed (overrides cfg["seed"] if provided)
    """

    def __init__(self, cfg: dict, seed: int | None = None) -> None:
        self.cfg = cfg
        sw = cfg["swarm"]
        ph = cfg["physics"]

        self.N = int(sw["N"])
        self.world_size = float(sw["world_size"])
        self.dt = float(sw["dt"])
        self.formation = cfg.get("formation", "random")

        self.max_speed = float(ph["max_speed"])
        self.sep_r = float(ph["separation_radius"])
        self.ali_r = float(ph["alignment_radius"])
        self.coh_r = float(ph["cohesion_radius"])
        self.w_sep = float(ph["w_sep"])
        self.w_ali = float(ph["w_ali"])
        self.w_coh = float(ph["w_coh"])

        self._seed = seed if seed is not None else int(cfg.get("seed", 42))
        self._rng = np.random.default_rng(self._seed)

        self.pos: np.ndarray = np.zeros((self.N, 2), dtype=np.float64)
        self.vel: np.ndarray = np.zeros((self.N, 2), dtype=np.float64)
        self.t: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> tuple[np.ndarray, np.ndarray]:
        """Reset environment. Returns (pos, vel) each shape (N, 2)."""
        self._rng = np.random.default_rng(self._seed)
        self.pos, self.vel = make_formation(
            self.formation, self.N, self.world_size, self._rng
        )
        self.t = 0
        return self.pos.copy(), self.vel.copy()

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Advance one timestep using vectorised boids rules.
        Returns updated (pos, vel) each shape (N, 2).
        """
        sep, ali, coh = self._compute_forces()

        acc = self.w_sep * sep + self.w_ali * ali + self.w_coh * coh
        self.vel = self.vel + acc * self.dt
        self.vel = self._cap_speed(self.vel)
        self.pos = (self.pos + self.vel * self.dt) % self.world_size
        self.t += 1
        return self.pos.copy(), self.vel.copy()

    # ------------------------------------------------------------------
    # Internal helpers (fully vectorised)
    # ------------------------------------------------------------------

    def _pairwise_delta(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute toroidal pairwise displacement and distance matrices.

        Returns
        -------
        delta : (N, N, 2)  — delta[i,j] = pos[j] - pos[i] (toroidal)
        dist  : (N, N)     — Euclidean distance
        """
        # (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        delta = self.pos[np.newaxis, :, :] - self.pos[:, np.newaxis, :]
        # Toroidal shortest path
        delta = (delta + self.world_size / 2) % self.world_size - self.world_size / 2
        dist = np.linalg.norm(delta, axis=2)  # (N, N)
        return delta, dist

    def _compute_forces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (separation, alignment, cohesion) steering vectors, shape (N, 2)."""
        delta, dist = self._pairwise_delta()

        # Masks — exclude self (diagonal = 0, not in any radius)
        # dist == 0 on diagonal; we skip those via > 0 check in masks
        sep_mask = (dist > 0) & (dist < self.sep_r)   # (N, N)
        ali_mask = (dist > 0) & (dist < self.ali_r)
        coh_mask = (dist > 0) & (dist < self.coh_r)

        # --- Separation: steer away from close neighbours ---
        # avoid divide-by-zero (dist==0 excluded by mask)
        safe_dist = np.where(sep_mask, dist, 1.0)
        # unit vector pointing FROM neighbour TO self = -delta / dist
        away = -delta / safe_dist[:, :, np.newaxis]   # (N, N, 2)
        sep_count = sep_mask.sum(axis=1, keepdims=True).clip(min=1)  # (N,1)
        sep = (away * sep_mask[:, :, np.newaxis]).sum(axis=1) / sep_count  # (N,2)

        # --- Alignment: match average velocity of neighbours ---
        ali_count = ali_mask.sum(axis=1, keepdims=True).clip(min=1)
        nbr_vel = (self.vel[np.newaxis, :, :] * ali_mask[:, :, np.newaxis]).sum(axis=1)
        ali = nbr_vel / ali_count - self.vel  # (N, 2)

        # --- Cohesion: steer toward centroid of neighbours ---
        coh_count = coh_mask.sum(axis=1, keepdims=True).clip(min=1)
        centroid = (
            (self.pos[np.newaxis, :, :] * coh_mask[:, :, np.newaxis]).sum(axis=1)
            / coh_count
        )
        coh = centroid - self.pos  # (N, 2)

        return sep, ali, coh

    @staticmethod
    def _cap_speed(vel: np.ndarray, max_speed: float = None) -> np.ndarray:
        """Clip velocity vectors to max_speed (preserves direction)."""
        speed = np.linalg.norm(vel, axis=1, keepdims=True)  # (N,1)
        if max_speed is None:
            # use a large default; callers should pass the actual value
            return vel
        scale = np.where(speed > max_speed, max_speed / speed.clip(min=1e-9), 1.0)
        return vel * scale

    def step(self) -> tuple[np.ndarray, np.ndarray]:
        """Advance one timestep. Returns (pos, vel) shape (N, 2)."""
        sep, ali, coh = self._compute_forces()
        acc = self.w_sep * sep + self.w_ali * ali + self.w_coh * coh
        self.vel = self.vel + acc * self.dt

        # Cap speed
        speed = np.linalg.norm(self.vel, axis=1, keepdims=True)
        scale = np.where(speed > self.max_speed,
                         self.max_speed / speed.clip(min=1e-9), 1.0)
        self.vel = self.vel * scale

        self.pos = (self.pos + self.vel * self.dt) % self.world_size
        self.t += 1
        return self.pos.copy(), self.vel.copy()