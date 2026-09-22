"""
models/koopman.py
-----------------
Koopman operator predictor for swarm trajectory forecasting.

Theory
------
Nonlinear dynamics  x_{t+1} = F(x_t)  are hard to predict long-horizon.
Koopman theory says there exists a lifting map  psi: R^n -> R^d  (d >> n)
such that the lifted dynamics ARE linear:

    z_{t+1} = K * z_t,    z_t = psi(x_t)

We learn K offline via Extended Dynamic Mode Decomposition (EDMD):
    K = argmin ||Z' - K Z||_F  -->  K = Z' Z^+ (least squares)

Then predict:
    z_hat_{t+h} = K^h z_t  -->  x_hat = psi_inv(z_hat)  (linear projection back)

Lifting
-------
We use a normalised RBF feature map with fixed random centres, giving
a smooth, nonlinear but fixed embedding. The inverse map is a linear
regression (pseudo-inverse) learned from the same training data.

Online update
-------------
After deployment, we support Recursive Least Squares (RLS) to adapt K
incrementally without storing the full dataset.

Public API
----------
KoopmanPredictor(cfg)
    .fit(traj)          traj: (T, N, 4)  -- batch EDMD
    .predict(state, h)  state: (N, 4)    -- h-step forecast -> (h, N, 4)
    .update(z, z_next)  online RLS update of K
    .lift(state)        state: (N, 4)    -> z: (N, obs_dim) lifted features
"""
import numpy as np


class KoopmanPredictor:
    """
    Koopman operator predictor with RBF lifting and EDMD fitting.

    Parameters
    ----------
    cfg : dict  -- full config dict (uses cfg["koopman"] sub-dict)
    seed : int  -- seed for random RBF centres
    """

    def __init__(self, cfg: dict, seed: int | None = None) -> None:
        kp = cfg["koopman"]
        self.obs_dim = int(kp["obs_dim"])          # lifted space dimension d
        self.reg     = float(kp["reg"])            # Tikhonov regularisation lambda
        self.forgetting = float(kp["rls_forgetting"])   # RLS forgetting factor
        self.forecast_steps = int(kp["forecast_steps"]) # default prediction horizon

        # Input dimension: per-agent state dim = 4 (pos_x, pos_y, vel_x, vel_y)
        self.state_dim = 4
        self.world_size = float(cfg["swarm"]["world_size"])
        self.v_max      = float(cfg["physics"]["max_speed"])

        if self.obs_dim < 6:
            raise ValueError(
                "koopman.obs_dim must be >= 6: the toroidal Fourier basis "
                "uses 6 base features (sin/cos of x and y, vx, vy)."
            )

        # Random RBF centres on the 6-D continuous base representation
        # (used only when obs_dim > 6 to add extra lifted features)
        _seed = seed if seed is not None else int(cfg.get("seed", 42))
        rng = np.random.default_rng(_seed)
        self._centres = rng.uniform(
            -1.0, 1.0, (self.obs_dim - 6, 6)
        ).astype(np.float64)
        self._log_bw  = 0.0   # log bandwidth (learnable later; start at 1.0)

        # Koopman operator K: (obs_dim, obs_dim) -- filled by fit()
        self.K: np.ndarray | None = None

        # Linear readout W: obs_dim -> state_dim  -- filled by fit()
        self.W: np.ndarray | None = None

        # RLS state (P matrix for K rows)
        self._P: np.ndarray | None = None

        self._fitted = False

    # ── Lifting ───────────────────────────────────────────────────────────────

    def _toroidal_base(self, state: np.ndarray) -> np.ndarray:
        """
        Continuous toroidal embedding of a raw state frame (N, 6).

        The simulation wraps positions at world_size (toroidal arena), so RAW
        coordinates are DISCONTINUOUS at the boundary (x: 99.9 -> 0.1). Any
        basis built on raw coordinates inherits that discontinuity and wrecks
        the linear-operator fit. Embedding the position angle theta = 2*pi*x/L
        as (sin, cos) is continuous everywhere on the torus:

            z_base = [sin(2*pi*x/L), cos(2*pi*x/L),
                      sin(2*pi*y/L), cos(2*pi*y/L),
                      vx / v_max,    vy / v_max]
        """
        state = np.asarray(state, dtype=np.float64)
        L  = self.world_size
        vm = self.v_max
        th_x = 2.0 * np.pi * state[:, 0] / L
        th_y = 2.0 * np.pi * state[:, 1] / L
        return np.column_stack([
            np.sin(th_x), np.cos(th_x),
            np.sin(th_y), np.cos(th_y),
            state[:, 2] / vm, state[:, 3] / vm,
        ])

    def _decode(self, Z: np.ndarray) -> np.ndarray:
        """
        Decode lifted features (N, obs_dim) back to raw states (N, 4).

        Positions are recovered from the Fourier pair via atan2 (exact
        inverse of the toroidal embedding, continuous across the wrap);
        velocities from the normalised components.
        """
        L  = self.world_size
        vm = self.v_max
        x  = (np.arctan2(Z[:, 0], Z[:, 1]) * L / (2.0 * np.pi)) % L
        y  = (np.arctan2(Z[:, 2], Z[:, 3]) * L / (2.0 * np.pi)) % L
        vx = np.clip(Z[:, 4] * vm, -vm, vm)
        vy = np.clip(Z[:, 5] * vm, -vm, vm)
        return np.column_stack([x, y, vx, vy])

    def lift(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the lifting map psi to a single frame.

        First 6 dims: toroidal Fourier features (continuous across the
        periodic boundary). Remaining obs_dim-6 dims (if any): RBF features
        of the continuous base representation with fixed centres.

        Parameters
        ----------
        state : (N, 4) float  agent states

        Returns
        -------
        Z : (N, obs_dim) float64  lifted features
        """
        state = np.asarray(state, dtype=np.float64)
        base = self._toroidal_base(state)                    # (N, 6)
        if self.obs_dim == 6:
            return base
        # Extra RBF features on the wrap-continuous base representation
        bw   = np.exp(self._log_bw)
        diff = base[:, np.newaxis, :] - self._centres[np.newaxis, :, :]
        sq   = np.sum(diff ** 2, axis=2)                     # (N, d-6)
        rbf  = np.exp(-sq / bw)
        return np.hstack([base, rbf]).astype(np.float64)

    # ── Batch EDMD fit ────────────────────────────────────────────────────────

    def fit(self, traj: np.ndarray) -> "KoopmanPredictor":
        """
        Fit Koopman operator K and readout W from trajectory data.

        Parameters
        ----------
        traj : (T, N, 4) single episode OR (E, T, N, 4) stacked episodes.
               All frames from all episodes are stacked into ONE EDMD
               least-squares problem, so K is fitted on the full training
               distribution instead of a single episode.

        Returns
        -------
        self  (for chaining)
        """
        traj = np.asarray(traj, dtype=np.float64)
        if traj.ndim == 3:                      # single episode (T, N, 4)
            episodes = [traj]
        elif traj.ndim == 4:                    # stacked episodes (E, T, N, 4)
            episodes = list(traj)
        else:
            raise ValueError(
                f"fit() expects (T, N, 4) or (E, T, N, 4), got shape {traj.shape}"
            )

        # Lift every frame: collect (frames*N, obs_dim) matrices.
        # Consecutive pairs are taken WITHIN each episode only.
        Z_list  = []   # z_t
        Zp_list = []   # z_{t+1}
        X_list  = []   # x_t  (for readout regression)

        for ep in episodes:
            T = ep.shape[0]
            for t in range(T - 1):
                Zt  = self.lift(ep[t])          # (N, d)
                Ztp = self.lift(ep[t + 1])      # (N, d)
                Z_list.append(Zt)
                Zp_list.append(Ztp)
                X_list.append(ep[t].astype(np.float64))

        Z  = np.vstack(Z_list)   # ((T-1)*N, d)
        Zp = np.vstack(Zp_list)  # ((T-1)*N, d)
        X  = np.vstack(X_list)   # ((T-1)*N, 4)

        # EDMD least squares with Tikhonov regularisation:
        # K = (Z'Z + lambda I)^{-1} Z'Zp   [transposed for row convention]
        d = self.obs_dim
        A = Z.T @ Z + self.reg * np.eye(d)    # (d, d)
        B = Z.T @ Zp                           # (d, d)
        self.K = np.linalg.solve(A, B).T      # (d, d)  K z_t ~= z_{t+1}

        # Clamp spectral radius to <=1 for stability
        self.K = self._clamp_spectral_radius(self.K, max_rho=1.0)

        # Linear readout: W such that W z ~= x
        Aw = Z.T @ Z + self.reg * np.eye(d)
        Bw = Z.T @ X                           # (d, 4)
        self.W = np.linalg.solve(Aw, Bw).T    # (4, d)

        # Initialise RLS covariance matrix
        self._P = (1.0 / self.reg) * np.eye(d)

        self._fitted = True
        return self

    # ── Prediction ────────────────────────────────────────────────────────────

    def predict(self, state: np.ndarray, h: int | None = None) -> np.ndarray:
        """
        Multi-step forecast from a single frame.

        Parameters
        ----------
        state : (N, 4)  current agent states
        h     : number of forecast steps (defaults to cfg forecast_steps)

        Returns
        -------
        forecast : (h, N, 4) float32  predicted future states
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before predict()")
        if h is None:
            h = self.forecast_steps

        state = np.asarray(state, dtype=np.float64)
        N = state.shape[0]

        # NOTE on history: the original readout multiplied the linear W-decode
        # by norm=[100,100,3,3] even though W was fitted on raw states, which
        # scaled positions 100x too large (~5000 m RMSE). We now use a
        # structured toroidal decode (atan2 on the Fourier pair), which is the
        # exact inverse of the lifting map and continuous across the wrap.

        Z = self.lift(state)        # (N, d)  current lifted state
        forecast = np.empty((h, N, 4), dtype=np.float32)

        for step in range(h):
            Z = (self.K @ Z.T).T    # (N, d)  apply K
            x_hat = self._decode(Z)           # (N, 4)  toroidal decode
            forecast[step] = x_hat.astype(np.float32)

        return forecast

    # ── Online RLS update ─────────────────────────────────────────────────────

    def update(self, z: np.ndarray, z_next: np.ndarray) -> None:
        """
        Online RLS update of K given one lifted state pair.

        Parameters
        ----------
        z      : (d,) or (1, d)  current lifted state
        z_next : (d,) or (1, d)  next lifted state
        """
        if not self._fitted:
            raise RuntimeError("Call fit() before update()")

        z      = np.asarray(z,      dtype=np.float64).ravel()[:, None]   # (d, 1)
        z_next = np.asarray(z_next, dtype=np.float64).ravel()[:, None]

        lam = self.forgetting
        P   = self._P                                    # (d, d)

        # Standard RLS update
        Pz     = P @ z                                   # (d, 1)
        denom  = lam + (z.T @ Pz).item()                # scalar
        gain   = Pz / denom                              # (d, 1)
        error  = z_next - self.K @ z                    # (d, 1)
        self.K = self.K + error @ gain.T                # (d, d)
        self._P = (P - gain @ Pz.T) / lam               # (d, d)
        # Re-clamp spectral radius to preserve stability guarantee
        self.K = self._clamp_spectral_radius(self.K, max_rho=1.0)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _clamp_spectral_radius(K: np.ndarray, max_rho: float = 1.0) -> np.ndarray:
        """Scale K so its spectral radius <= max_rho."""
        eigvals = np.linalg.eigvals(K)
        rho = np.max(np.abs(eigvals))
        if rho > max_rho + 1e-9:
            K = K * (max_rho / rho)
        return K

    @property
    def spectral_radius(self) -> float:
        """Current spectral radius of K."""
        if self.K is None:
            return float("nan")
        return float(np.max(np.abs(np.linalg.eigvals(self.K))))