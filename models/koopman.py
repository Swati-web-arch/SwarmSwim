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

        # Random RBF centres: shape (obs_dim, state_dim)
        _seed = seed if seed is not None else int(cfg.get("seed", 42))
        rng = np.random.default_rng(_seed)
        self._centres = rng.standard_normal((self.obs_dim, self.state_dim)).astype(np.float64)
        self._log_bw  = 0.0   # log bandwidth (learnable later; start at 1.0)

        # Koopman operator K: (obs_dim, obs_dim) -- filled by fit()
        self.K: np.ndarray | None = None

        # Linear readout W: obs_dim -> state_dim  -- filled by fit()
        self.W: np.ndarray | None = None

        # RLS state (P matrix for K rows)
        self._P: np.ndarray | None = None

        self._fitted = False

    # ── Lifting ───────────────────────────────────────────────────────────────

    def lift(self, state: np.ndarray) -> np.ndarray:
        """
        Apply the RBF lifting map psi to a single frame.

        Parameters
        ----------
        state : (N, 4) float  agent states

        Returns
        -------
        Z : (N, obs_dim) float64  lifted features, L2-normalised per agent
        """
        state = np.asarray(state, dtype=np.float64)
        # Normalise state to [-1, 1] range (assume world_size=100, speed<=3)
        norm = np.array([100.0, 100.0, 3.0, 3.0], dtype=np.float64)
        x = state / norm                                        # (N, 4)

        # RBF: phi_k(x) = exp(-||x - c_k||^2 / bw)
        bw = np.exp(self._log_bw)
        diff = x[:, np.newaxis, :] - self._centres[np.newaxis, :, :]  # (N, d, 4)
        sq_dist = np.sum(diff ** 2, axis=2)                            # (N, d)
        Z = np.exp(-sq_dist / bw)                                      # (N, d)

        # L2 normalise each row to stabilise K
        norms = np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-9)
        Z = Z / norms
        return Z.astype(np.float64)

    # ── Batch EDMD fit ────────────────────────────────────────────────────────

    def fit(self, traj: np.ndarray) -> "KoopmanPredictor":
        """
        Fit Koopman operator K and readout W from a trajectory.

        Parameters
        ----------
        traj : (T, N, 4)  single episode trajectory

        Returns
        -------
        self  (for chaining)
        """
        traj = np.asarray(traj, dtype=np.float64)
        T, N, _ = traj.shape

        # Lift every frame: collect (T*N, obs_dim) matrices
        Z_list  = []   # z_t
        Zp_list = []   # z_{t+1}
        X_list  = []   # x_t  (for readout regression)

        for t in range(T - 1):
            Zt  = self.lift(traj[t])       # (N, d)
            Ztp = self.lift(traj[t + 1])   # (N, d)
            Z_list.append(Zt)
            Zp_list.append(Ztp)
            X_list.append(traj[t].astype(np.float64))

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

        # Denormalisation scale
        norm = np.array([100.0, 100.0, 3.0, 3.0], dtype=np.float64)

        Z = self.lift(state)        # (N, d)  current lifted state
        forecast = np.empty((h, N, 4), dtype=np.float32)

        for step in range(h):
            Z = (self.K @ Z.T).T    # (N, d)  apply K
            x_hat = (self.W @ Z.T).T * norm   # (N, 4)  decode + denorm
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