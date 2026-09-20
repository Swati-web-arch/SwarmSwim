"""
sim/formations.py
-----------------
Initial formation generators for the swarm.

Supported kinds: "random", "circle", "grid", "two_clusters"
Returns (positions, velocities) both shape (N, 2).
"""
import numpy as np


def make_formation(
    kind: str,
    N: int,
    world_size: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate initial positions and velocities for N agents.

    Parameters
    ----------
    kind       : one of "random", "circle", "grid", "two_clusters"
    N          : number of agents
    world_size : arena side length (agents live in [0, world_size]^2)
    rng        : seeded numpy Generator for reproducibility

    Returns
    -------
    pos : (N, 2) float64 - initial positions
    vel : (N, 2) float64 - initial velocities (small random perturbation)
    """
    half = world_size / 2.0

    if kind == "random":
        pos = rng.uniform(0.0, world_size, size=(N, 2))

    elif kind == "circle":
        radius = world_size * 0.35
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        pos = np.stack([
            half + radius * np.cos(angles),
            half + radius * np.sin(angles),
        ], axis=1)

    elif kind == "grid":
        side = int(np.ceil(np.sqrt(N)))
        xs = np.linspace(world_size * 0.1, world_size * 0.9, side)
        ys = np.linspace(world_size * 0.1, world_size * 0.9, side)
        gx, gy = np.meshgrid(xs, ys)
        grid_pts = np.stack([gx.ravel(), gy.ravel()], axis=1)
        pos = grid_pts[:N].copy()

    elif kind == "two_clusters":
        n1 = N // 2
        n2 = N - n1
        c1 = rng.normal(loc=[world_size * 0.25, half], scale=world_size * 0.05, size=(n1, 2))
        c2 = rng.normal(loc=[world_size * 0.75, half], scale=world_size * 0.05, size=(n2, 2))
        pos = np.vstack([c1, c2])

    else:
        raise ValueError(f"Unknown formation kind: {kind!r}. "
                         f"Choose from 'random', 'circle', 'grid', 'two_clusters'.")

    # Clip to arena bounds
    pos = np.clip(pos, 0.0, world_size)

    # Small random initial velocities
    vel = rng.uniform(-0.5, 0.5, size=(N, 2))

    return pos.astype(np.float64), vel.astype(np.float64)