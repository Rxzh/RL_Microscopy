"""Gymnasium environment for adaptive EBSD beam control via RL."""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from utils import (
    interpolate_latents,
    compute_gradient_map,
    compute_uncertainty_map,
)


class EBSDEnv(gym.Env):
    """Adaptive EBSD scanning environment.

    The agent selects beam positions one at a time, maximizing information
    gain (reduction in gradient prediction error) within a fixed step budget.

    Observation channels (H, W, N+3):
        [0]        — binary mask (1=sampled, 0=not)
        [1:N+1]    — interpolated N-D latent map (nearest-neighbor)
        [N+1]      — predicted gradient magnitude map
        [N+2]      — uncertainty map (normalized distance to nearest sample)

    Action space: Discrete(H * W) — flat grid index.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        latent_map: np.ndarray,
        true_grad_map: np.ndarray,
        max_steps: int = None,
        step_penalty: float = 0.01,
        lambda_scale: float = 1.0,
        seed_grid_size: int = 5,
    ):
        """
        Args:
            latent_map: (H, W, N) ground-truth latent map.
            true_grad_map: (H, W) ground-truth gradient magnitude.
            max_steps: budget (defaults to 20% of H*W).
            step_penalty: β subtracted from reward each step.
            lambda_scale: λ multiplier on information-gain term.
            seed_grid_size: number of points per side in the initial seed grid.
        """
        super().__init__()
        self.latent_map = latent_map.astype(np.float32)
        self.true_grad_map = true_grad_map.astype(np.float32)

        self.H, self.W, self.N = latent_map.shape
        self.n_pixels = self.H * self.W
        self.max_steps = max_steps if max_steps is not None else max(1, int(0.2 * self.n_pixels))
        self.beta = step_penalty
        self.lam = lambda_scale
        self.seed_grid_size = seed_grid_size

        n_obs_channels = self.N + 3  # mask + latents + grad + uncertainty
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(n_obs_channels, self.H, self.W),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(self.n_pixels)

        # State buffers (initialised in reset)
        self._mask: np.ndarray = None
        self._sampled_coords: list = None
        self._sampled_values: list = None
        self._interp_latents: np.ndarray = None
        self._grad_map: np.ndarray = None
        self._uncertainty_map: np.ndarray = None
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self._mask = np.zeros((self.H, self.W), dtype=np.float32)
        self._sampled_coords = []
        self._sampled_values = []
        self.step_count = 0

        # Seed a uniform grid of initial observations
        rows = np.linspace(0, self.H - 1, self.seed_grid_size, dtype=int)
        cols = np.linspace(0, self.W - 1, self.seed_grid_size, dtype=int)
        for r in rows:
            for c in cols:
                self._add_sample(r, c)

        self._rebuild_state()
        return self._get_obs(), {}

    def step(self, action: int):
        row, col = divmod(int(action), self.W)

        # Predicted gradient at this location before sampling
        predicted_grad = float(self._grad_map[row, col])

        # Add new sample
        self._add_sample(row, col)
        self._rebuild_state()

        true_grad = float(self.true_grad_map[row, col])
        reward = float(self.lam * (true_grad - predicted_grad) - self.beta)

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = False

        info = {
            "true_grad": true_grad,
            "predicted_grad": predicted_grad,
            "step_count": self.step_count,
            "row": row,
            "col": col,
        }
        return self._get_obs(), reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        """Boolean mask of valid (unsampled) actions. Shape: (H*W,)."""
        return self._mask.ravel() == 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _add_sample(self, row: int, col: int):
        self._mask[row, col] = 1.0
        self._sampled_coords.append((row, col))
        self._sampled_values.append(self.latent_map[row, col])

    def _rebuild_state(self):
        coords = np.array(self._sampled_coords, dtype=np.float32)   # (K, 2)
        values = np.array(self._sampled_values, dtype=np.float32)   # (K, N)
        self._interp_latents = interpolate_latents(coords, values, (self.H, self.W))
        self._grad_map = compute_gradient_map(self._interp_latents)
        self._uncertainty_map = compute_uncertainty_map(coords, (self.H, self.W))

    def _get_obs(self) -> np.ndarray:
        # Build (H, W, N+3) then transpose to (N+3, H, W) for SB3
        obs_hw = np.concatenate(
            [
                self._mask[:, :, np.newaxis],           # (H, W, 1)
                self._interp_latents,                   # (H, W, N)
                self._grad_map[:, :, np.newaxis],        # (H, W, 1)
                self._uncertainty_map[:, :, np.newaxis], # (H, W, 1)
            ],
            axis=-1,
        )  # (H, W, N+3)
        return obs_hw.transpose(2, 0, 1).astype(np.float32)  # (N+3, H, W)
