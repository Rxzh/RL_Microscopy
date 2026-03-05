"""Shared utilities for EBSD RL environment."""

import os
import re
import numpy as np
import tifffile
from scipy.interpolate import NearestNDInterpolator
from scipy.ndimage import distance_transform_edt


def load_latent_map(tif_dir: str) -> np.ndarray:
    """Load and stack TIF latent channel files into a single array.

    Files must follow the pattern `*_z{index}.tif`. They are stacked
    in ascending z-index order, yielding shape (H, W, N) where N is
    the number of TIF files found.
    """
    pattern = re.compile(r"_z(\d+)\.tif$", re.IGNORECASE)
    entries = []
    for fname in os.listdir(tif_dir):
        m = pattern.search(fname)
        if m:
            entries.append((int(m.group(1)), os.path.join(tif_dir, fname)))
    if not entries:
        raise FileNotFoundError(f"No TIF files matching *_z{{index}}.tif found in {tif_dir}")
    entries.sort(key=lambda x: x[0])
    channels = [tifffile.imread(path).astype(np.float32) for _, path in entries]
    return np.stack(channels, axis=-1)  # (H, W, N)


def compute_true_gradients(latent_map: np.ndarray) -> np.ndarray:
    """Compute per-pixel gradient magnitude across all latent channels.

    For each channel c, compute sqrt(gx²+gy²) then take the L2 norm
    across channels, yielding a scalar map of shape (H, W).
    """
    H, W, N = latent_map.shape
    grad_mag = np.zeros((H, W, N), dtype=np.float32)
    for c in range(N):
        gy, gx = np.gradient(latent_map[:, :, c])
        grad_mag[:, :, c] = np.sqrt(gx ** 2 + gy ** 2)
    return np.linalg.norm(grad_mag, axis=-1)  # (H, W)


def interpolate_latents(
    sampled_coords: np.ndarray,
    sampled_values: np.ndarray,
    grid_shape: tuple,
) -> np.ndarray:
    """Nearest-neighbor interpolation of latent vectors onto a full grid.

    Args:
        sampled_coords: (K, 2) array of (row, col) coordinates.
        sampled_values: (K, N) array of latent vectors.
        grid_shape: (H, W) of the output grid.

    Returns:
        (H, W, N) interpolated latent map.
    """
    H, W = grid_shape
    rows = np.arange(H)
    cols = np.arange(W)
    grid_cols, grid_rows = np.meshgrid(cols, rows)
    query_points = np.column_stack([grid_rows.ravel(), grid_cols.ravel()])

    interpolator = NearestNDInterpolator(sampled_coords, sampled_values)
    interpolated = interpolator(query_points)  # (H*W, N)
    return interpolated.reshape(H, W, -1).astype(np.float32)


def compute_gradient_map(latent_map: np.ndarray) -> np.ndarray:
    """Gradient magnitude map from any (H, W, N) latent map. Returns (H, W)."""
    return compute_true_gradients(latent_map)


def compute_uncertainty_map(sampled_coords: np.ndarray, grid_shape: tuple) -> np.ndarray:
    """Euclidean distance from each pixel to its nearest sampled point.

    Args:
        sampled_coords: (K, 2) array of (row, col) sampled positions.
        grid_shape: (H, W).

    Returns:
        (H, W) distance map, normalized to [0, 1] by dividing by max distance.
    """
    H, W = grid_shape
    mask = np.ones((H, W), dtype=bool)  # True = unsampled
    for r, c in sampled_coords:
        mask[int(r), int(c)] = False
    dist = distance_transform_edt(mask).astype(np.float32)
    max_dist = dist.max()
    if max_dist > 0:
        dist /= max_dist
    return dist
