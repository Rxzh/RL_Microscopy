# RL_Microscopy

Reinforcement learning agent for adaptive EBSD (Electron Backscatter Diffraction) beam control. The agent learns where to scan on a sample to maximize information gain (gradient discovery) within a fixed step budget, using PPO (Stable-Baselines3 + PyTorch).

## Overview

Instead of scanning a sample uniformly, the agent sequentially selects beam positions that are most likely to reveal new structural information. At each step it observes the current interpolated latent map, a predicted gradient map, and an uncertainty map, then picks the next pixel to sample.

**Reward:** `λ * (true_grad − predicted_grad) − β`
- `λ` (`lambda_scale=1.0`): information-gain multiplier
- `β` (`step_penalty=0.01`): per-step cost

## File Structure

```
train.py        # Training entry point (argparse CLI)
ebsd_env.py     # Gymnasium environment: EBSDEnv
model.py        # Custom SB3 policy: EBSDPolicy + EBSDFeaturesExtractor
utils.py        # Data loading, interpolation, gradient/uncertainty maps
requirements.txt
TIF/            # Latent channel TIF files (NN256_classic_map1_z{index}.tif)
logs/           # TensorBoard logs
checkpoints/    # Saved model checkpoints
figures/        # Episode visualizations (saved every N episodes)
```

## Architecture

### Observation Space: `(N+3, H, W)` channels-first
| Channel | Description |
|---------|-------------|
| `[0]` | Binary mask (1 = sampled, 0 = not) |
| `[1:N+1]` | Nearest-neighbor interpolated latent map (N channels) |
| `[N+1]` | Predicted gradient magnitude map |
| `[N+2]` | Uncertainty map (normalized distance to nearest sample) |

### Action Space
`Discrete(H * W)` — flat grid index within the current tile. Already-sampled pixels are masked out in the policy.

### Policy (`model.py`)
- **Backbone** (`EBSDFeaturesExtractor`): Conv2d stack → spatial features `(B, 32, H, W)`
- **Actor**: `Conv2d(32→1)` over spatial features → H\*W logits, masked at sampled pixels
- **Critic**: `AdaptiveAvgPool + Linear(32→1)` on spatial features

### Environment (`ebsd_env.py`)
- Each episode: random tile `(tile_h, tile_w)` cropped from a randomly selected dataset
- Starts with a `seed_grid_size × seed_grid_size` uniform grid of initial observations
- `max_steps` defaults to 20% of tile pixels
- Incremental nearest-neighbor interpolation updated after each sample

## Installation

```bash
pip install -r requirements.txt
```

**Requirements:** `torch>=2.0`, `gymnasium>=0.29`, `stable-baselines3>=2.3`, `numpy`, `scipy`, `scikit-image`, `tifffile`, `matplotlib`, `tensorboard`

## Usage

### Training on real data

```bash
python train.py --tif_dirs TIF --total_timesteps 1000000 --run_name my_run
```

Multiple dataset directories can be provided:

```bash
python train.py --tif_dirs TIF TIF2 --total_timesteps 1000000 --run_name multi_run
```

### Synthetic smoke test

```bash
python train.py --synthetic --total_timesteps 10000
```

### Key CLI arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--tif_dirs` | `TIF` | One or more dirs with latent TIF files |
| `--tile_h` | `128` | Tile height per episode |
| `--tile_w` | `128` | Tile width per episode |
| `--total_timesteps` | `1000000` | Total training steps |
| `--n_steps` | `512` | PPO rollout steps per update |
| `--batch_size` | `64` | PPO minibatch size |
| `--seed` | `0` | Random seed |
| `--run_name` | `ebsd_ppo` | Name for logs/checkpoints |
| `--checkpoint_freq` | `50000` | Save checkpoint every N steps |
| `--eval_freq` | `25000` | Deterministic eval every N steps (0=off) |
| `--figure_every` | `50` | Save episode figure every N episodes (0=off) |
| `--figure_dir` | `figures` | Directory for episode figures |
| `--device` | `auto` | PyTorch device (`auto`, `cpu`, `cuda`, `mps`) |
| `--synthetic` | — | Use random synthetic data for testing |

## TIF File Format

Files should be named `NN256_classic_map1_z{index}.tif` and placed in the TIF directory. They are loaded and stacked in ascending z-index order to form the `(H, W, N)` latent map.

## Monitoring

TensorBoard logs are written to `logs/{run_name}/`:

```bash
tensorboard --logdir logs
```

Logged metrics include per-episode reward, episode length, and tile position.

## Checkpoints

Checkpoints (model + VecNormalize stats) are saved to `checkpoints/{run_name}/` every `--checkpoint_freq` steps. The best model (by eval reward) is saved to `checkpoints/{run_name}/best/`.

## Quick Verification

```bash
# Environment only
python -c "
from ebsd_env import EBSDEnv
import numpy as np
env = EBSDEnv([(np.random.randn(50,50,23).astype('f'), np.random.randn(50,50).astype('f'))])
obs, _ = env.reset()
obs2, r, _, trunc, _ = env.step(env.action_space.sample())
print(obs2.shape, r, trunc)
"

# Full training
python train.py --synthetic --total_timesteps 10000
```
