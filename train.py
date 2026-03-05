"""Training script for the EBSD adaptive scanning RL agent."""

import argparse
import os

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)

from utils import load_multiple_datasets, compute_true_gradients
from ebsd_env import EBSDEnv
from model import EBSDPolicy


def make_env(dataset_list, tile_h, tile_w, seed=0):
    def _init():
        env = EBSDEnv(dataset_list, tile_h=tile_h, tile_w=tile_w)
        env.reset(seed=seed)
        return env
    return _init


def build_venv(dataset_list, tile_h, tile_w, seed=0):
    venv = DummyVecEnv([make_env(dataset_list, tile_h, tile_w, seed)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


class EpisodeLogCallback(BaseCallback):
    def __init__(self, figure_every=50, figure_dir="figures", verbose=1):
        super().__init__(verbose)
        self.figure_every = figure_every
        self.figure_dir = figure_dir
        self._episode_count = 0
        self._current_ep_reward = 0.0

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]   # shape (n_envs,)
        dones   = self.locals["dones"]     # shape (n_envs,)
        infos   = self.locals["infos"]

        for i in range(len(dones)):
            self._current_ep_reward += float(rewards[i])
            if dones[i]:
                info = infos[i]
                ep_len    = info.get("step_count", -1)
                tile_orig = info.get("tile_origin", (-1, -1))
                ds_idx    = info.get("dataset_idx", 0)
                ep_reward = self._current_ep_reward

                print(
                    f"[Episode {self._episode_count + 1:>5d}] "
                    f"reward={ep_reward:+.3f}  steps={ep_len}  "
                    f"tile=({tile_orig[0]},{tile_orig[1]})  dataset={ds_idx}"
                )

                self.logger.record("episode/reward", ep_reward)
                self.logger.record("episode/length", ep_len)
                self.logger.record("episode/tile_row", tile_orig[0])
                self.logger.record("episode/tile_col", tile_orig[1])
                self.logger.dump(self.num_timesteps)

                self._episode_count += 1
                self._current_ep_reward = 0.0

                if (self.figure_every > 0
                        and self._episode_count % self.figure_every == 0):
                    self._save_figure(i)
        return True

    def _save_figure(self, env_idx: int):
        import matplotlib.pyplot as plt

        try:
            inner_env = self.training_env.venv.envs[env_idx]
        except AttributeError:
            inner_env = self.training_env.envs[env_idx]
        snap = getattr(inner_env, "_last_episode_snapshot", None)
        if snap is None:
            return

        os.makedirs(self.figure_dir, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].imshow(snap["interp_latent_ch0"], cmap="viridis", origin="upper")
        axes[0].set_title("Final interp latent (ch 0)")
        axes[0].axis("off")

        axes[1].imshow(snap["interp_latent_ch0"], cmap="viridis", origin="upper", alpha=0.7)
        ys, xs = np.where(snap["mask"] > 0)
        axes[1].scatter(xs, ys, s=2, c="red", label="sampled")
        axes[1].set_title(
            f"Sampling mask  tile=({snap['tile_origin'][0]},{snap['tile_origin'][1]})"
            f"  steps={snap['step_count']}"
        )
        axes[1].axis("off")

        fig.tight_layout()
        fname = os.path.join(
            self.figure_dir,
            f"ep{self._episode_count:06d}_tile{snap['tile_origin'][0]}x{snap['tile_origin'][1]}.png",
        )
        fig.savefig(fname, dpi=100)
        plt.close(fig)
        if self.verbose:
            print(f"  [Figure saved → {fname}]")


def parse_args():
    parser = argparse.ArgumentParser(description="Train EBSD RL agent")
    parser.add_argument("--tif_dirs", nargs="+", default=["TIF"],
                        help="One or more directories containing latent TIF files")
    parser.add_argument("--tile_h", type=int, default=128,
                        help="Tile height for each episode")
    parser.add_argument("--tile_w", type=int, default=128,
                        help="Tile width for each episode")
    parser.add_argument("--total_timesteps", type=int, default=1_000_000)
    parser.add_argument("--n_steps", type=int, default=512,
                        help="PPO rollout steps per update")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="ebsd_ppo",
                        help="Used for checkpoint/log naming")
    parser.add_argument("--checkpoint_freq", type=int, default=50_000,
                        help="Save a checkpoint every N env steps")
    parser.add_argument("--eval_freq", type=int, default=25_000,
                        help="Run deterministic eval every N env steps (0=disabled)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use a synthetic 50x50 random latent map (for testing)")
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cpu", "cuda", "mps"],
                        help="PyTorch device")
    parser.add_argument("--figure_every", type=int, default=50,
                        help="Save a figure every N episodes (0=disabled)")
    parser.add_argument("--figure_dir", type=str, default="figures",
                        help="Directory for episode figures")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if args.synthetic:
        print("Using synthetic 50x50 random latent map for testing.")
        rng = np.random.default_rng(args.seed)
        lm = rng.standard_normal((50, 50, 23)).astype(np.float32)
        gm = compute_true_gradients(lm)
        dataset_list = [(lm, gm)]
    else:
        print(f"Loading latent maps from {args.tif_dirs} ...")
        dataset_list = load_multiple_datasets(args.tif_dirs)
        for i, (lm, _) in enumerate(dataset_list):
            print(f"  Dataset {i}: shape={lm.shape}")

    # ------------------------------------------------------------------ #
    # 2. Build vectorised environment
    # ------------------------------------------------------------------ #
    venv = build_venv(dataset_list, args.tile_h, args.tile_w, seed=args.seed)

    # ------------------------------------------------------------------ #
    # 3. Callbacks
    # ------------------------------------------------------------------ #
    log_dir = os.path.join("logs", args.run_name)
    ckpt_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    callbacks = []

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // 1, 1),
        save_path=ckpt_dir,
        name_prefix=args.run_name,
        save_vecnormalize=True,
    )
    callbacks.append(ckpt_cb)

    if args.eval_freq > 0:
        eval_venv = build_venv(dataset_list, args.tile_h, args.tile_w, seed=args.seed + 1)
        eval_cb = EvalCallback(
            eval_venv,
            best_model_save_path=os.path.join(ckpt_dir, "best"),
            log_path=log_dir,
            eval_freq=max(args.eval_freq // 1, 1),
            n_eval_episodes=3,
            deterministic=True,
        )
        callbacks.append(eval_cb)

    callbacks.append(
        EpisodeLogCallback(
            figure_every=args.figure_every,
            figure_dir=args.figure_dir,
            verbose=1,
        )
    )

    # ------------------------------------------------------------------ #
    # 4. Configure and create PPO model
    # ------------------------------------------------------------------ #
    model = PPO(
        policy=EBSDPolicy,
        env=venv,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        tensorboard_log=log_dir,
        verbose=1,
        seed=args.seed,
        device=args.device,
    )
    print(f"Training on device: {model.device}")

    # ------------------------------------------------------------------ #
    # 5. Train
    # ------------------------------------------------------------------ #
    print(f"Starting training for {args.total_timesteps} timesteps ...")
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=callbacks,
        tb_log_name=args.run_name,
        reset_num_timesteps=True,
    )

    # ------------------------------------------------------------------ #
    # 6. Save final model
    # ------------------------------------------------------------------ #
    final_path = os.path.join(ckpt_dir, f"{args.run_name}_final")
    model.save(final_path)
    venv.save(os.path.join(ckpt_dir, f"{args.run_name}_vecnormalize.pkl"))
    print(f"Training complete. Model saved to {final_path}.")


if __name__ == "__main__":
    main()
