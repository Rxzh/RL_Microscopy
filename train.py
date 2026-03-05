"""Training script for the EBSD adaptive scanning RL agent."""

import argparse
import os

import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from utils import load_latent_map, compute_true_gradients
from ebsd_env import EBSDEnv
from model import EBSDPolicy


def make_env(latent_map: np.ndarray, true_grad_map: np.ndarray, seed: int = 0):
    def _init():
        env = EBSDEnv(latent_map, true_grad_map)
        env.reset(seed=seed)
        return env
    return _init


def build_venv(latent_map: np.ndarray, true_grad_map: np.ndarray, seed: int = 0):
    venv = DummyVecEnv([make_env(latent_map, true_grad_map, seed)])
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return venv


def parse_args():
    parser = argparse.ArgumentParser(description="Train EBSD RL agent")
    parser.add_argument("--tif_dir", type=str, default="TIF",
                        help="Directory containing latent TIF files")
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
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="PyTorch device for policy/rollout buffer (auto=SB3 picks best available)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ #
    # 1. Load data
    # ------------------------------------------------------------------ #
    if args.synthetic:
        print("Using synthetic 50x50 random latent map for testing.")
        rng = np.random.default_rng(args.seed)
        latent_map = rng.standard_normal((50, 50, 23)).astype(np.float32)
        true_grad_map = compute_true_gradients(latent_map)
    else:
        print(f"Loading latent maps from {args.tif_dir} ...")
        latent_map = load_latent_map(args.tif_dir)
        print(f"  Latent map shape: {latent_map.shape}")
        print("Computing true gradient map ...")
        true_grad_map = compute_true_gradients(latent_map)

    # ------------------------------------------------------------------ #
    # 2. Build vectorised environment
    # ------------------------------------------------------------------ #
    venv = build_venv(latent_map, true_grad_map, seed=args.seed)

    # ------------------------------------------------------------------ #
    # 3. Build evaluation env (separate, non-normalised stats)
    # ------------------------------------------------------------------ #
    callbacks = []

    log_dir = os.path.join("logs", args.run_name)
    ckpt_dir = os.path.join("checkpoints", args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    ckpt_cb = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // 1, 1),
        save_path=ckpt_dir,
        name_prefix=args.run_name,
        save_vecnormalize=True,
    )
    callbacks.append(ckpt_cb)

    if args.eval_freq > 0:
        eval_venv = build_venv(latent_map, true_grad_map, seed=args.seed + 1)
        eval_cb = EvalCallback(
            eval_venv,
            best_model_save_path=os.path.join(ckpt_dir, "best"),
            log_path=log_dir,
            eval_freq=max(args.eval_freq // 1, 1),
            n_eval_episodes=3,
            deterministic=True,
        )
        callbacks.append(eval_cb)

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
