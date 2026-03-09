#!/bin/bash
#SBATCH --job-name=ebsd-rl-train
#SBATCH -C a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=24
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu_a100-t3
#SBATCH --time=20:00:00
#SBATCH --account=dxe@a100
#SBATCH --output=logs/%j_ebsd_rl_train.out
#SBATCH --error=logs/%j_ebsd_rl_train.err

# Jean-Zay A100 partition (gpu_p6): 96 CPU, 468 GB RAM, 4x A100
# cpus-per-task=24 reserves 1/4 of node memory per GPU (117 GB)
# ntasks-per-node=1: single-process RL training (not distributed)

module purge
module load arch/a100
module load anaconda-py3/2024.06
conda activate /lustre/fswork/projects/rech/dxe/ueq71sr/pytorch-gpu-1.11+py3.9.12-ryzh


cd $WORK/RL_Microscopy
export PYTHONPATH=$PWD:$PYTHONPATH

srun python train.py \
    --device cuda

echo "Training completed"
