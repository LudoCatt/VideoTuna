#!/bin/bash
#SBATCH --job-name=cog_fullft_5b 
#SBATCH --partition=gpu   
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=2
#SBATCH --gres=gpumem:80g 
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=13G
#SBATCH --output=logs/%x_%j_out.txt
#SBATCH --error=logs/%x_%j_err.txt

# ───── Environment ────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate videotuna
export TOKENIZERS_PARALLELISM=false

# Tell Lightning to IGNORE Slurm and spawn workers itself
export SLURM_JOB_NAME=interactive      # any of {interactive,bash,sh} works

# ───── Parameters ────────────────────────────────────────────────
CONFIG=configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml
RESROOT=/cluster/scratch/$USER/results/train
RUNNAME="$(date +%Y%m%d%H%M%S)_cogvideox_i2v_5b_fullft"

# ───── Launch Lightning (one process will spawn two ranks) ───────
python scripts/train.py -t \
        --base   "$CONFIG" \
        --logdir "$RESROOT" \
        --name   "$RUNNAME" \
        --ckpt "/cluster/scratch/lcattaneo/results/train/20250507025951_cogvideox_i2v_5b_fullft/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt" \
        --devices 0,1 \
        lightning.trainer.num_nodes=1
