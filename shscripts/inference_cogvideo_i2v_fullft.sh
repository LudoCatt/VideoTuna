#!/bin/bash
#SBATCH --job-name=cog_inf            # Job name
#SBATCH --output=cog_inf.txt   # Output file
#SBATCH --error=cog_inf.txt     # Error file
#SBATCH --time=4:00:00                    # Max runtime (10 hours)
#SBATCH --ntasks=1                         # Number of tasks
#SBATCH --cpus-per-task=4                  # Number of CPU cores per task
#SBATCH --gpus=1                           # Request 1 GPU
#SBATCH --gres=gpumem:79g                  # Request 79 GB GPU memory
#SBATCH --partition=gpu                    # Use the GPU partition
#SBATCH --mem-per-cpu=79G                  # Memory per CPU core

# Load the Conda environment
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate videotuna

# Run the Python script
cd ~/VideoTuna

config=configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml
ckpt=/cluster/scratch/lcattaneo/results/train/20250508030007_cogvideox_i2v_5b_fullft/checkpoints/last.ckpt/checkpoint/mp_rank_00_model_states.pt
prompt_dir=inputs/i2v/576x1024

current_time=$(date +%Y%m%d%H%M%S)
savedir="results/inference/i2v/cogvideox-i2v-fullft-400-epochs-weird"

python3 scripts/inference_cogvideo.py \
    --config $config \
    --ckpt_path $ckpt \
    --prompt_dir $prompt_dir \
    --savedir $savedir \
    --bs 1 --height 480 --width 720 \
    --fps 16 \
    --seed 6666 \
    --mode i2v \
    --denoiser_precision bf16
