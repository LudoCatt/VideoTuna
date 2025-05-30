#!/bin/bash
#SBATCH --job-name=cog_inf            # Job name
#SBATCH --output=cog_inf.txt   # Output file
#SBATCH --error=cog_inf.txt     # Error file
#SBATCH --time=10:00:00                    # Max runtime (10 hours)
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
poetry run inference-cogvideo-i2v-lora