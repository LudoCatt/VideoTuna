#!/bin/bash
#SBATCH --job-name=cog_lora               # Job name
#SBATCH --output=cog_lora.txt             # Output file
#SBATCH --error=cog_lora.txt              # Error file
#SBATCH --time=24:00:00                   # Max runtime (10 hours)
#SBATCH --ntasks=1                        # Number of tasks
#SBATCH --cpus-per-task=8                 # (rule of thumb ≈ 4 CPU cores / GPU)
#SBATCH --gpus=2                          # Request 2 GPUs
#SBATCH --gres=gpumem:79g                 # Each GPU must have ≥ 79 GB HBM
#SBATCH --partition=gpu                   # Use the GPU partition
#SBATCH --mem-per-cpu=79G                 # System RAM per CPU core

# Load the Conda environment
export PATH=~/miniconda3/bin:$PATH
source ~/miniconda3/etc/profile.d/conda.sh
conda activate videotuna

export TOKENIZERS_PARALLELISM=false

# dependencies
CONFIG="configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml"   # experiment config

# exp saving directory: ${RESROOT}/${CURRENT_TIME}_${EXPNAME}
RESROOT="/cluster/scratch/lcattaneo/results/train"              # experiment saving directory
EXPNAME="cogvideox_i2v_5b_fullft"   # experiment name
CURRENT_TIME=$(date +%Y%m%d%H%M%S)  # current time

# run
python scripts/train.py \
-t \
--base $CONFIG \
--logdir $RESROOT \
--name "$CURRENT_TIME"_$EXPNAME \
--devices '0,1' \
lightning.trainer.num_nodes=1 \
# --auto_resume
