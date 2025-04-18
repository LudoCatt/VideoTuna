#!/bin/bash
#SBATCH --job-name=cog_lora            # Job name
#SBATCH --output=cog_lora.txt   # Output file
#SBATCH --error=cog_lora.txt     # Error file
#SBATCH --time=24:00:00                    # Max runtime (10 hours)
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

export TOKENIZERS_PARALLELISM=false

# dependencies
CONFIG="configs/004_cogvideox/cogvideo5b-i2v.yaml"   # experiment config

# exp saving directory: ${RESROOT}/${CURRENT_TIME}_${EXPNAME}
RESROOT="/cluster/scratch/lcattaneo/results/train"             # experiment saving directory
EXPNAME="cogvideox_i2v_5b"          # experiment name
CURRENT_TIME=$(date +%Y%m%d%H%M%S)  # current time
DATAPATH="data/apply_lipstick/metadata.csv"

# run
python scripts/train.py \
-t \
--base $CONFIG \
--logdir $RESROOT \
--name "$CURRENT_TIME"_$EXPNAME \
--devices '0,' \
lightning.trainer.num_nodes=1 \
data.params.train.params.csv_path=$DATAPATH \
data.params.validation.params.csv_path=$DATAPATH \
--ckpt  /cluster/scratch/lcattaneo/results/train/20250415152758_cogvideox_i2v_5b/checkpoints/last.ckpt
