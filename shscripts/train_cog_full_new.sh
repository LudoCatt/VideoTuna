#!/bin/bash
#SBATCH --job-name=cog_fullft_5b                  # appears in squeue / sacct
#SBATCH --time=24:00:00                           # wall-clock limit
#SBATCH --ntasks=1                                # one task (rank-0 starter)
#SBATCH --cpus-per-task=16                        # 8 logical cores per GPU
#SBATCH --gpus=2                                  # 2 CUDA devices
#SBATCH --gres=gpumem:79g                         # A100-80 GB flavour
#SBATCH --mem-per-cpu=13G                         # 16 × 13 G ≃ 208 G host RAM
#SBATCH --partition=gpu
#SBATCH --output=logs/%x_%j.out                   # stdout
#SBATCH --error=logs/%x_%j.err                    # stderr

####################  Environment  ###########################################

# make 'conda' command available
export PATH="$HOME/miniconda3/bin:$PATH"
source "$HOME/miniconda3/etc/profile.d/conda.sh"

# activate your env (make sure it already contains torch>=2.3)
conda activate videotuna

# silence tokenizer fork warning
export TOKENIZERS_PARALLELISM=false

####################  Experiment parameters  #################################

CONFIG=configs/004_cogvideox/cogvideo5b-i2v-fullft.yaml
RESROOT=/cluster/scratch/$USER/results/train
EXPNAME=cogvideox_i2v_5b_fullft
RUNNAME="$(date +%Y%m%d%H%M%S)_${EXPNAME}"

####################  Run  ####################################################

srun python scripts/train.py -t \
      --base "$CONFIG" \
      --logdir "$RESROOT" \
      --name  "$RUNNAME" \
      --devices 0,1 \
      lightning.trainer.num_nodes=1
