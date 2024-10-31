
ckpt=checkpoints/dynamicrafter/i2v_576x1024/model.ckpt
config=configs/train/002_dynamicrafterft_1024/config.yaml
prompt_dir=inputs/i2v/576x1024
savedir=results/dc-i2v-576x1024

python3 scripts/inference.py \
--mode 'i2v' \
--ckpt_path $ckpt \
--config $config \
--prompt_dir $prompt_dir \
--savedir $savedir \
--bs 1 --height 576 --width 1024 \
--fps 10 \
--seed 123


