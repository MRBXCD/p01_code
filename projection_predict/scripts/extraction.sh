export CUDA_VISIBLE_DEVICES=0

python main.py \
    --device cuda\
    --epochs 250\
    --batch_size 1\
    --lr 1e-4\
    --if_load_weight True\
    --check_point 200\
    --stage 8-16\
    --norm False\
    --loss_method Perceptual\
    --if_extraction True\
    --dropout 0.1\
    --wandb disabled

echo '====================================================================================================================='
