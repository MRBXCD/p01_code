export CUDA_VISIBLE_DEVICES=0

python main.py \
    --device cuda\
    --epochs 200\
    --batch_size 8\
    --lr 1e-4\
    --check_point 0\
    --stage 8-16\
    --loss_method L1\
    --dropout 0.1\
    --wandb online

echo '====================================================================================================================='
