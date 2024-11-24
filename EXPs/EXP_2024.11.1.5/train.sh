export CUDA_VISIBLE_DEVICES=0

cd /root/autodl-tmp/p01_code/projection_predict

for loss_method in Combined_loss
do

python main.py \
    --exp_id EXP_2024.11.1.5 \
    --data_path /root/autodl-tmp/shared_data/projection/raw/\
    --device cuda\
    --epochs 200\
    --batch_size 16\
    --lr 2e-4\
    --scheduler True\
    --early_stop True \
    --check_point 0\
    --stage 8-16\
    --loss_method $loss_method\
    --dropout 0.1\
    --wandb online

echo '====================================================================================================================='
done