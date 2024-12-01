export CUDA_VISIBLE_DEVICES=0

cd /root/autodl-tmp/p01_code/projection_predict

for dropout_rate in 0.2 0.3 0.4
do

python main.py \
    # edit these two parameters before each experiment
    --exp_id EXP_2024.12.1.8 \
    --suffix dropout\
    --net unet\
    --data_path /root/autodl-tmp/shared_data/projection/raw\
    --device cuda\
    --epochs 200\
    --batch_size 8\
    --lr 1e-4\
    --dropout $dropout_rate\
    --scheduler True\
    --early_stop True \
    --check_point 0\
    --stage 8-16\
    --loss_method Combined_loss\
    --wandb online\
    --exp_name dropout\

echo '====================================================================================================================='
done