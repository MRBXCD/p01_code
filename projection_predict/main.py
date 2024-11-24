import argparse
import trainer
import random
import numpy as np
import torch
import wandb


def main():
    fix_seed = 3407
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_id', type=str, default='EXP',
                        help='index of experiment')
    parser.add_argument('--data_path', type=str, help='path to data folder')
    parser.add_argument('--net', type=str, default='unet',
                        help='unet/unet++/atten_unet')
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda/cpu')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training.')
    
    # Optimization settings
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--scheduler', type=bool, default=False,
                        help='If using scheduler to adjust the learning rate')
    parser.add_argument('--tmax', type=int, default=20,
                        help='scheduler will adjust the learning rate every tmax epoch')
    parser.add_argument('--dropout', type=float, default=0.1, 
                        help='dropout percent') 
    parser.add_argument('--early_stop', type=bool, default=False, 
                        help='if activate early stop function')
    parser.add_argument('--patience', type=int, default=10, 
                        help='early stop patience')
    
    # Model save
    parser.add_argument('--if_load_weight', type=bool, default=False,
                        help='if load weight from pth file.')       
    parser.add_argument('--check_point', type=int, default=0,
                        help='which epoch to start.')
    parser.add_argument('--stage', type=str, default='8-16',
                        help='which stage the model is training')
    parser.add_argument('--norm', type=bool, default=False,
                        help='if set loss as ssim, choose true')      
    parser.add_argument('--loss_method', type=str, default='L1',
                        help='how to calc loss: MSE or MSE+SSIM or Perceptual or L1')       
    parser.add_argument('--if_extraction', type=bool, default=False, 
                        help='if extract the data.')   
    parser.add_argument('--if_infer', type=bool, default=False, 
                        help='if do inference.')     
    parser.add_argument('--wandb', type=str, default='online', 
                        help='if use wandb')                                              
    params = parser.parse_args()
    print(params)

    wandb.init(
        project='projection-domain-dl',
        name=params.loss_method,
        group='loss_method_exp',
        config=params,
        resume='allow',
        job_type='training',
        mode=params.wandb
    )

    if params.if_extraction:
        model_extract = trainer.Trainer(params)
        model_extract.data_extraction()
    else:
        model_train = trainer.Trainer(params)
        model_train.train()
    wandb.finish()


if __name__ == "__main__":
    main()
