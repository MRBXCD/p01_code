import argparse
import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda/cpu')
    parser.add_argument('--epochs', type=int, default=250,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--if_load_weight', type=bool, default=True,
                        help='if load weight from pth file.')       
    parser.add_argument('--check_point', type=int, default=360,
                        help='which epoch to start.')
    parser.add_argument('--stage', type=str, default='32-64',
                        help='which stage the model is training')
    parser.add_argument('--norm', type=bool, default=False,
                        help='if set loss as ssim, choose true')      
    parser.add_argument('--loss_method', type=str, default='MSE',
                        help='how to calc loss: MSE or MSE+SSIM')       
    parser.add_argument('--if_extraction', type=bool, default=True, 
                        help='if extract the data.')   

    parser.add_argument('--if_infer', type=bool, default=False, 
                        help='if do inference.')                                                    
    params = parser.parse_args()
    print(params)

    if params.if_extraction:
        model_extract = trainer.Trainer(params)
        model_extract.data_extraction()
    else:
        model_train = trainer.Trainer(params)
        model_train.train()


if __name__ == "__main__":
    main()
