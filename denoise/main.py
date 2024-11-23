import argparse
import trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_input', type=str, default='/root/autodl-tmp/denoise/data/Train/Train_Recons_SIRT_8_angles.npz',
                        help='Path to training input data.')
    parser.add_argument('--train_target', type=str, default='/root/autodl-tmp/denoise/data/Train/Train_LIDC_128.npz',
                        help='Path to training target data.')

    parser.add_argument('--val_input', type=str, default='/root/autodl-tmp/denoise/data/Val/Val_Recons_SIRT_8_angles.npz',
                        help='Path to validation input data.')
    parser.add_argument('--val_target', type=str, default='/root/autodl-tmp/denoise/data/Val/Val_LIDC_128.npz',
                        help='Path to validation target data.')

    parser.add_argument('--device', type=str, default='cuda',
                        help='cuda/cpu')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs.')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for training.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate for the optimizer.')
    parser.add_argument('--if_load_weight', type=bool, default=False,
                        help='if load weight from pth file.')       
    parser.add_argument('--check_point', type=int, default=80,
                        help='which epoch to start.')
    parser.add_argument('--stage', type=str, default='8-16',
                        help='which stage the model is training')            
    parser.add_argument('--if_extraction', type=bool, default=True, 
                        help='if extract the data.')                                                      
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
