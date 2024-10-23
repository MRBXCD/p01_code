import astra
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from model import UNet
from model_mutihead import UNet3Head

# import 3 model individually, train them with 3 individual loss
from models_3_path.model1 import UNet3_1
from models_3_path.model2 import UNet3_2
from models_3_path.model3 import UNet3_3


from projectionDataloader import ProjectionDataset_16, ProjectionDataset_32, ProjectionDataset_16_32, ProjectionDataset_2_4, ProjectionDataset_4_8
import onnx
import onnx
import onnx.utils
import onnx.version_converter
import os
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, params):
        # define model
        self.NUM_GPU = torch.cuda.device_count()
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU', exist_ok=True)
        if self.NUM_GPU > 1:
            print(f'Total GPU number: {self.NUM_GPU}')
            self.model = nn.DataParallel(self.model)
        
        
        self.if_infer = params.if_infer
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epochs = params.epochs
        self.device = params.device
        self.mse_loss = nn.MSELoss()
        self.if_load_weight = params.if_load_weight
        self.check_point = params.check_point
        self.stage = params.stage
        self.if_norm = params.norm
        self.loss_method = params.loss_method
        self.loss_fn = torch.nn.MSELoss()
        self.loss_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        self.if_extraction = params.if_extraction

        # define losses
        self.train_losses = []
        self.val_losses = []

        # define the hyper parameters of model, [proj_mse_angle, amount]
        hyper_parameters = {
            '2-4': [2, 4],
            '4-8': [4, 8],
            '8-16': [8,16],
            '16-32': [16,32],
            '8-32_1Enc_mutiDec': [8, 32],
            '8-32_mutiEnc_mutiDec': [8, 32],
        }
        self.model_parameters = hyper_parameters[self.stage]

        # load train data
        self.data_stage = hyper_parameters[self.stage][1]
        self.train_input = f'/root/autodl-tmp/Projection_predict/data/train/Projection_train_data_{self.data_stage}_angles_padded.npz'
        print(self.train_input)

        # load val data
        self.val_input = f'/root/autodl-tmp/Projection_predict/data/val/Projection_val_data_{self.data_stage}_angles_padded.npz'
        print(self.val_input)

        # initialize dataloader
        if self.stage == '8-32_mutiEnc_mutiDec':
            print('Train data information:')
            train_dataset = ProjectionDataset_32(self.train_input, self.if_norm)
            print('---------------------------------------------')
            print('Val data information:')
            val_dataset = ProjectionDataset_32(self.val_input, self.if_norm)
            print('---------------------------------------------')
        elif self.stage == '8-32_1Enc_mutiDec':
            print('Train data information:')
            train_dataset = ProjectionDataset_32(self.train_input, self.if_norm)
            print('---------------------------------------------')
            print('Val data information:')
            val_dataset = ProjectionDataset_32(self.val_input, self.if_norm)
            print('---------------------------------------------')
        elif self.stage == '8-16':
            print('Train data information:')
            train_dataset = ProjectionDataset_16(self.train_input, self.if_norm)
            print('---------------------------------------------')
            print('Val data information:')
            val_dataset = ProjectionDataset_16(self.val_input, self.if_norm)
            print('---------------------------------------------')
        elif self.stage == '16-32':
            print('Train data information:')
            train_dataset = ProjectionDataset_16_32(self.train_input, self.if_norm)
            print('---------------------------------------------')
            print('Val data information:')
            val_dataset = ProjectionDataset_16_32(self.val_input, self.if_norm)
            print('---------------------------------------------')
        elif self.stage == '2-4':
            print('Train data information:')
            train_dataset = ProjectionDataset_2_4(self.train_input, self.if_norm)
            print('---------------------------------------------')
            print('Val data information:')
            val_dataset = ProjectionDataset_2_4(self.val_input, self.if_norm)
            print('---------------------------------------------')
        elif self.stage == '4-8':
            print('Train data information:')
            train_dataset = ProjectionDataset_4_8(self.train_input, self.if_norm)
            print('---------------------------------------------')
            print('Val data information:')
            val_dataset = ProjectionDataset_4_8(self.val_input, self.if_norm)
            print('---------------------------------------------')


        self.init_mse_train = train_dataset.initial_mse()
        self.train_norm = train_dataset.data_status()
        self.init_mse_val = val_dataset.initial_mse()
        self.val_norm = val_dataset.data_status()


        if not self.data_extraction:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            print('data shuffled')
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
            self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
            print('data not shuffled')

        self.data_show(self.train_loader, 'train')
        self.data_show(self.val_loader, 'val')

        if self.stage == '2-4':
            self.model = UNet()
            self.model_name = 'UNet()'
        elif self.stage == '4-8':
            self.model = UNet()
            self.model_name = 'UNet()'
        elif self.stage == '8-16':
            self.model = UNet()
            self.model_name = 'UNet()'
        elif self.stage == '16-32':
            self.model = UNet()
            self.model_name = 'UNet()'
        self.model.to(self.device)



    def mix_loss(self, input, target, amount=0.5):
        input = input / torch.max(input)
        target = target / torch.max(target)
        ssim_loss = 1 - self.loss_ssim(input, target)
        mse_loss = self.loss_fn(input, target)
        total_loss = ssim_loss * amount + mse_loss * (1-amount)
        return total_loss
    
    def SSIM_loss(self, input, target):
        input = input / torch.max(input)
        target = target / torch.max(target)
        ssim_loss = 1 - self.loss_ssim(input, target)
        return ssim_loss
    
    def extraction_epoch(self):
        self.model.eval()
        losses_train = []
        input_train = []
        result_train = []
        
        losses_val = []
        input_val = []
        result_val = []
        
        os.makedirs('./result/inference/logs', exist_ok=True)
        first_batch_inputs, first_batch_targets = next(iter(self.train_loader))
        plt.imsave('/root/autodl-tmp/Projection_predict/save_for_paper/input_2_4.png', first_batch_inputs[0], cmap='gray')
        plt.imsave('/root/autodl-tmp/Projection_predict/save_for_paper/target_2_4.png', first_batch_targets[0], cmap='gray')

        with torch.no_grad():
            for input, target in tqdm(self.train_loader, desc='Extracting train'):
                
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_image = self.model.forward(input)
                if self.loss_method == 'MSE/SSIM':
                    loss_train = self.mix_loss(prediction_image, target)
                elif self.loss_method == 'SSIM':
                    loss_train = self.SSIM_loss(prediction_image, target)
                elif self.loss_method == 'MSE':
                    loss_train = self.loss_fn(prediction_image, target)
                losses_train.append(loss_train.item())
                prediction_image = prediction_image.squeeze(1).cpu().numpy()
                # plt.imsave('/root/autodl-tmp/Projection_predict/save_for_paper/prediction_4_8.png', prediction_image[0], cmap='gray')
                result_train.append(prediction_image)
                input = input.squeeze(1).cpu().numpy()
                input_train.append(input)
        plt.imsave('/root/autodl-tmp/Projection_predict/save_for_paper/prediction_2_4.png', result_train[0].squeeze(), cmap='gray')
        np.savez(f'./result/inference/{self.stage}_{self.loss_method}_model_output_train.npz', result_train)
        np.savez(f'./result/inference/{self.stage}_{self.loss_method}_model_input_train.npz', input_train)
        np.savetxt(f'./result/inference/logs/loss_output_train_{self.loss_method}_{self.stage}.txt', losses_train)
        # avg_loss_train = sum(losses_train[-len(self.train_loader):]) / len(self.train_loader)
        avg_loss_train = sum(losses_train) / len(losses_train)
        
        with torch.no_grad():
            for input, target in tqdm(self.val_loader, desc='Extracting val'):
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_image = self.model.forward(input)
                if self.loss_method == 'MSE/SSIM':
                    loss_val = self.mix_loss(prediction_image, target)
                elif self.loss_method == 'SSIM':
                    loss_val = self.SSIM_loss(prediction_image, target)
                else:
                    loss_val = self.loss_fn(prediction_image, target)
                losses_val.append(loss_val.item())
                prediction_image = prediction_image.squeeze(1).cpu().numpy()
                result_val.append(prediction_image)
                input = input.squeeze(1).cpu().numpy()
                input_val.append(input)
        np.savez(f'./result/inference/{self.stage}_{self.loss_method}_model_output_val.npz', result_val)
        np.savez(f'./result/inference/{self.stage}_{self.loss_method}_model_input_val.npz', input_val)
        #np.savetxt(f'./result/inference/logs/loss_output_val_{self.loss_method}_{self.stage}.txt', losses_val)
        # avg_loss_val = sum(losses_val[-len(self.val_loader):]) / len(self.val_loader)
        avg_loss_val = sum(losses_val) / len(losses_val)
        return avg_loss_train, avg_loss_val

        