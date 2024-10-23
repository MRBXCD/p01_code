import astra
import skimage.metrics as sk
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import data_compose
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from model import UNet
from model_mutihead import UNet3Head

# Import 3 models individually, train them with 3 individual loss
from models_3_path.model1 import UNet3_1
from models_3_path.model2 import UNet3_2
from models_3_path.model3 import UNet3_3

from projectionDataloader import ProjectionDataset, ProjectionDataset_inference, ProjectionDataset_FineTune
import onnx
import onnx.utils
import onnx.version_converter
import os
import torchvision.transforms.functional as F
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict


class Trainer:
    def __init__(self, params):
        # Define model
        self.NUM_GPU = torch.cuda.device_count()
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU', exist_ok=True)

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
        # self.loss_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        self.if_extraction = params.if_extraction

        # Define losses
        self.train_losses = []
        self.val_losses = []

        # Define the hyperparameters of the model
        hyper_parameters = {
            '2-4': [2, 4],
            '4-8': [4, 8],
            '8-16': [8, 16],
            '16-32': [16, 32],
            '32-64': [32, 64],
            '64-128': [64, 128],
        }
        self.model_parameters = hyper_parameters[self.stage]

        # Load train data
        self.data_stage = hyper_parameters[self.stage][1]
        self.train_input = f'./data/train/Projection_train_data_{self.data_stage}_angles_padded.npz'
        print(self.train_input)

        # Load val data
        self.val_input = f'./data/val/Projection_val_data_{self.data_stage}_angles_padded.npz'
        print(self.val_input)

        print('Train data information:')
        train_dataset = ProjectionDataset(self.train_input, if_norm=False)
        print('---------------------------------------------')
        val_dataset = ProjectionDataset(self.val_input, if_norm=False)
        print('---------------------------------------------')

        # Initialize dataloader
        self.init_mse_train = train_dataset.initial_mse()
        self.train_norm = train_dataset.data_status()
        self.init_mse_val = val_dataset.initial_mse()
        self.val_norm = val_dataset.data_status()

        if not self.if_extraction:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            print('Data shuffled')
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
            self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
            print('Data not shuffled')

        self.data_show(self.train_loader, 'train')
        self.data_show(self.val_loader, 'val')

        if self.if_infer:
            self.model1 = UNet()
            self.model2 = UNet()
            self.model3 = UNet()
            self.model4 = UNet()
        else:
            self.model = UNet()
            self.model_name = 'UNet()'
            self.model.to(self.device)

            # If more than one GPU is available, use DataParallel
            if self.NUM_GPU > 1:
                print(f"Using {self.NUM_GPU} GPUs for training")
                self.model = nn.DataParallel(self.model)

        # Initialize tensorboard
        self.writer = SummaryWriter(f'logs/{self.stage}_training')

    def load_weights(self, state_dict):
        """
        Load weights into the model, adjusting the state dictionary if the weights were trained on a multi-GPU setup.
        """
        # Check if 'module.' is in keys of state_dict
        is_parallel = any(k.startswith('module.') for k in state_dict.keys())

        if not isinstance(self.model, nn.DataParallel) and is_parallel:
            # If model is not DataParallel but state_dict has 'module.', remove 'module.'
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # remove 'module.' prefix
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        elif isinstance(self.model, nn.DataParallel) and not is_parallel:
            # If model is DataParallel but state_dict doesn't have 'module.', add 'module.'
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = 'module.' + k  # add 'module.' prefix
                new_state_dict[name] = v
            self.model.load_state_dict(new_state_dict)
        else:
            # No need to adjust the keys
            self.model.load_state_dict(state_dict)

    def evaluation_metrics(self, predict, target):
        predict = predict.squeeze().cpu().numpy()
        target = target.squeeze().cpu().numpy()
        normalized_rmse = sk.normalized_root_mse(target, predict, normalization='min-max')
        psnr = sk.peak_signal_noise_ratio(target, predict, data_range=np.max(target) - np.min(target))
        ssim = sk.structural_similarity(target, predict, data_range=np.max(target) - np.min(target))
        return normalized_rmse, psnr, ssim

    def metrics_process(self, train_metrics, val_metrics):
        train_metrics = np.array(train_metrics)
        val_metrics = np.array(val_metrics)
        avg_nrmse_calc = lambda a: sum(a[:, 0]) / len(a[:, 0])
        avg_nrmse_train = avg_nrmse_calc(train_metrics)
        avg_nrmse_val = avg_nrmse_calc(val_metrics)

        avg_psnr_calc = lambda a: sum(a[:, 1]) / len(a[:, 1])
        avg_psnr_train = avg_psnr_calc(train_metrics)
        avg_psnr_val = avg_psnr_calc(val_metrics)

        avg_ssim_calc = lambda a: sum(a[:, 2]) / len(a[:, 2])
        avg_ssim_train = avg_ssim_calc(train_metrics)
        avg_ssim_val = avg_ssim_calc(val_metrics)
        with open(f'./save_for_paper/{self.stage}_metrics.txt', 'w') as file:
            file.write("train/val\n")
            file.write(f"avg_nrmse:{avg_nrmse_train, avg_nrmse_val}\n")
            file.write(f"avg_psnr:{avg_psnr_train, avg_psnr_val}\n")
            file.write(f"avg_ssim:{avg_ssim_train, avg_ssim_val}\n")

    def mix_loss(self, input, target, amount=0.5):
        input = input / torch.max(input)
        target = target / torch.max(target)
        ssim_loss = 1 - self.loss_ssim(input, target)
        mse_loss = self.loss_fn(input, target)
        total_loss = ssim_loss * amount + mse_loss * (1 - amount)
        return total_loss

    def SSIM_loss(self, input, target):
        input = input / torch.max(input)
        target = target / torch.max(target)
        ssim_loss = 1 - self.loss_ssim(input, target)
        return ssim_loss

    def data_show(self, dataloader, tit):
        first_batch_inputs, first_batch_targets = next(iter(dataloader))
        # Create a figure to display the images
        fig, axs = plt.subplots(2, 1)

        # Adjust spacing
        plt.subplots_adjust(wspace=0.1, hspace=0.2)

        # Move data to CPU and convert to numpy arrays
        input = first_batch_inputs[0, :, :].cpu().numpy()
        target = first_batch_targets[0, :, :].cpu().numpy()

        # Display input image
        axs[0].imshow(input, cmap='gray')
        axs[0].set_title(f'Input #{1}')
        axs[0].axis('off')

        # Display target image
        axs[1].imshow(target, cmap='gray')
        axs[1].set_title(f'Target #{1}')
        axs[1].axis('off')

        # Set main title
        fig.suptitle(tit)

        # Adjust layout
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the figure
        plt.savefig(f'{tit}.png', dpi=150)

    def train_epoch(self, epoch):
        self.model.train()
        total_losses = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for input, target in self.train_loader:
            input = input.unsqueeze(1).to(self.device)
            target = target.unsqueeze(1).to(self.device)
            predict = self.model(input)
            loss = self.loss_fn(predict, target)
            self.writer.add_scalar('training total loss', loss, epoch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_losses.append(loss.item())
        avg_total_loss = sum(total_losses[-len(self.train_loader):]) / len(self.train_loader)
        return avg_total_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_losses = []

        with torch.no_grad():
            for input, target in self.val_loader:
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)

                predict = self.model(input)

                loss = self.loss_fn(predict, target)
                self.writer.add_scalar('val total loss', loss, epoch)

                total_losses.append(loss.item())
        avg_total_loss = sum(total_losses[-len(self.val_loader):]) / len(self.val_loader)
        return avg_total_loss

    def extraction_epoch(self):
        self.model.eval()
        losses_train = []
        input_train = []
        result_train = []
        metrics_train = []

        losses_val = []
        input_val = []
        result_val = []
        metrics_val = []

        os.makedirs(f'./result/extraction/{self.stage}/logs', exist_ok=True)
        first_batch_inputs, first_batch_targets = next(iter(self.train_loader))
        plt.imsave('./save_for_paper/input_2_4.png', first_batch_inputs[0], cmap='gray')
        plt.imsave('./save_for_paper/target_2_4.png', first_batch_targets[0], cmap='gray')

        with torch.no_grad():
            for input, target in tqdm(self.train_loader, desc='Extracting train'):
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)
                target1 = target.cpu().numpy()
                plt.imsave(f'./save_for_paper/{self.stage}.png', target1[0, 0, :, :], cmap='gray')

                prediction_image = self.model(input)
                loss_train = self.loss_fn(prediction_image, target)

                metric_train = self.evaluation_metrics(prediction_image, target)
                metrics_train.append(metric_train)
                losses_train.append(loss_train.item())
                prediction_image = prediction_image.squeeze(1).cpu().numpy()
                result_train.append(prediction_image)
                input = input.squeeze(1).cpu().numpy()
                input_train.append(input)
        plt.imsave('./save_for_paper/prediction_2_4.png', result_train[0].squeeze(), cmap='gray')
        np.savez(f'./result/extraction/{self.stage}/{self.loss_method}_model_output_train.npz', result_train)
        np.savez(f'./result/extraction/{self.stage}/{self.loss_method}_model_input_train.npz', input_train)
        np.savetxt(f'./result/extraction/{self.stage}/logs/loss_output_train_{self.loss_method}_{self.stage}.txt', losses_train)
        avg_loss_train = sum(losses_train) / len(losses_train)

        with torch.no_grad():
            for input, target in tqdm(self.val_loader, desc='Extracting val'):
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)

                prediction_image = self.model(input)

                loss_val = self.loss_fn(prediction_image, target)

                metric_val = self.evaluation_metrics(prediction_image, target)
                metrics_val.append(metric_val)
                losses_val.append(loss_val.item())
                prediction_image = prediction_image.squeeze(1).cpu().numpy()
                result_val.append(prediction_image)
                input = input.squeeze(1).cpu().numpy()
                input_val.append(input)
        plt.imsave('./save_for_paper/prediction_2_4.png', result_val[0].squeeze(), cmap='gray')
        np.savez(f'./result/extraction/{self.stage}/{self.loss_method}_model_output_val.npz', result_val)
        np.savez(f'./result/extraction/{self.stage}/{self.loss_method}_model_input_val.npz', input_val)
        np.savetxt(f'./result/extraction/{self.stage}/logs/loss_output_val_{self.loss_method}_{self.stage}.txt', losses_val)
        avg_loss_val = sum(losses_val) / len(losses_val)
        self.metrics_process(metrics_train, metrics_val)

        train_data = [input_train, ]
        data_compose(self.model_parameters[0], [input_train, result_train], [input_val, result_val])

        return avg_loss_train, avg_loss_val

    def data_extraction(self):
        path = f'./pretrained_model/{self.loss_method}/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth'
        pretrained_information = torch.load(path, map_location=self.device)
        print(f'Weight loaded from {path}')
        state_dict = pretrained_information['weight']
        self.load_weights(state_dict)
        self.train_losses = pretrained_information['losses_train']
        self.val_losses = pretrained_information['losses_val']

        print(f'-------Weight Loaded From {self.check_point} epoch-------')
        loss = self.extraction_epoch()
        print(f'Average loss is {loss}')

    def model_checkpoint_save(self, epoch, losses_train, losses_val):
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU/{self.stage}', exist_ok=True)
        with open(f'./weight/{self.NUM_GPU}_GPU/{self.stage}/readme.txt', 'w') as file:
            file.write(f"MODEL: {self.model_name}\n")
            file.write(f"LOSS METHOD: {self.loss_method}\n")
            file.write(f"BATCH SIZE: {self.batch_size}\n")
            file.write(f"IF NORMALIZED: TRAIN-{self.train_norm} / VAL-{self.val_norm}\n")
            file.write('--------------------------------------------------------------\n')
            file.write(f"LOSS OF {epoch + 1} EPOCH: \nTRAIN LOSS: {losses_train[epoch]}\nVAL LOSS: {losses_val[epoch]}\n")
            file.write('--------------------------------------------------------------\n')
        # Save model state_dict, ensure 'module.' prefix is handled
        if isinstance(self.model, nn.DataParallel):
            # Save the model without 'module.' prefix
            model_state_dict = self.model.module.state_dict()
        else:
            model_state_dict = self.model.state_dict()
        torch.save({'weight': model_state_dict,
                    'epoch': epoch,
                    'losses_train': losses_train,
                    'losses_val': losses_val,
                    'stage': self.stage,
                    'lr': self.lr
                    },
                   f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{epoch + 1}_epoch.pth')
        print('-------Model Saved-------')

    def train(self):
        if self.if_load_weight:
            pretrained_information = torch.load(f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth', map_location=self.device)
            state_dict = pretrained_information['weight']
            self.load_weights(state_dict)
            pretrained_epoch = pretrained_information['epoch']
            self.train_losses = pretrained_information['losses_train']
            self.val_losses = pretrained_information['losses_val']
            print(f'-------Weight Loaded From {self.stage}/{self.check_point} epoch-------')
            for epoch in range(self.epochs):
                loss_train = self.train_epoch(epoch + 1 + self.check_point)
                loss_val = self.val_epoch(epoch + 1 + self.check_point)
                self.train_losses.append(loss_train)
                self.val_losses.append(loss_val)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {pretrained_epoch + epoch + 2}/{self.epochs + pretrained_epoch + 1}\n"
                          f"Train Total Loss: {loss_train:.6f}\n"
                          f"Val Total Loss: {loss_val:.6f}\n-----------------------")
                if (epoch + 1) % 20 == 0:
                    self.model_checkpoint_save(epoch + self.check_point, self.train_losses, self.val_losses)
        else:
            print('-------Train From Beginning-------')
            for epoch in range(self.epochs):
                loss_train = self.train_epoch(epoch + 1)
                loss_val = self.val_epoch(epoch + 1)
                self.train_losses.append(loss_train)
                self.val_losses.append(loss_val)
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}\nTrain Total Loss: {loss_train:.6f}\n"
                          f"Val Total Loss: {loss_val:.6f}\n-----------------------")
                if (epoch + 1) % 20 == 0:
                    self.model_checkpoint_save(epoch, self.train_losses, self.val_losses)
        print('-------Training Complete-------')
        self.writer.close()

    def data_compose(self, input, predict):
        item = input.squeeze().shape[1] / 148
        input = input.squeeze().reshape(148, item, 148)
        predict = predict.squeeze().reshape(148, item, 148)
        composed_data = []
        for i in range(item):
            even_proj = input[:, i, :]
            odd_proj = predict[:, i, :]
            composed_data.append(even_proj)
            composed_data.append(odd_proj)
        composed_data = np.transpose(composed_data, (1, 0, 2))
        return composed_data

    def overall_infer(self):
        pretrained_information_2_4 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/2-4/model_checkpoint_300_epoch.pth', map_location=self.device)
        pretrained_information_4_8 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/4-8/model_checkpoint_300_epoch.pth', map_location=self.device)
        pretrained_information_8_16 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/8-16/model_checkpoint_300_epoch.pth', map_location=self.device)
        pretrained_information_16_32 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/16-32/model_checkpoint_300_epoch.pth', map_location=self.device)

        # Load weights
        self.model1 = UNet().to(self.device)
        self.model2 = UNet().to(self.device)
        self.model3 = UNet().to(self.device)
        self.model4 = UNet().to(self.device)

        # Adjust models for multiple GPUs if necessary
        if self.NUM_GPU > 1:
            self.model1 = nn.DataParallel(self.model1)
            self.model2 = nn.DataParallel(self.model2)
            self.model3 = nn.DataParallel(self.model3)
            self.model4 = nn.DataParallel(self.model4)

        self.load_weights(pretrained_information_2_4['weight'])
        self.load_weights(pretrained_information_4_8['weight'])
        self.load_weights(pretrained_information_8_16['weight'])
        self.load_weights(pretrained_information_16_32['weight'])

        for input1, target1 in self.train_loader:
            input1 = input1.unsqueeze(1).to(self.device)
            target1 = target1.unsqueeze(1).to(self.device)

            predict1 = self.model1(input1)
            loss1 = self.loss_fn(predict1, target1)
            input2 = self.data_compose(input1, predict1)
            predict2 = self.model2(input2)
            loss2 = self.loss_fn(predict2, target1)
            # Continue the inference process as needed

    def stage_infer(self):
        pretrained_information_2_4 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/2-4/model_checkpoint_300_epoch.pth', map_location=self.device)
        self.model = UNet().to(self.device)
        if self.NUM_GPU > 1:
            self.model = nn.DataParallel(self.model)
        self.load_weights(pretrained_information_2_4['weight'])

        for input, target in self.train_loader:
            input = input.unsqueeze(1).to(self.device)
            target = target.unsqueeze(1).to(self.device)
            predict = self.model(input)
            loss = self.loss_fn(predict, target)
            # Process outputs as needed
