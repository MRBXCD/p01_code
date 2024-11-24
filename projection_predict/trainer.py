import wandb
import skimage.metrics as sk
from sklearn.metrics import root_mean_squared_error as rmse
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.utils import data_compose
from loss_method import PerceptualLoss, CombinedLoss
# from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from models.attention_unet import AttU_Net
from models.unetpp import Unetpp
from models.model import UNet
from models.model_mutihead import UNet3Head

# Import 3 models individually, train them with 3 individual loss
from models_3_path.model1 import UNet3_1
from models_3_path.model2 import UNet3_2
from models_3_path.model3 import UNet3_3

from projectionDataloader import ProjectionDataset, ProjectionDataset_inference, ProjectionDataset_FineTune
import os
import torchvision.transforms.functional as F
from collections import OrderedDict

from utils.earlyStop import EarlyStopping

class Trainer:
    def __init__(self, params):
        self.exp_id = params.exp_id
        # Define model
        self.NUM_GPU = torch.cuda.device_count()
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU', exist_ok=True)

        self.if_infer = params.if_infer
        self.batch_size = params.batch_size
        self.net = params.net

        self.lr = params.lr
        self.scheduler = params.scheduler
        self.tmax = params.tmax

        self.data_path = params.data_path

        self.epochs = params.epochs
        self.device = params.device
        self.if_load_weight = params.if_load_weight
        self.check_point = params.check_point
        self.stage = params.stage
        self.if_norm = params.norm
        self.loss_method = params.loss_method
        # self.loss_fn = torch.nn.MSELoss()
        # self.loss_ssim = SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=1)
        if self.loss_method == 'Perceptual':
            self.loss_fn = PerceptualLoss().to(self.device)
        elif self.loss_method == 'L1':
            self.loss_fn = nn.L1Loss()
        elif self.loss_method == 'MSE':
            self.loss_fn = nn.MSELoss()
        elif self.loss_method == 'Combined_loss':
            self.loss_fn = CombinedLoss().to(self.device)

        self.if_extraction = params.if_extraction

        # Define losses
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []

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

        if self.net == 'unet':
            self.model = UNet()
        elif self.net == 'unet++':
            self.model = Unetpp()
        elif self.net == 'atten_unet':
            self.model = AttU_Net()
        
        # Load train data
        self.data_stage = hyper_parameters[self.stage][1]
        self.train_input = os.path.join(self.data_path, f'Projection_train_data_{self.data_stage}_angles_padded.npz')
        # self.train_input = f'/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_{self.data_stage}_angles_padded.npz'
        print(f'Loaded training data from: {self.train_input}')

        # Load val data
        self.val_input = os.path.join(self.data_path, f'Projection_val_data_{self.data_stage}_angles_padded.npz')
        # self.val_input = f'/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_val_data_{self.data_stage}_angles_padded.npz'
        print(f'Loaded val data from: {self.val_input}')

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


        self.model_name = self.net
        self.model.to(self.device)
        # If more than one GPU is available, use DataParallel
        if self.NUM_GPU > 1:
            print(f"Using {self.NUM_GPU} GPUs for training")
            self.model = nn.DataParallel(self.model)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.early_stopping = EarlyStopping(patience=params.patience, verbose=True)
        
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
        predict = predict.detach().squeeze().cpu().numpy()
        target = target.detach().squeeze().cpu().numpy()
        # normalized_rmse = sk.normalized_root_mse(target, predict, normalization='min-max')
        root_mse = rmse(target.flatten(), predict.flatten())
        psnr = sk.peak_signal_noise_ratio(target, predict, data_range=np.max(target) - np.min(target))
        ssim = sk.structural_similarity(target, predict, data_range=np.max(target) - np.min(target))
        return (root_mse, psnr, ssim)

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
        if self.if_extraction:
            with open(f'./save_for_paper/{self.stage}_metrics.txt', 'w') as file:
                file.write("train/val\n")
                file.write(f"avg_rmse:{avg_nrmse_train, avg_nrmse_val}\n")
                file.write(f"avg_psnr:{avg_psnr_train, avg_psnr_val}\n")
                file.write(f"avg_ssim:{avg_ssim_train, avg_ssim_val}\n")
            return (avg_nrmse_train, avg_nrmse_val), (avg_psnr_train, avg_psnr_val), (avg_ssim_train, avg_ssim_val)

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
        metrics = []
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.tmax, eta_min=1e-7)
        for input, target in self.train_loader:
            input = input.unsqueeze(1).to(self.device)
            target = target.unsqueeze(1).to(self.device)
            predict = self.model(input)
            loss = self.loss_fn(predict, target)
            metrics.append(self.evaluation_metrics(predict, target))
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_losses.append(loss.item())
        if self.scheduler:
            scheduler.step()
        avg_total_loss = sum(total_losses[-len(self.train_loader):]) / len(self.train_loader)
        avg_normalized_rmse = sum(metric[0] for metric in metrics) / len(metrics)
        avg_normalized_psnr = sum(metric[1] for metric in metrics) / len(metrics)
        avg_normalized_ssim = sum(metric[2] for metric in metrics) / len(metrics)
        return avg_total_loss, avg_normalized_rmse, avg_normalized_psnr, avg_normalized_ssim

    def val_epoch(self, epoch):
        self.model.eval()
        total_losses = []
        metrics = []
        with torch.no_grad():
            for input, target in self.val_loader:
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)
                predict = self.model(input)
                loss = self.loss_fn(predict, target)
                metrics.append(self.evaluation_metrics(predict, target))
                total_losses.append(loss.item())
        avg_total_loss = sum(total_losses[-len(self.val_loader):]) / len(self.val_loader)
        avg_normalized_rmse = sum(metric[0] for metric in metrics) / len(metrics)
        avg_normalized_psnr = sum(metric[1] for metric in metrics) / len(metrics)
        avg_normalized_ssim = sum(metric[2] for metric in metrics) / len(metrics)
        return avg_total_loss, avg_normalized_rmse, avg_normalized_psnr, avg_normalized_ssim

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
        nrmse, psnr, ssim = self.metrics_process(metrics_train, metrics_val)

        data_compose(self.model_parameters[0], [input_train, result_train], [input_val, result_val])

        return (avg_loss_train, avg_loss_val), nrmse, psnr, ssim

    def data_extraction(self):
        path = f'./pretrained_model/{self.loss_method}/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth'
        pretrained_information = torch.load(path, map_location=self.device)
        print(f'Weight loaded from {path}')
        state_dict = pretrained_information['weight']
        self.load_weights(state_dict)
        self.train_metrics = pretrained_information['metrics_train']
        self.val_metrics = pretrained_information['metrics_val']
        self.train_losses = [metric[0] for metric in self.train_metrics]
        self.val_losses = [metric[0] for metric in self.val_metrics]

        print(f'-------Weight Loaded From {self.check_point} epoch-------')
        metric = self.extraction_epoch()
        print(f'Average loss: {metric[0]}, Average RMSE: {metric[1]}, Average PSNR: {metric[2]}, Average SSIM: {metric[3]},')

    # def model_checkpoint_save(self, epoch, metrics_train, metrics_val):
    #     os.makedirs(f'./weight/{self.NUM_GPU}_GPU/{self.stage}', exist_ok=True)
    #     # Save model state_dict, ensure 'module.' prefix is handled
    #     if isinstance(self.model, nn.DataParallel):
    #         # Save the model without 'module.' prefix
    #         model_state_dict = self.model.module.state_dict()
    #     else:
    #         model_state_dict = self.model.state_dict()
    #     torch.save({'weight': model_state_dict,
    #                 'epoch': epoch,
    #                 'metrics_train': metrics_train, # here metrics include (loss, nrmse, psnr, ssim)
    #                 'metrics_val': metrics_val,
    #                 'stage': self.stage,
    #                 'lr': self.lr
    #                 },
    #                f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{epoch + 1}_epoch.pth')
    #     wandb.save(f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{epoch + 1}_epoch.pth', 
    #                base_path=f'./weight/{self.NUM_GPU}_GPU/{self.stage}',
    #                policy='live'
    #                )
    #     print('-------Model Saved-------')

    def train(self):
        if self.if_load_weight:
            pretrained_information = torch.load(f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth', map_location=self.device)
            state_dict = pretrained_information['weight']
            self.load_weights(state_dict)
            pretrained_epoch = pretrained_information['epoch']
            self.train_metrics = pretrained_information['metrics_train']
            self.val_metrics = pretrained_information['metrics_val']
            print(f'-------Weight Loaded From {self.stage}/{self.check_point} epoch-------')
            for epoch in range(self.epochs):
                metric_train = self.train_epoch(epoch + 1 + self.check_point)
                metric_val = self.val_epoch(epoch + 1 + self.check_point)
                self.train_metrics.append(metric_train)
                self.val_metrics.append(metric_val)
                wandb.log(
                        {
                            'Present Epoch': pretrained_epoch + epoch + 2,
                            'Total Epoch': self.epochs + pretrained_epoch + 1,
                            'Train Total Loss': round(metric_train[0],6),
                            'Val Total Loss': round(metric_val[0],6),
                            'Train Total RMSE': round(metric_train[1],6),
                            'Val Total RMSE': round(metric_val[1],6),
                            'Train Total PSNR': round(metric_train[2],6),
                            'Val Total PSNR': round(metric_val[2],6),
                            'Train Total SSIM': round(metric_train[3],6),
                            'Val Total SSIM': round(metric_val[3],6)
                        }
                    )
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {pretrained_epoch + epoch + 2}/{self.epochs + pretrained_epoch + 1}\n"
                          f"Train:  Loss - {metric_train[0]:.6f}  | RMSE - {metric_train[1]:.6f}  | PSNR - {metric_train[2]:.6f}  | SSIM - {metric_train[3]:.6f}\n"
                          f"Val:    Loss - {metric_val[0]:.6f}  | RMSE - {metric_val[1]:.6f}  | PSNR - {metric_val[2]:.6f}  | SSIM - {metric_val[3]:.6f}\n"
                          '------------------------------------------------------------------------------------------------------------------------')
                if (epoch + 1) % 20 == 0:
                    print("lr = {:.10f}".format(self.optimizer.param_groups[0]['lr']))
                    # self.model_checkpoint_save(epoch + self.check_point, self.train_metrics, self.val_metrics)

                self.early_stopping(metric_val[0], self.model, self.exp_id, self.net)

                if self.early_stopping.early_stop:
                    print("Early stopped, training terminated")
                    break

        else:
            print('-------Train From Beginning-------')
            for epoch in range(self.epochs):
                metric_train = self.train_epoch(epoch + 1)
                metric_val = self.val_epoch(epoch + 1)
                self.train_metrics.append(metric_train)
                self.val_metrics.append(metric_val)
                wandb.log(
                        {
                            'Present Epoch': epoch + 1,
                            'Total Epoch': self.epochs,
                            'Train Total Loss': round(metric_train[0],6),
                            'Val Total Loss': round(metric_val[0],6),
                            'Train Total RMSE': round(metric_train[1],6),
                            'Val Total RMSE': round(metric_val[1],6),
                            'Train Total PSNR': round(metric_train[2],6),
                            'Val Total PSNR': round(metric_val[2],6),
                            'Train Total SSIM': round(metric_train[3],6),
                            'Val Total SSIM': round(metric_val[3],6)
                        }
                    )
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}/{self.epochs}\n"
                          f"Train:  Loss - {metric_train[0]:.6f}  | RMSE - {metric_train[1]:.6f}  | PSNR - {metric_train[2]:.6f}  | SSIM - {metric_train[3]:.6f}\n"
                          f"Val:    Loss - {metric_val[0]:.6f}  | RMSE - {metric_val[1]:.6f}  | PSNR - {metric_val[2]:.6f}  | SSIM - {metric_val[3]:.6f}\n"
                          "------------------------------------------------------------------------------------------------------------------------") 
                if (epoch + 1) % 20 == 0:
                    print("lr = {:.10f}".format(self.optimizer.param_groups[0]['lr']))
                    # self.model_checkpoint_save(epoch, self.train_metrics, self.val_metrics)
                
                self.early_stopping(metric_val[0], self.model, self.exp_id, self.net)

                if self.early_stopping.early_stop:
                    print("Early stopped, training terminated")
                    break
        print('-------Training Complete-------')

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

    # The following code is not up to date and will be updated in the future, DO NOT use the following function
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
