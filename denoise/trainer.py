import astra
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from model_3D import UNet3D
from voxelDataLoader import VoxelDenoiseDataset
import onnx
import onnx.utils
import onnx.version_converter
import os
import torchvision.transforms.functional as TF
import voxelOperator
from torch.utils.tensorboard import SummaryWriter
from skimage.metrics import structural_similarity as ssim
import skimage.metrics as sk

# def data_show(dataloader, tit):
#     first_batch_inputs, first_batch_targets = next(iter(dataloader))
#     first_batch_inputs = torch.squeeze(first_batch_inputs)
#     first_batch_targets = torch.squeeze(first_batch_targets)
#     # 假设我们想要展示每个体素的中间切片
#     # 计算切片索引，这里我们取深度方向的中间
#     slice_index = first_batch_inputs.shape[2] // 2

#     # 创建一个大图来展示所有切片
#     fig, axs = plt.subplots(2, first_batch_inputs.shape[0], figsize=(20, 10))  # 2行，每个批次的样本数为列数

#     for i in range(first_batch_inputs.shape[0]):
#         # 对于GPU上的数据，确保先将其移动到CPU，并转换为numpy数组
#         input_slice = first_batch_inputs[i, slice_index, :, :].cpu().numpy()
#         target_slice = first_batch_targets[i, slice_index, :, :].cpu().numpy()

#         # 展示输入图像的切片
#         axs[0, i].imshow(input_slice, cmap='gray')
#         axs[0, i].set_title(f'Input #{i + 1}')
#         axs[0, i].axis('off')  # 不显示坐标轴

#         # 展示目标图像的切片
#         axs[1, i].imshow(target_slice, cmap='gray')
#         axs[1, i].set_title(f'Target #{i + 1}')
#         axs[1, i].axis('off')  # 不显示坐标轴
#     plt.title(f'{tit}')
#     plt.savefig(f'{tit}.png')


class Trainer:
    def __init__(self, params):
        # define model
        self.model = UNet3D(in_channel=1)
        self.NUM_GPU = torch.cuda.device_count()
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU', exist_ok=True)
        if self.NUM_GPU > 1:
            print(f'Total GPU number: {self.NUM_GPU}')
            self.model = nn.DataParallel(self.model)
        
        self.batch_size = params.batch_size
        self.lr = params.lr
        self.epochs = params.epochs
        self.device = params.device
        self.mse_loss = nn.MSELoss()
        self.if_load_weight = params.if_load_weight
        self.check_point = params.check_point
        self.stage = params.stage
        self.if_extraction = params.if_extraction
        # define losses
        self.train_losses = []
        self.val_losses = []


        # define the hyper parameters of model, [proj_mse_angle, amount]
        hyper_parameters = {
            '2-4': [128, 0.8],
            '4-8': [64, 0],
            '4-6': [64, 0],
            '4-5': [64, 0.7],
            '8-16': [32, 0],
            '16-360': [16, 0],
            '32-360':[16,0],
            '64-360': [2, 0],
            '128-360': [2, 0.2]
        }
        self.model_parameters = hyper_parameters[self.stage]
        print('Number of projection angles when clac MSE', self.model_parameters[0])
        print('Weight of projection MSE', self.model_parameters[1])

        # load train data
        self.train_input = params.train_input
        self.train_target = params.train_target

        # load val data
        self.val_input = params.val_input
        self.val_target = params.val_target

        # initialize dataloader
        train_dataset = VoxelDenoiseDataset(self.train_input, self.train_target)
        val_dataset = VoxelDenoiseDataset(self.val_input, self.val_target)
        self.init_mse_train = train_dataset.initial_mse()
        self.init_mse_val = val_dataset.initial_mse()
        if not self.if_extraction:
            self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            self.val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
        else:
            self.train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, drop_last=False)
            self.val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, drop_last=False)
        # data_show(self.train_loader, 'train')
        # data_show(self.val_loader, 'val')
        self.model.to(self.device)

        # initialize tensorboard
        self.writer = SummaryWriter(f'logs/{self.stage}_training')

    def loss_mix(self, input, target, amount):
        input_padded, if_list = voxelOperator.padding(input, 10)
        target_padded, if_list = voxelOperator.padding(target, 10)

        if if_list == 0:
            input_projection = voxelOperator.projection(input_padded, self.model_parameters[0], 1000, 50)
            target_projection = voxelOperator.projection(target_padded, self.model_parameters[0], 1000, 50)
            input_projection_tensor = torch.tensor(input_projection).to('cuda')
            target_projection_tensor = torch.tensor(target_projection).to('cuda')
        else:        
            input_projection = []
            target_projection = []
            for index in range(len(input_padded)):
                input_projection.append(voxelOperator.projection(input_padded[index], self.model_parameters[0], 1000, 50))
                target_projection.append(voxelOperator.projection(target_padded[index], self.model_parameters[0], 1000, 50))
            input_projection_tensor = torch.tensor(np.array(input_projection)).to('cuda')
            target_projection_tensor = torch.tensor(np.array(target_projection)).to('cuda')

        proj_mse_raw = self.mse_loss(input_projection_tensor, target_projection_tensor)
        proj_mse = proj_mse_raw / self.model_parameters[0]

        voxel_mse = self.mse_loss(input, target)

        overall_mse = proj_mse * amount + voxel_mse * (1-amount)
        return overall_mse, proj_mse, voxel_mse

    def ssim_loss(self, volRaw, volPredict):
        volRaw_np = volRaw.detach().to('cpu').numpy().squeeze()
        volPredict_np = volPredict.detach().to('cpu').numpy().squeeze()
        # volRaw_np = torch.squeeze(volRaw_np)
        # volPredict_np = torch.suqeeze(volPredict_np)
        ssim3D = ssim(volRaw_np, volPredict_np, data_range = volRaw_np.max()-volRaw_np.min())
        return ssim3D

    def visualization(self, input, target, prediction, epoch):
        input_slice = input.squeeze(1)[:,64,:,:]
        prediction_slice = prediction.squeeze(1)[:,64,:,:]
        target_slice = target.squeeze(1)[:,64,:,:]
        combined = torch.cat((input_slice, prediction_slice, target_slice), 0)
        img_grid = torchvision.utils.make_grid(combined, nrow=input_slice.size(0))
        self.writer.add_image('Input-Predict-Target', img_grid)

    def evaluation_metrics(self, predict, target):
        predict = predict.squeeze()
        target = target.squeeze().cpu().numpy()
        normalized_rmse = sk.normalized_root_mse(target, predict, normalization='min-max')
        psnr = sk.peak_signal_noise_ratio(target, predict, data_range=np.max(target)-np.min(target))
        ssim = sk.structural_similarity(target, predict, data_range=np.max(target)-np.min(target))
        return normalized_rmse, psnr, ssim

    def metrics_process(self, train_metrics, val_metrics):
        train_metrics = np.array(train_metrics)
        val_metrics = np.array(val_metrics)
        avg_nrmse_calc = lambda a:sum(a[:,0])/len(a[:,0])
        avg_nrmse_train = avg_nrmse_calc(train_metrics)
        avg_nrmse_val = avg_nrmse_calc(val_metrics)

        avg_psnr_calc = lambda a:sum(a[:,1])/len(a[:,1])
        avg_psnr_train = avg_psnr_calc(train_metrics)
        avg_psnr_val = avg_psnr_calc(val_metrics)

        avg_ssim_calc = lambda a:sum(a[:,2])/len(a[:,2])
        avg_ssim_train = avg_ssim_calc(train_metrics)
        avg_ssim_val = avg_ssim_calc(val_metrics)
        with open(f'/root/autodl-tmp/denoise/save_for_paper/{self.stage}_metrics.txt', 'w') as file:
            file.write("train/val\n")
            file.write(f"avg_nrmse:{avg_nrmse_train, avg_nrmse_val}\n")
            file.write(f"avg_psnr:{avg_psnr_train, avg_psnr_val}\n")
            file.write(f"avg_ssim:{avg_ssim_train, avg_ssim_val}\n")

    def train_epoch(self, epoch):
        self.model.train()
        total_losses = []
        proj_losses = []
        voxel_losses = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.model_save()
        # print('|model saved|')

        for noisy_voxel, clean_voxel in tqdm(self.train_loader):
            noisy_voxel = noisy_voxel.unsqueeze(1).to(self.device)
            clean_voxel = clean_voxel.unsqueeze(1).to(self.device)
            # noisy_voxel = noisy_voxel.to(self.device)
            # clean_voxel = clean_voxel.to(self.device)
            # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

            prediction_voxel = self.model.forward(noisy_voxel)
            # prediction_voxel = torch.squeeze(prediction_voxel)
            loss = self.ssim_loss(clean_voxel, prediction_voxel)
            # self.visualization(noisy_voxel, prediction_voxel, clean_voxel, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
        avg_loss = sum(losses[-len(self.train_loader):]) / len(self.train_loader)
        return avg_loss

    def val_epoch(self, epoch):
        self.model.eval()
        total_losses = []
        proj_losses = []
        voxel_losses = []

        with torch.no_grad():
            for noisy_voxel, clean_voxel in tqdm(self.val_loader):
                noisy_voxel = noisy_voxel.unsqueeze(1).to(self.device)
                clean_voxel = clean_voxel.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_voxel = self.model.forward(noisy_voxel)
                total_loss, proj_loss, voxel_loss = self.loss_calc(prediction_voxel, clean_voxel, self.model_parameters[1])
                # self.visualization(noisy_voxel, prediction_voxel, clean_voxel, epoch)
                self.writer.add_scalar('val total loss', total_loss, epoch)
                self.writer.add_scalar('val proj loss', proj_loss, epoch)
                self.writer.add_scalar('val voxel loss', voxel_loss, epoch)

                losses.append(loss.item())
        avg_loss = sum(losses[-len(self.val_loader):]) / len(self.val_loader)
        return avg_loss

    def eval_result(self, voxel1, voxel2):
        ssim = torchvision.transforms.functional.ssim

    def extraction_epoch(self):
        self.model.eval()
        losses_train= []
        losses_val = []
        result_train = []
        result_val = []
        metrics_train = []
        metrics_val = []

        with torch.no_grad():
            for noisy_voxel, clean_voxel in tqdm(self.train_loader, desc='Extracting train'):
                noisy_voxel = noisy_voxel.unsqueeze(1).to(self.device)
                clean_voxel = clean_voxel.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_voxel = self.model.forward(noisy_voxel)
                loss_train = self.ssim_loss(clean_voxel, prediction_voxel)
                losses_train.append(loss_train.item())
                prediction_voxel = prediction_voxel.squeeze(1).cpu().numpy()
                result_train.append(prediction_voxel)
                metric_train = self.evaluation_metrics(prediction_voxel, clean_voxel)
                metrics_train.append(metric_train)
        np.savez('Train_Recons_16-32_DL.npz', result_train)
        np.savetxt('loss_32_DL_train.txt', losses_train)
        avg_loss_train = sum(losses_train[-len(self.train_loader):]) / len(self.train_loader)

        with torch.no_grad():
            for noisy_voxel, clean_voxel in tqdm(self.val_loader, desc='Extracting val'):
                noisy_voxel = noisy_voxel.unsqueeze(1).to(self.device)
                clean_voxel = clean_voxel.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_voxel = self.model.forward(noisy_voxel)
                # loss_val = self.loss_calc(prediction_voxel, clean_voxel, self.model_parameters[1])
                loss_val = self.ssim_loss(clean_voxel, prediction_voxel)
                losses_val.append(loss_val.item())
                prediction_voxel = prediction_voxel.squeeze(1).cpu().numpy()
                result_val.append(prediction_voxel)
                metric_val = self.evaluation_metrics(prediction_voxel, clean_voxel)
                metrics_val.append(metric_val)
        np.savez('Val_Recons_16-32_DL.npz', result_val)
        self.metrics_process(metrics_train, metrics_val)
        np.savetxt('loss_4_DL_val.txt', losses_val)
        avg_loss_val = sum(losses_val[-len(self.val_loader):]) / len(self.val_loader)
        return avg_loss_train, avg_loss_val
    
    def data_extraction(self):
        pretrained_information = torch.load(f'/root/autodl-tmp/denoise/weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth')
        self.model.load_state_dict(pretrained_information['weight'])
        pretrained_epoch = pretrained_information['epoch']
        self.train_losses = pretrained_information['losses_train']
        self.val_losses = pretrained_information['losses_val']
        print(f'-------Weight Loaded From {self.check_point} epoch-------')
        loss = self.extraction_epoch()
        print(f'average loss is {loss}')

    def model_checkpoint_save(self, epoch, losses_train, losses_val):
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU/{self.stage}', exist_ok=True)
        torch.save({'weight': self.model.state_dict(),
                    'epoch': epoch,
                    'losses_train': losses_train,
                    'losses_val': losses_val,
                    }, 
                   f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{epoch + 1}_epoch.pth')
        print('-------Model Saved-------')     

    def train(self):
        if self.if_load_weight:
            pretrained_information = torch.load(f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth')
            self.model.load_state_dict(pretrained_information['weight'])
            pretrained_epoch = pretrained_information['epoch']
            self.train_losses = pretrained_information['losses_train']
            self.val_losses = pretrained_information['losses_val']
            
            print(f'-------Weight Loaded From {self.stage}/{self.check_point} epoch-------')
            for epoch in range(self.epochs):
                loss_train = self.train_epoch(epoch + 1 + self.check_point)
                loss_val = self.val_epoch(epoch + 1 + self.check_point)
                improvement_train = -100 * (loss_train[2] - self.init_mse_train) / self.init_mse_train
                improvement_val = -100 * (loss_val[2] - self.init_mse_val) / self.init_mse_val
                self.train_losses.append(loss_train)
                self.val_losses.append(loss_val)
                #self.model_structure_save()
                if (epoch + 1) % 5 == 0:
                    self.model_checkpoint_save(epoch + self.check_point, self.train_losses, self.val_losses)
                    
                print(f"Epoch {pretrained_epoch + epoch + 2}/{self.epochs + pretrained_epoch + 1}\nTrain Total Loss: {loss_train[0]:.6f}, Train Proj Loss: {loss_train[1]:.6f}, Train Voxel Loss: {loss_train[2]:.6f}, Train improvement: {improvement_train:.2f}%\nVal Total Loss: {loss_val[0]:.6f}, Val Proj Loss: {loss_val[1]:.6f}, Val Voxel Loss: {loss_val[2]:.6f}, Val improvement: {improvement_val:.2f}%")
        else:
            print('-------Train From Beginning-------')
            for epoch in range(self.epochs):
                #self.model_structure_save()
                loss_train = self.train_epoch(epoch + 1)
                loss_val = self.val_epoch(epoch + 1)
                improvement_train = -100 * (loss_train[2] - self.init_mse_train) / self.init_mse_train
                improvement_val = -100 * (loss_val[2] - self.init_mse_val) / self.init_mse_val
                self.train_losses.append(loss_train)
                self.val_losses.append(loss_val)
                if (epoch + 1) % 5 == 0:
                    self.model_checkpoint_save(epoch, self.train_losses, self.val_losses)
                print(f"Epoch {epoch + 1}/{self.epochs}\nTrain Total Loss: {loss_train[0]:.6f}, Train Proj Loss: {loss_train[1]:.6f}, Train Voxel Loss: {loss_train[2]:.6f}, Train improvement: {improvement_train:.2f}%\nVal Total Loss: {loss_val[0]:.6f}, Val Proj Loss: {loss_val[1]:.6f}, Val Voxel Loss: {loss_val[2]:.6f}, Val improvement: {improvement_val:.2f}%")
        print('-------Training Complete-------')
        self.writer.close()