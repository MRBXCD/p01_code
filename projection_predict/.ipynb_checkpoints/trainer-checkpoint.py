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
            self.model.to(self.device)
        elif self.stage == '4-8':
            self.model = UNet()
            self.model_name = 'UNet()'
            self.model.to(self.device)
        elif self.stage == '8-16':
            self.model = UNet()
            self.model_name = 'UNet()'
            self.model.to(self.device)
        elif self.stage == '16-32':
            self.model = UNet()
            self.model_name = 'UNet()'
            self.model.to(self.device)
        elif self.stage == '8-32_1Enc_mutiDec':
            self.model = UNet3Head()
            self.model_name = 'model_mutihead.UNet3Head()'
            self.model.to(self.device)
        elif self.stage == '8-32_mutiEnc_mutiDec':
            self.model1 = UNet3_1()
            self.model2 = UNet3_2()
            self.model3 = UNet3_3()
            self.model_name = './models_3_path/model1->3'
            self.model1.to(self.device)
            self.model2.to(self.device)
            self.model3.to(self.device)
        elif self.if_infer: 
            self.model1 = UNet()
            self.model2 = UNet()
            self.model3 = UNet()
            self.model4 = UNet()
        # initialize tensorboard
        self.writer = SummaryWriter(f'logs/{self.stage}_training')

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

    def data_show(self, dataloader, tit):
        first_batch_inputs, first_batch_targets = next(iter(dataloader))
        # 假设我们想要展示每个体素的中间切片
        # 创建一个大图来展示所有切片
        # 调整 figsize 参数来控制图像显示的大小，DPI 控制图像的清晰度
        fig, axs = plt.subplots(2, 1)  # 调整figsize和dpi以满足需求

        # 调整子图之间的间距
        plt.subplots_adjust(wspace=0.1, hspace=0.2)  # 减小宽度间距和高度间距

        
        # 对于GPU上的数据，确保先将其移动到CPU，并转换为numpy数组
        input = first_batch_inputs[0, :, :].cpu().numpy()
        target = first_batch_targets[0, :, :].cpu().numpy()

        # 展示输入图像的切片
        axs[0].imshow(input, cmap='gray')
        axs[0].set_title(f'Input #{1}')
        axs[0].axis('off')  # 不显示坐标轴

        # 展示目标图像的切片
        axs[1].imshow(target, cmap='gray')
        axs[1].set_title(f'Target #{1}')
        axs[1].axis('off')  # 不显示坐标轴

        # 给整个大图设置标题
        fig.suptitle(tit)
        
        # 调整整个大图的布局，确保标题和子图之间的间距足够
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图像，调整DPI以提高保存图像的清晰度
        plt.savefig(f'{tit}.png', dpi=150)

    def train_epoch(self, epoch):
        self.model.train()
        total_losses = []

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        # self.model_save()
        # print('|model saved|')

        for input, target in self.train_loader:
            
            input = input.unsqueeze(1).to(self.device)
            target = target.unsqueeze(1).to(self.device)

            predict = self.model.forward(input)
            # prediction_voxel = torch.squeeze(prediction_voxel)
            if self.loss_method == 'MSE/SSIM':
                loss = self.mix_loss(predict, target)
            elif self.loss_method == 'SSIM':
                loss = self.SSIM_loss(predict, target)
            else:
                loss = self.loss_fn(predict, target)
            # self.visualization(noisy_voxel, prediction_voxel, clean_voxel, epoch)
            self.writer.add_scalar('training total loss', loss, epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_losses.append(loss.item())
        avg_total_loss = sum(total_losses[-len(self.train_loader):]) / len(self.train_loader)
        return avg_total_loss
    
    def train_epoch_3_head(self, epoch):
        '''
            This is the train epoch function for 8 to 32 angles predict. 
            This function needs to be activated manually by replacing the epoch function in train process function
        '''
        if self.stage == '8-32_mutiEnc_mutiDec':
            self.model1.train()
            self.model1.train()
            self.model1.train()
            optimizer1 = torch.optim.Adam(self.model1.parameters(), lr=self.lr)
            optimizer2 = torch.optim.Adam(self.model2.parameters(), lr=self.lr)
            optimizer3 = torch.optim.Adam(self.model3.parameters(), lr=self.lr)
        else:
            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        losses1 = []
        losses2 = []
        losses3 = []
        total_losses = []

        # self.model_save()
        # print('|model saved|')

        for input, target1, target2, target3 in self.train_loader:

            input = input.unsqueeze(1).to(self.device)
            target1 = target1.unsqueeze(1).to(self.device)
            target2 = target2.unsqueeze(1).to(self.device)
            target3 = target3.unsqueeze(1).to(self.device)

            if self.stage == '8-32_mutiEnc_mutiDec':
                predict1 = self.model1.forward(input)
                predict2 = self.model2.forward(input)
                predict3 = self.model3.forward(input)
            else:
                predict1, predict2, predict3 = self.model.forward(input)

            if self.loss_method == 'MSE/SSIM':
                loss1 = self.mix_loss(predict1, target1)
                loss2 = self.mix_loss(predict2, target2)
                loss3 = self.mix_loss(predict3, target3)
            elif self.loss_method == 'SSIM':
                loss1 = self.SSIM_loss(predict1, target1)
                loss2 = self.SSIM_loss(predict2, target2)
                loss3 = self.SSIM_loss(predict3, target3)
            else:
                loss1 = self.loss_fn(predict1, target1)
                loss2 = self.loss_fn(predict2, target2)
                loss3 = self.loss_fn(predict3, target3)
            loss = loss1 + loss2 + loss3

            self.writer.add_scalar('training total loss', loss, epoch)

            if self.stage == '8-32_mutiEnc_mutiDec':
                optimizer1.zero_grad()
                optimizer2.zero_grad()
                optimizer3.zero_grad()
                loss1.backward()
                loss2.backward()
                loss3.backward()
                optimizer1.step()
                optimizer2.step()
                optimizer3.step()
            else:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            losses1.append(loss1.item())
            losses2.append(loss2.item())
            losses3.append(loss3.item())
            total_losses.append(loss.item())
        avg_1_loss = sum(losses1[-len(self.val_loader):]) / len(self.val_loader)
        avg_2_loss = sum(losses2[-len(self.val_loader):]) / len(self.val_loader)
        avg_3_loss = sum(losses3[-len(self.val_loader):]) / len(self.val_loader)
        avg_total_loss = sum(total_losses[-len(self.train_loader):]) / len(self.train_loader)
        return [avg_total_loss, avg_1_loss, avg_2_loss, avg_3_loss]
    
    def val_epoch_3_head(self, epoch):
        if self.stage == '8-32_mutiEnc_mutiDec':
            self.model1.eval()
            self.model1.eval()
            self.model1.eval()
        else:
            self.model.eval()
        losses1 = []
        losses2 = []
        losses3 = []
        total_losses = []

        with torch.no_grad():
            for input, target1, target2, target3 in self.val_loader:
                input = input.unsqueeze(1).to(self.device)
                target1 = target1.unsqueeze(1).to(self.device)
                target2 = target2.unsqueeze(1).to(self.device)
                target3 = target3.unsqueeze(1).to(self.device)

                if self.stage == '8-32_mutiEnc_mutiDec':
                    predict1 = self.model1.forward(input)
                    predict2 = self.model2.forward(input)
                    predict3 = self.model3.forward(input)
                else:
                    predict1, predict2, predict3 = self.model.forward(input)
                
                if self.loss_method == 'MSE/SSIM':
                    loss1 = self.mix_loss(predict1, target1)
                    loss2 = self.mix_loss(predict2, target2)
                    loss3 = self.mix_loss(predict3, target3)
                elif self.loss_method == 'SSIM':
                    loss1 = self.SSIM_loss(predict1, target1)
                    loss2 = self.SSIM_loss(predict2, target2)
                    loss3 = self.SSIM_loss(predict3, target3)
                else:
                    loss1 = self.loss_fn(predict1, target1)
                    loss2 = self.loss_fn(predict2, target2)
                    loss3 = self.loss_fn(predict3, target3)
                loss = loss1 + loss2 + loss3
                
                self.writer.add_scalar('val total loss', loss, epoch)
                
                losses1.append(loss1.item())
                losses2.append(loss2.item())
                losses3.append(loss3.item())
                total_losses.append(loss.item())
        avg_1_loss = sum(losses1[-len(self.val_loader):]) / len(self.val_loader)
        avg_2_loss = sum(losses2[-len(self.val_loader):]) / len(self.val_loader)
        avg_3_loss = sum(losses3[-len(self.val_loader):]) / len(self.val_loader)
        avg_total_loss = sum(total_losses[-len(self.val_loader):]) / len(self.val_loader)
        return [avg_total_loss, avg_1_loss, avg_2_loss, avg_3_loss]
    
    def val_epoch(self, epoch):
        self.model.eval()
        total_losses = []

        with torch.no_grad():
            for input, target in self.val_loader:
                input = input.unsqueeze(1).to(self.device)
                target = target.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                predict = self.model.forward(input)
                if self.loss_method == 'MSE/SSIM':
                    loss = self.mix_loss(predict, target)
                elif self.loss_method == 'SSIM':
                    loss = self.SSIM_loss(predict, target)
                else:
                    loss = self.loss_fn(predict, target)
                # self.visualization(noisy_voxel, prediction_voxel, clean_voxel, epoch)
                self.writer.add_scalar('val total loss', loss, epoch)

                total_losses.append(loss.item())
        avg_total_loss = sum(total_losses[-len(self.val_loader):]) / len(self.val_loader)
        return avg_total_loss


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
        np.savetxt(f'./result/inference/logs/loss_output_val_{self.loss_method}_{self.stage}.txt', losses_val)
        # avg_loss_val = sum(losses_val[-len(self.val_loader):]) / len(self.val_loader)
        avg_loss_val = sum(losses_val) / len(losses_val)
        return avg_loss_train, avg_loss_val
    
    def extraction_epoch_32(self):
        self.model.eval()
        losses_train = []

        result_train = []
        result_val = []

        losses_val = []
        result_val = []
        
        os.makedirs('./result/inference/logs', exist_ok=True)

        with torch.no_grad():
            for input, target1, target2, target3 in tqdm(self.train_loader, desc='Extracting train'):
                input = input.unsqueeze(1).to(self.device)
                target1 = target1.unsqueeze(1).to(self.device)
                target2 = target2.unsqueeze(1).to(self.device)
                target3 = target3.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_image1, prediction_image2, prediction_image3 = self.model.forward(input)
                if self.loss_method == 'MSE/SSIM':
                    loss_train1 = self.mix_loss(prediction_image1, target1)
                    loss_train2 = self.mix_loss(prediction_image2, target2)
                    loss_train3 = self.mix_loss(prediction_image3, target3)
                elif self.loss_method == 'SSIM':
                    loss_train1 = self.SSIM_loss(prediction_image1, target1)
                    loss_train2 = self.SSIM_loss(prediction_image2, target2)
                    loss_train3 = self.SSIM_loss(prediction_image3, target3)
                else:
                    loss_train1 = self.loss_fn(prediction_image1, target1)
                    loss_train2 = self.loss_fn(prediction_image2, target2)
                    loss_train3 = self.loss_fn(prediction_image3, target3)
                loss_train = loss_train1 + loss_train2 + loss_train3
                losses_train.append(loss_train.item())
                
                output1 = input.squeeze(1).cpu().numpy().reshape(148,8,148)
                output2 = prediction_image1.squeeze(1).cpu().numpy().reshape(148,8,148)
                output3 = prediction_image2.squeeze(1).cpu().numpy().reshape(148,8,148)
                output4 = prediction_image3.squeeze(1).cpu().numpy().reshape(148,8,148)
                
                composed_data = np.zeros((148, 32, 148))
                matrices = [output1, output2, output3, output4]
                for i, matrix in enumerate(matrices):
                     for j in range(8):
                        target_index = j*4 + i
                        composed_data[:, target_index, :] = matrix[:, j, :]
                result_train.append(composed_data)

        print(np.shape(result_train))
        np.savez(f'./result/inference/{self.stage}_model_output_train.npz', result_train)
        np.savetxt('./result/inference/logs/loss_output_train.txt', losses_train)
        avg_loss_train = sum(losses_train[-len(self.train_loader):]) / len(self.train_loader)
        
        with torch.no_grad():
            for input, target1, target2, target3 in tqdm(self.val_loader, desc='Extracting train'):
                input = input.unsqueeze(1).to(self.device)
                target1 = target1.unsqueeze(1).to(self.device)
                target2 = target2.unsqueeze(1).to(self.device)
                target3 = target3.unsqueeze(1).to(self.device)
                # print(f'Shape of noisy voxel is {noisy_voxel.size()}')

                prediction_image1, prediction_image2, prediction_image3 = self.model.forward(input)
                if self.loss_method == 'MSE/SSIM':
                    loss_val1 = self.mix_loss(prediction_image1, target1)
                    loss_val2 = self.mix_loss(prediction_image2, target2)
                    loss_val3 = self.mix_loss(prediction_image3, target3)
                elif self.loss_method == 'SSIM':
                    loss_train1 = self.SSIM_loss(prediction_image1, target1)
                    loss_train2 = self.SSIM_loss(prediction_image2, target2)
                    loss_train3 = self.SSIM_loss(prediction_image3, target3)
                else:
                    loss_val1 = self.loss_fn(prediction_image1, target1)
                    loss_val2 = self.loss_fn(prediction_image2, target2)
                    loss_val3 = self.loss_fn(prediction_image3, target3)
                loss_val = loss_val1 + loss_val2 + loss_val3
                losses_val.append(loss_val.item())
                
                output1 = input.squeeze(1).cpu().numpy().reshape(148,8,148)
                output2 = prediction_image1.squeeze(1).cpu().numpy().reshape(148,8,148)
                output3 = prediction_image2.squeeze(1).cpu().numpy().reshape(148,8,148)
                output4 = prediction_image3.squeeze(1).cpu().numpy().reshape(148,8,148)
                
                composed_data = np.zeros((148, 32, 148))
                matrices = [output1, output2, output3, output4]
                for i, matrix in enumerate(matrices):
                     for j in range(8):
                        target_index = j*4 + i
                        composed_data[:, target_index, :] = matrix[:, j, :]
                result_val.append(composed_data)
        print(np.shape(result_val))
        np.savez(f'./result/inference/{self.stage}_model_output_val.npz', result_val)
        np.savetxt('./result/inference/logs/loss_output_val.txt', losses_val)
        avg_loss_val = sum(losses_val[-len(self.val_loader):]) / len(self.val_loader)
        return avg_loss_train, avg_loss_val

    def data_extraction(self):
        if self.stage == '8-32_mutiEnc_mutiDec' or self.stage == '8-32_1Enc_mutiDec':
            path = f'/root/autodl-tmp/Projection_predict/pretrained_model/{self.loss_method}/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth'
            pretrained_information = torch.load(path)
            print(f'Weight loaded from {path}')
            self.model1.load_state_dict(pretrained_information['weight1'])
            self.model2.load_state_dict(pretrained_information['weight2'])
            self.model3.load_state_dict(pretrained_information['weight3'])
            self.train_losses = pretrained_information['losses_train']
            self.val_losses = pretrained_information['losses_val']  
            loss = self.extraction_epoch_32()
        else:
            path = f'/root/autodl-tmp/Projection_predict/pretrained_model/{self.loss_method}/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth'
            pretrained_information = torch.load(path)
            print(f'Weight loaded from {path}')
            self.model.load_state_dict(pretrained_information['weight'])
            self.train_losses = pretrained_information['losses_train']
            self.val_losses = pretrained_information['losses_val']
            print(f'-------Weight Loaded From {self.check_point} epoch-------')
            loss = self.extraction_epoch()
        print(f'average loss is {loss}')

    def model_checkpoint_save(self, epoch, losses_train, losses_val):
        os.makedirs(f'./weight/{self.NUM_GPU}_GPU/{self.stage}', exist_ok=True)
        with open(f'./weight/{self.NUM_GPU}_GPU/{self.stage}/readme.txt', 'w') as file:
            file.write(f"MODEL:{self.model_name}\n")
            file.write(f"LOSS METHOD:{self.loss_method}\n")
            file.write(f"BATCH SIZE:{self.batch_size}\n")
            file.write(f"IF NORMALIZED: TRAIN-{self.train_norm} / VAL-{self.val_norm}\n")
            file.write('--------------------------------------------------------------\n')
            file.write(f"LOSS OF {epoch+1} EPOCH: \nTRAIN LOSS: {losses_train[epoch]}\nVAL LOSS: {losses_val[epoch]}\n")
            file.write('--------------------------------------------------------------\n')       
        if self.stage == '8-32_mutiEnc_mutiDec':    
            torch.save({'weight1': self.model1.state_dict(),
                        'weight2': self.model2.state_dict(),
                        'weight3': self.model3.state_dict(),
                        'epoch': epoch,
                        'losses_train': losses_train,
                        'losses_val': losses_val,
                        'stage': self.stage,
                        'lr': self.lr
                        }, 
                    f'./weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{epoch + 1}_epoch.pth')
            print('-------Model Saved-------')
        else:
            torch.save({'weight': self.model.state_dict(),
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
            if self.stage == '8-32_mutiEnc_mutiDec' or self.stage == '8-32_1Enc_mutiDec':
                pretrained_information = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth')
                self.model1.load_state_dict(pretrained_information['weight1'])
                self.model2.load_state_dict(pretrained_information['weight2'])
                self.model3.load_state_dict(pretrained_information['weight3'])
                pretrained_epoch = pretrained_information['epoch']
                self.train_losses = pretrained_information['losses_train']
                self.val_losses = pretrained_information['losses_val']
            else:    
                pretrained_information = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/{self.stage}/model_checkpoint_{self.check_point}_epoch.pth')
                self.model.load_state_dict(pretrained_information['weight'])
                pretrained_epoch = pretrained_information['epoch']
                self.train_losses = pretrained_information['losses_train']
                self.val_losses = pretrained_information['losses_val']
            
            print(f'-------Weight Loaded From {self.stage}/{self.check_point} epoch-------')
            for epoch in range(self.epochs):
                if self.stage == '8-32_mutiEnc_mutiDec' or self.stage == '8-32_1Enc_mutiDec':
                    loss_train = self.train_epoch_3_head(epoch + 1 + self.check_point)
                    loss_val = self.val_epoch_3_head(epoch + 1 + self.check_point)
                else:
                    loss_train = self.train_epoch(epoch + 1 + self.check_point)
                    loss_val = self.val_epoch(epoch + 1 + self.check_point)
                self.train_losses.append(loss_train)
                self.val_losses.append(loss_val)
                #self.model_structure_save()
                if (epoch + 1) % 10 == 0:
                    if self.stage == '8-32_mutiEnc_mutiDec' or self.stage == '8-32_1Enc_mutiDec':
                        print(f"Epoch {pretrained_epoch + epoch + 2}/{self.epochs + pretrained_epoch + 1}\nTrain Total Loss: {loss_train[0]:.6f}, Train 1 Loss: {loss_train[1]:.6f}, Train 2 Loss: {loss_train[2]:.6f}, Train 3 Loss: {loss_train[3]:.6f}\nVal Total Loss: {loss_val[0]:.6f}, Val 1 Loss: {loss_val[1]:.6f}, Val 2 Loss: {loss_val[2]:.6f}, Val 3 Loss: {loss_val[3]:.6f}\n-----------------------")
                    else:
                        print(f"Epoch {pretrained_epoch + epoch + 2}/{self.epochs + pretrained_epoch + 1}\nTrain Total Loss: {loss_train:.6f}\nVal Total Loss: {loss_val:.6f}\n-----------------------")
                if (epoch + 1) % 20 == 0:
                    self.model_checkpoint_save(epoch + self.check_point, self.train_losses, self.val_losses)
                    
                
        else:
            print('-------Train From Beginning-------')
            for epoch in range(self.epochs):
                #self.model_structure_save()
                if self.stage == '8-32_mutiEnc_mutiDec' or self.stage == '8-32_1Enc_mutiDec':
                    loss_train = self.train_epoch_3_head(epoch + 1)
                    loss_val = self.val_epoch_3_head(epoch + 1)
                else:
                    loss_train = self.train_epoch(epoch + 1)
                    loss_val = self.val_epoch(epoch + 1)
                self.train_losses.append(loss_train)
                self.val_losses.append(loss_val)
                if (epoch + 1) % 10 == 0:
                    if self.stage == '8-32_mutiEnc_mutiDec' or self.stage == '8-32_1Enc_mutiDec':
                        print(f"Epoch {epoch + 1}/{self.epochs}\nTrain Total Loss: {loss_train[0]:.6f}, Train 1 Loss: {loss_train[1]:.6f}, Train 2 Loss: {loss_train[2]:.6f}, Train 3 Loss: {loss_train[3]:.6f}\nVal Total Loss: {loss_val[0]:.6f}, Val 1 Loss: {loss_val[1]:.6f}, Val 2 Loss: {loss_val[2]:.6f}, Val 3 Loss: {loss_val[3]:.6f}\n-----------------------")
                    else:
                        print(f"Epoch {epoch + 1}/{self.epochs}\nTrain Total Loss: {loss_train:.6f}\nVal Total Loss: {loss_val:.6f}\n-----------------------")
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
            even_proj = input[:,i,:]
            odd_proj = predict[:,i,:]
            composed_data.append(even_proj)
            composed_data.append(odd_proj)
        composed_data = np.transpose(composed_data, (1,0,2))
        return composed_data

    def overall_infer(self):
        pretrained_information_2_4 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/2-4/model_checkpoint_300_epoch.pth')
        pretrained_information_4_8 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/4-8/model_checkpoint_300_epoch.pth')
        pretrained_information_8_16 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/8-16/model_checkpoint_300_epoch.pth')
        pretrained_information_16_32 = torch.load(f'/root/autodl-tmp/Projection_predict/weight/{self.NUM_GPU}_GPU/16-32/model_checkpoint_300_epoch.pth')
        self.model1.load_state_dict(pretrained_information_2_4['weight'])
        self.model2.load_state_dict(pretrained_information_4_8['weight'])
        self.model3.load_state_dict(pretrained_information_8_16['weight'])
        self.model4.load_state_dict(pretrained_information_16_32['weight'])

        for input1, target1 in self.train_loader:
            input1 = input.unsqueeze(1).to(self.device)
            target1 = target1.unsqueeze(1).to(self.device)

            predict1 = self.model1.forward(input)
            loss1 = self.loss_fn(predict1, target1)
            input2 = self.data_compose(input1, target1)
            predict1 = self.model1.forward(input)
            loss1 = self.loss_fn(predict1, target1)
            input2 = self.data_compose(input1, target1)