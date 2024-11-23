import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np


def voxelTransform(voxel):
    return np.transpose(voxel, (0, 3, 2, 1))


class VoxelDenoiseDataset(Dataset):
    def __init__(self, input_file, target_file, transform=None):
        """
        Args:
            input_file (string): 有噪声输入数据的npz文件路径。
            target_file (string): 无噪声目标数据的npz文件路径。
            transform (callable, optional): 可选的转换函数，用于对样本进行处理。
        """
        self.transform = transform
        self.input_data = np.load(input_file)['arr_0']
        if int(np.max(self.input_data)) != 1:
            print('Caution: Input data is not normalized to 1')
        self.target_data = np.load(target_file)['arr_0']
        if int(np.max(self.target_data)) != 1:
            print('Caution: Target data is not normalized to 1')
        else:
            print('Data checked')
        loss_fn = torch.nn.MSELoss()
        self.init_mse = loss_fn(torch.tensor(self.input_data), torch.tensor(self.target_data))
        print(f'Initial Voxel MSE is {self.init_mse:.6f}')

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):
        input_voxel = self.input_data[idx]
        target_voxel = self.target_data[idx]

        if self.transform:
            input_voxel = self.transform(input_voxel)
            target_voxel = self.transform(target_voxel)

        return torch.tensor(input_voxel, dtype=torch.float), torch.tensor(target_voxel, dtype=torch.float)

    def initial_mse(self):
        return self.init_mse.item()