import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ProjectionDataset_16_32(Dataset):
    def __init__(self, input_file, if_norm):
        self.norm = if_norm
        self.data = np.load(input_file)['arr_0']
        if int(np.max(self.data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is 32 prediction model, only the input data will be loaded, the target will be created automatically')
        self.init_mse = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_projections = self.data[idx]

        if self.norm:
            for index in range(input_projections.shape[1]):
                print(np.max(input_projections[:,index,:]))
                input_projections[:,index,:] = input_projections[:,index,:] / np.max(input_projections[:,index,:])
                print(np.max(input_projections[:,index,:]))

        input_index = [0,2,4,6,8,10,12,14,16,18,20,22,24,26,28,30]
        target_index = [1,3,5,7,9,11,13,15,17,19,21,23,25,27,29,31]
        input_data = input_projections[:,input_index,:]
        target_data = input_projections[:,target_index,:]

        inputs = input_data.reshape(148, input_data.shape[1]*148)
        target = target_data.reshape(148, input_data.shape[1]*148)

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.data)) == 1:
            return True
        else:
            return False

        
class ProjectionDataset_2_4(Dataset):
    def __init__(self, input_file, if_norm):
        self.norm = if_norm
        self.data = np.load(input_file)['arr_0']
        if int(np.max(self.data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is 32 prediction model, only the input data will be loaded, the target will be created automatically')
        self.init_mse = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_projections = self.data[idx]

        if self.norm:
            for index in range(input_projections.shape[1]):
                print(np.max(input_projections[:,index,:]))
                input_projections[:,index,:] = input_projections[:,index,:] / np.max(input_projections[:,index,:])
                print(np.max(input_projections[:,index,:]))

        input_index = [0,2]
        target_index = [1,3]
        input_data = input_projections[:,input_index,:]
        target_data = input_projections[:,target_index,:]

        inputs = input_data.reshape(148, input_data.shape[1]*148)
        target = target_data.reshape(148, input_data.shape[1]*148)

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.data)) == 1:
            return True
        else:
            return False        

class ProjectionDataset_16(Dataset):
    def __init__(self, input_file, if_norm):
        self.norm = if_norm
        self.data = np.load(input_file)['arr_0']
        if int(np.max(self.data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is 32 prediction model, only the input data will be loaded, the target will be created automatically')
        self.init_mse = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_projections = self.data[idx]

        input_index = [0,2,4,6,8,10,12,14]
        target_index = [1,3,5,7,9,11,13,15]

        input_data = input_projections[:,input_index,:]
        target_data = input_projections[:,target_index,:]

        inputs = input_data.reshape(148, input_data.shape[1]*148)
        target = target_data.reshape(148, input_data.shape[1]*148)

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.data)) == 1:
            return True
        else:
            return False


class ProjectionDataset_32(Dataset):
    def __init__(self, input_file, if_norm):
        self.norm = if_norm
        self.data = np.load(input_file)['arr_0']
        if int(np.max(self.data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is 32 prediction model, only the input data will be loaded, the target will be created automatically')
        self.init_mse = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_projections = self.data[idx]

        if self.norm:
            for index in range(input_projections.shape[1]):
                input_projections[:,index,:] = input_projections[:,index,:] / np.max(input_projections[:,index,:])
        
        input_index = [0,4,8,12,16,20,24,28]
        target_index_1 = [1,5,9,13,17,21,25,29]
        target_index_2 = [2,6,10,14,18,22,26,30]
        target_index_3 = [3,7,11,15,19,23,27,31]

        input_data = input_projections[:,input_index,:]
        target_data_1 = input_projections[:,target_index_1,:]
        target_data_2 = input_projections[:,target_index_2,:]
        target_data_3 = input_projections[:,target_index_3,:]

        inputs = input_data.reshape(148, input_data.shape[1]*148)
        target_1 = target_data_1.reshape(148, input_data.shape[1]*148)
        target_2 = target_data_2.reshape(148, input_data.shape[1]*148)
        target_3 = target_data_3.reshape(148, input_data.shape[1]*148)

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target_1, dtype=torch.float), torch.tensor(target_2, dtype=torch.float), torch.tensor(target_3, dtype=torch.float)

    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.data)) == 1:
            return True
        else:
            return False