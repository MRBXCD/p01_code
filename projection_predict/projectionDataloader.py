import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ProjectionDataset(Dataset):
    def __init__(self, input_file, if_norm):
        self.norm = if_norm
        self.data = np.load(input_file)['arr_0']
        shape = self.data.shape
        self.projection_views = shape[2]
        if int(np.max(self.data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is projection model, only the input data will be loaded, the target will be created automatically')
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

        input_index  = [i for i in range(0, self.projection_views-1, 2)]
        # print(input_index)
        target_index = [i for i in range(1, self.projection_views, 2)]
        # print(target_index)
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

        
class ProjectionDataset_inference(Dataset):
    def __init__(self, former_stage_data, target_stage_data, if_norm):
        self.norm = if_norm
        self.former_stage_data = np.load(former_stage_data)['arr_0']
        self.target_stage_data = np.load(target_stage_data)['arr_0']
        self.projection_views = self.target_stage_data.shape[2]
        if int(np.max(self.former_stage_data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.former_stage_data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is prediction model, only the input data will be loaded, the target will be created automatically')
        self.init_mse = 0

    def __len__(self):
        return len(self.former_stage_data)

    def __getitem__(self, idx):
        input_projections = self.former_stage_data[idx]
        target_stage_projections = self.target_stage_data[idx]

        if self.norm:
            for index in range(input_projections.shape[1]):
                print(np.max(input_projections[:,index,:]))
                input_projections[:,index,:] = input_projections[:,index,:] / np.max(input_projections[:,index,:])
                print(np.max(input_projections[:,index,:]))

        target_index = [i for i in range(1, self.projection_views, 2)]
        target_data = target_stage_projections[:,target_index,:]

        inputs = input_projections.reshape(148, input_projections.shape[1]*148)
        targets = target_data.reshape(148, input_projections.shape[1]*148)

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
    
    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.former_stage_data)) == 1:
            return True
        else:
            return False

class ProjectionDataset_archv2(Dataset):
    def __init__(self, input_file, if_norm):
            self.norm = if_norm
            self.data = np.load(input_file)['arr_0']
            shape = self.data.shape
            self.projection_views = shape[2]
            if int(np.max(self.data)) != 1:
                print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
            else:
                print('Data checked')
            print('Now is projection model, only the input data will be loaded, the target will be created automatically')
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

        input_index  = [i for i in range(0, self.projection_views-1, 2)]
        # print(input_index)
        target_index = [i for i in range(1, self.projection_views, 2)]
        # print(target_index)
        input_data = input_projections[:,input_index,:]
        target_data = input_projections[:,target_index,:]

        inputs = np.transpose(input_data, (1,0,2)).squeeze()
        target = np.transpose(target_data, (1,0,2)).squeeze()

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)

class ProjectionDataset_archv2_5(Dataset):
    def __init__(self, input_file, if_norm):
            self.norm = if_norm
            self.data = np.load(input_file)['arr_0']
            shape = self.data.shape
            self.projection_views = shape[2]
            if int(np.max(self.data)) != 1:
                print(f'Caution: Max value of input data is {np.max(self.data)}, do data normalization')
            else:
                print('Data checked')
            print('Now is projection model, only the input data will be loaded, the target will be created automatically')
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

        input_index  = [i for i in range(0, self.projection_views-1, 2)]
        # print(input_index)
        target_index = [i for i in range(1, self.projection_views, 2)]
        # print(target_index)
        input_data = input_projections[:,input_index,:]
        target_data = input_projections

        inputs = np.transpose(input_data, (1,0,2)).squeeze()
        target = np.transpose(target_data, (1,0,2)).squeeze()

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(target, dtype=torch.float)

    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.data)) == 1:
            return True
        else:
            return False

class ProjectionDataset_FineTune(Dataset):
    def __init__(self, former_stage_data, target_stage_data, if_norm):
        self.norm = if_norm
        self.former_stage_data = np.load(former_stage_data)['arr_0']
        self.target_stage_data = np.load(target_stage_data)['arr_0']
        self.projection_views = self.target_stage_data.shape[2]
        if int(np.max(self.former_stage_data)) != 1:
            print(f'Caution: Max value of input data is {np.max(self.former_stage_data)}, do data normalization')
        else:
            print('Data checked')
        print('Now is prediction model, only the input data will be loaded, the target will be created automatically')
        self.init_mse = 0

    def __len__(self):
        return len(self.former_stage_data)

    def __getitem__(self, idx):
        input_projections = self.former_stage_data[idx]
        target_stage_projections = self.target_stage_data[idx]

        if self.norm:
            for index in range(input_projections.shape[1]):
                print(np.max(input_projections[:,index,:]))
                input_projections[:,index,:] = input_projections[:,index,:] / np.max(input_projections[:,index,:])
                print(np.max(input_projections[:,index,:]))

        target_index = [i for i in range(1, self.projection_views, 2)]
        target_data = target_stage_projections[:,target_index,:]

        inputs = input_projections.reshape(148, input_projections.shape[1]*148)
        targets = target_data.reshape(148, input_projections.shape[1]*148)

        return torch.tensor(inputs, dtype=torch.float), torch.tensor(targets, dtype=torch.float)
    
    def initial_mse(self):
        return self.init_mse
    
    def data_status(self):
        if int(np.max(self.former_stage_data)) == 1:
            return True
        else:
            return False
