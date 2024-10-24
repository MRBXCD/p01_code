from utils import data_compose
import numpy as np
from tqdm import tqdm
from sklearn.metrics import mean_squared_error as rmse

def add_noise(data_raw, std, mean):
    data_shape = data_raw.shape
    noise = np.random.normal(mean,std,data_shape)
    noise_abs = np.abs(noise)
    data_noise = data_raw + noise_abs
    return data_noise

data_for_compare = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_16_angles_padded.npz')['arr_0']
data_raw = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_16_angles_padded.npz')['arr_0']

for item in tqdm(range(data_raw.shape[0]), desc='Add noise...'):
    indices = 0
    for index in range(int(data_raw.shape[2]/2)):
        data_raw[item,:,indices,:] = add_noise(data_raw[item,:,indices,:],0.5,0)
        indices += 2        
        
root_mse = rmse(data_for_compare.flatten(), data_raw.flatten())
print(root_mse)
