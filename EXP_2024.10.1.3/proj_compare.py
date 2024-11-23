import numpy as np
from sklearn.metrics import root_mean_squared_error as rmse
import os

projection_raw = np.load("/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_val_data_16_angles_padded.npz")['arr_0']
projection_8to16_backproj = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/projections/Projection_val_data_16_angles_padded.npz')['arr_0']
projection_8to16_dl = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/dl/projections_val_8-16.npz')['arr_0']

raw_vs_backproj = rmse(projection_raw.flatten(), projection_8to16_backproj.flatten())
raw_vs_dl = rmse(projection_raw.flatten(), projection_8to16_dl.flatten())

print(raw_vs_backproj, raw_vs_dl)




