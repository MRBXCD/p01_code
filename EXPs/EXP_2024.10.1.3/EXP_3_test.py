import numpy as np

def test():
   data1 = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/train.npz')['arr_0']
   data2 = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_16_angles_padded.npz')['arr_0']

   assert np.array_equal(data1[1,:,1,:], data2[1,:,1,:])
   assert not np.array_equal(data1[1,:,0,:], data2[1,:,0,:])