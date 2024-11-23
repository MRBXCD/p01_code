import numpy as np

def replace_proj(data, data_backproj, flag):
    indices = [i for i in range(0, data.shape[2]-1, 2)]
    print(indices)
    for i in range(data.shape[0]):
        data_backproj[i,:,indices,:] = data[i,:,indices,:]
    
    # np.savez(f'./{flag}.npz', data_backproj)




proj_train = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_16_angles_padded.npz')['arr_0']
proj_val = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_val_data_16_angles_padded.npz')['arr_0']

proj_train_backprojected = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/projections/Projection_train_data_16_angles_padded.npz')['arr_0']
proj_val_backprojected = np.load('/home/mrb2/experiments/graduation_project/p01_code/EXP_2024.10.1.3/projections/Projection_val_data_16_angles_padded.npz')['arr_0']

replace_proj(proj_train, proj_train_backprojected, 'train')
replace_proj(proj_val, proj_val_backprojected, 'val')
replace_proj(proj_val, proj_val_backprojected, 'train')


