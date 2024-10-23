import numpy as np
import torch
import os

def main():
    folder = '/root/autodl-tmp/Reconstruction/data/voxel'
    for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        voxel_normalize(filepath)

def voxel_squeeze(filename):
    data = np.load(filename)['arr_0']
    print(data.shape)
    if data.ndim == 4:
        print(f'{filename} already squeezed')
    else:
        squeezed_data = np.squeeze(data, axis=None)
        np.savez(filename, squeezed_data)
        print(f'{filename} squeeze complete')

def voxel_transpose(data):
    transposed_data = []
    for index in range(len(data)):
        transposed_data.append(np.transpose(data[index], (2, 1, 0)))
    np.savez('data/Val_LIDC_128_transposed.npz', transposed_data)
    print('end')

def voxel_normalize(filename):
    data = np.load(filename)['arr_0']
    num_norm = 0
    data_save = np.zeros((len(data),128,128,128))

    for index in range(len(data)):
        voxel = data[index]
        if np.max(voxel) != 1:
            data_save[index] = voxel / np.max(voxel)
            num_norm += 1
        else:
            data_save[index] = voxel 
    np.savez(f'{filename}_1.npz', data_save)
    print(f'{filename} - {num_norm} normalize complete')

if __name__ == '__main__':
    main()