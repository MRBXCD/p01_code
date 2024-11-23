import astra
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch

def padding(voxel, num_pad):
    """
    A function to pad the voxel
    :param voxel: a dataset contains all patients' 3D voxel
    :param num_pad: number of pixels that will be padded near the voxel
    :return: a dataset contains all padded voxels
    """
    
    pad_size = (num_pad, num_pad, num_pad, num_pad, num_pad, num_pad)
    if len(voxel) == 1:
        if_list = 0
        voxel = torch.squeeze(voxel)
        voxels_padded = F.pad(voxel, pad_size, mode='constant', value=0)
        return voxels_padded, if_list
    else:
        if_list = 1
        voxel = torch.squeeze(voxel)
        results = []
        for index in range(len(voxel)):
            voxels_padded = F.pad(voxel[index], pad_size, mode='constant', value=0)
            #voxels_padded = (voxels_padded / 255).astype(np.float32)
            results.append(voxels_padded)
        # exp = results[0]
        # plt.figure(figsize=(6, 6))
        # plt.imshow(exp[:, :, 74], cmap='gray')
        # plt.show()
        return results, if_list

def projection(voxel, num_angles, sod, sdd):
    # projection process
    voxel = voxel.cpu().detach().numpy()
    voxel_size = voxel.shape[1]
    #print(f'size of voxel: {voxel_size}')
    vol_geom = astra.create_vol_geom(voxel_size, voxel_size, voxel_size)
    angles = np.linspace(0, 2 * np.pi, num_angles, False)
    proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, voxel_size, voxel_size, angles,
                                       sod, sdd)
    vol_id = astra.data3d.create('-vol', vol_geom, data=voxel)
    proj_id = astra.create_sino3d_gpu(vol_id, proj_geom, vol_geom)
    projection_data = astra.data3d.get(proj_id[0])
    astra.data3d.delete(proj_id[0])
    astra.data3d.delete(vol_id)
    return projection_data