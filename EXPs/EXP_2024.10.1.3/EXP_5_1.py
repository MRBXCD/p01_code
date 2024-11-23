import numpy as np
import astra
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import root_mean_squared_error as rmse

def rep(data, data_backproj):
    indices = [i for i in range(0, data.shape[1]-1, 2)]
    #print(indices)
    for i in range(data.shape[0]):
        data_backproj[:,indices,:] = data[:,indices,:]

def projection(voxel, num_angles):
    # set volume geom
    vol_size = voxel.shape[0]
    vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)

    # create projection geom
    angles = np.linspace(0, np.pi, num_angles, False)
    proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, vol_size, vol_size, angles, 1000, 50)
    vol_id = astra.data3d.create('-vol', vol_geom, data=voxel)

    # create projection data
    proj_id = astra.create_sino3d_gpu(vol_id, proj_geom, vol_geom)

    # get projection
    projection_data = astra.data3d.get(proj_id[0])
    # for index in range(self.num_angles):
    #     projection_data[:,index,:] = projection_data[:,index,:] / np.max(projection_data[:,index,:])
    # clean src
    astra.data3d.delete(vol_id)
    astra.data3d.delete(proj_id[0])

    return projection_data

def astra_reconstruct_3d(projections, iterations=500):
    # 调整投影数据的维度以符合 ASTRA 的要求
    # projections = np.transpose(projections, (0, 2, 1))
    vol_size = projections.shape[0]
    num_angles = projections.shape[1]
    # 创建三维投影几何（锥束 CT 几何），包含自定义的 SOD 和 SDD
    proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, vol_size, vol_size,
                                        np.linspace(0, np.pi, num_angles, False), 1000, 50)

    # 创建体积几何
    # vol_geom = astra.create_vol_geom(self.vol_size, self.vol_size, self.vol_size)
    vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)

    # 创建用于重建的数据
    proj_id = astra.data3d.create('-proj3d', proj_geom, projections)

    initial_voxel = np.zeros([148, 148, 148])
    cfg = astra.astra_dict('SIRT3D_CUDA')
    cfg['ReconstructionDataId'] = astra.data3d.create('-vol', vol_geom, data=initial_voxel)
    cfg['ProjectionDataId'] = proj_id
    cfg['option'] = {}
    cfg['option']['MinConstraint'] = 0

    # 创建并运行算法
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id, iterations)

    # 获取重建结果
    recon = astra.data3d.get(cfg['ReconstructionDataId'])

    padding_thickness = 10

    original_array = recon[
                        padding_thickness:-padding_thickness,
                        padding_thickness:-padding_thickness,
                        padding_thickness:-padding_thickness
                        ]
    max_value = np.max(original_array)
    normalized_voxel = original_array / max_value
    # 清理
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(proj_id)
    astra.data3d.delete(cfg['ReconstructionDataId'])

    return normalized_voxel

# set exp sample index to reduce the experiment time consumption
train_index = [65, 332, 759, 480, 321, 716, 734, 427, 686, 52, 27, 732, 328, 713, 417, 453, 724, 605, 93, 552, 666, 414, 387, 335, 
               366, 456, 159, 198, 407, 538, 552, 575, 241, 696, 299, 134, 221, 16, 608, 172, 312, 763, 109, 493, 179, 345, 410, 
               668, 613, 611]
val_index = [45, 49, 41, 33, 55, 38]

# load voxels
train_voxels = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Train_LIDC_128.npz')['arr_0']
val_voxels = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Val_LIDC_128.npz')['arr_0']

# load projection images
proj_train = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_train_data_16_angles_padded.npz')['arr_0']
proj_train = proj_train[train_index]

proj_val = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/raw/Projection_val_data_16_angles_padded.npz')['arr_0']
proj_val = proj_val[val_index] 

proj_train_backprojected = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/backproj/Projection_train_data_16_angles_padded.npz')['arr_0']
proj_train_backpj = proj_train_backprojected[train_index]

proj_val_backprojected = np.load('/home/mrb2/experiments/graduation_project/shared_data/projection/backproj/Projection_val_data_16_angles_padded.npz')['arr_0']
proj_val_backpj = proj_val_backprojected[val_index]

iter_base = 500
rmse_train = np.zeros((10,11))
rmse_val = np.zeros((10,11))

for item in range(len(proj_train_backpj)):
    rep(proj_train[item], proj_train_backpj[item])
for item in range(len(proj_val_backpj)):
    rep(proj_val[item], proj_val_backpj[item])


for iters in tqdm(range(11)):
    for index in range(10):
        itera = iter_base + iters
        train_voxel = astra_reconstruct_3d(proj_train_backpj[index], iterations=itera)
        rmse_train[index][iters] = round(rmse(train_voxel.flatten(), train_voxels[index].flatten()),7)
      
    for index in range(6):
        itera = iter_base + iters 
        val_voxel = astra_reconstruct_3d(proj_val_backpj[index], iterations=itera)
        rmse_val[index][iters] = round(rmse(val_voxel.flatten(), val_voxels[index].flatten()),7)
        
rmses_train = np.sum(rmse_train, axis=0)
rmses_train = rmses_train / 10
np.savetxt('./EXP_5.1_train_rmse', rmses_train)
rmses_val = np.sum(rmse_val, axis=0)
rmses_val = rmses_val / 6
np.savetxt('./EXP_5.1_val_rmse', rmses_val)

x = [500, 501, 502, 503, 504, 505, 506, 507, 508, 509, 510] 
plt.figure()
plt.scatter(x, rmses_train)
plt.plot(x, rmses_train)
plt.savefig('/home/mrb2/experiments/graduation_project/p01_code/EXPs/EXP_2024.10.1.3/result/EXP_5.1_rmse_train.png')
plt.close()

plt.figure()
plt.scatter(x, rmses_val)
plt.plot(x, rmses_val)
plt.savefig('/home/mrb2/experiments/graduation_project/p01_code/EXPs/EXP_2024.10.1.3/result/EXP_5.1_rmse_val.png')