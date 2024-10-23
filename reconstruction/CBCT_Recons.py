import astra
import numpy as np
import matplotlib.pyplot as plt
import os
import data_handler
from tqdm import tqdm
from sklearn.metrics import mean_squared_error


class Projection:

    def __init__(self, params):
        # define parameter
        self.num_angles = params.n_angle
        self.vol_size = params.vol_size
        self.sod = params.d_s2o
        self.sdd = params.d_o2d
        self.if_test = params.if_test
        if not self.if_test:
            self.voxel_path_train = params.raw_voxel_path_train
            self.voxel_path_val = params.raw_voxel_path_val
        else:
            self.voxel_path = params.test_folder
        self.proj_result = params.proj_result_path
        self.if_image = params.if_image
        self.if_matrix = params.if_numpy

        # load data
        self.voxel_train = np.load(self.voxel_path_train)['arr_0']
        self.voxel_val = np.load(self.voxel_path_val)['arr_0']
        print(f'Loading voxel from: \ntrain-    {self.voxel_path_train} \nval-      {self.voxel_path_val}')
        self.volume_data_train = []
        self.volume_data_val = []

    def voxel_load(self):
        if not self.if_test:
            norm = 0
            for i in tqdm(range(len(self.voxel_train)), desc='Loading Train Data', leave=True):
                # Here we do padding for the raw voxels
                voxels_padded_train = np.pad(self.voxel_train[i], pad_width=10, mode='constant', constant_values=0)
                self.volume_data_train.append(voxels_padded_train)
                if np.max(self.voxel_train[i]) != 1:
                    norm += 1
            
            for i in tqdm(range(len(self.voxel_val)), desc='Loading Val Data', leave=True):
                # Here we do padding for the raw voxels
                voxels_padded_val = np.pad(self.voxel_val[i], pad_width=10, mode='constant', constant_values=0)
                self.volume_data_val.append(voxels_padded_val)
                if np.max(self.voxel_val[i]) != 1:
                    norm += 1
            if norm == 0:
                print('data normalized')
            else:
                print('data not normalized')
        else:
            for i in tqdm(range(len(self.voxel)), desc='Loading Test Data', leave=True):
                voxels_raw = self.voxel[i]
                voxels_padded = np.pad(voxels_raw, pad_width=10, mode='constant', constant_values=0)
                self.volume_data.append(voxels_padded)

    def get_xray(self, index, data_category):
        # set volume geom
        vol_size = self.volume_data_train[1].shape[0]
        vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)

        # create projection geom
        angles = np.linspace(0, np.pi, self.num_angles, False)
        proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, vol_size, vol_size, angles,
                                           self.sod, self.sdd)
        
        # create voxel data
        if data_category == 'train':
            vol_id = astra.data3d.create('-vol', vol_geom, data=self.volume_data_train[index])
        else:
            vol_id = astra.data3d.create('-vol', vol_geom, data=self.volume_data_val[index])

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

    def get_xray_1(self, index):
        vol_geom = astra.create_vol_geom(self.vol_size, )

    def save_projection(self, projection, result_individual_path, patient_index):
        unzipped_data_path = os.path.join(self.proj_result, 'unzipped_data')
        os.makedirs(unzipped_data_path, exist_ok=True)
        file_category_1 = 'image'
        if self.if_image:
            os.makedirs(os.path.join(unzipped_data_path, result_individual_path, file_category_1),
                        exist_ok=True)

        file_category_2 = 'matrix'
        if self.if_matrix:
            os.makedirs(os.path.join(unzipped_data_path, result_individual_path, file_category_2),
                        exist_ok=True)

        raw_data = []

        for index in range(len(projection[1])):
            if self.if_image:
                # 保存图像
                filename = f"image_{index}.png"
                file_path = os.path.join(unzipped_data_path, result_individual_path, file_category_1, filename)
                img = projection[:, index, :]
                plt.imsave(file_path, img, cmap='gray')

            if self.if_matrix:
                # 保存矩阵数据
                filename = f"matrix_{patient_index}.npz"
                file_category_2 = 'matrix'
                file_path = os.path.join(unzipped_data_path, result_individual_path, file_category_2, filename)
                matrix = projection[:, index, :]
                raw_data.append(matrix)
                np.savez(file_path, raw_data)

    def projection_process(self):
        self.voxel_load()
        projections_train = []
        os.makedirs(f'./result/projection/{self.num_angles}_angles', exist_ok=True)
        for i in tqdm(range(len(self.voxel_train)), desc='Doing Projection for Train Data', leave=True):
            result_individual_path = f'{i}'
            projections_train.append(self.get_xray(i, 'train'))
            self.save_projection(projections_train[i], result_individual_path, i)
        np.savez(os.path.join(self.proj_result, f'./{self.num_angles}_angles/Projection_train_data_{self.num_angles}_angles_padded.npz'), projections_train)
        print('train projection data saved')

        projections_val = []
        for i in tqdm(range(len(self.voxel_val)), desc='Doing Projection for Val Data', leave=True):
            result_individual_path = f'{i}'
            projections_val.append(self.get_xray(i, 'val'))
            self.save_projection(projections_val[i], result_individual_path, i)
        np.savez(os.path.join(self.proj_result, f'./{self.num_angles}_angles/Projection_val_data_{self.num_angles}_angles_padded.npz'), projections_val)
        print('val projection data saved')


def mse(target, ref):
    error = target - ref
    squared_error = error ** 2
    mse = np.mean(squared_error, dtype=np.float64)
    return mse


class Recons:

    def __init__(self, params):
        # parameters setting
        self.num_angles = params.n_angle
        self.sod = params.d_s2o
        self.sdd = params.d_o2d
        # self.data_path = os.path.join(params.projection_path, f'Projection_data_{self.num_angles}_angles_padded.npz')
        self.data_path_train = params.projection_path_train
        self.data_path_val = params.projection_path_val
        self.if_numpy = params.if_numpy
        self.algorithm = params.re_algorithm
        self.iterations = params.iterations
        self.if_test = params.if_test
        self.std = params.std
        self.if_noise = params.if_noise

    def load_projections(self):
        print('Present train projection path is: ', self.data_path_train)
        print('Present val projection path is: ', self.data_path_val)
        projections_train = np.load(self.data_path_train)['arr_0']
        projections_val = np.load(self.data_path_val)['arr_0']

        if not self.if_test:
            return projections_train, projections_val
        if self.if_test:
            return projections_train[0]

    def result_show(self, recon_voxels, label):
        result_folder = './result'
        if not os.path.exists(result_folder):
            os.makedirs(result_folder)

        plt.figure(figsize=(6, 6))
        plt.imshow(recon_voxels[:, :, int(128 / 2)], cmap='gray')
        plt.xlabel(label + '-' + f'{self.num_angles}')
        plt.show()

    def astra_reconstruct_3d(self, projections):
        # 调整投影数据的维度以符合 ASTRA 的要求
        # projections = np.transpose(projections, (0, 2, 1))
        vol_size = projections.shape[0]
        # 创建三维投影几何（锥束 CT 几何），包含自定义的 SOD 和 SDD
        proj_geom = astra.create_proj_geom('cone', 1.0, 1.0, vol_size, vol_size,
                                           np.linspace(0, np.pi, self.num_angles, False), self.sod, self.sdd)

        # 创建体积几何
        # vol_geom = astra.create_vol_geom(self.vol_size, self.vol_size, self.vol_size)
        vol_geom = astra.create_vol_geom(vol_size, vol_size, vol_size)

        # 创建用于重建的数据
        proj_id = astra.data3d.create('-proj3d', proj_geom, projections)

        # Create an initial value that follow the normal distribution
        mean = 0
        std = self.std
        if self.if_noise:
            initial_voxel = np.random.normal(mean, std, (148, 148, 148))
        else:
            initial_voxel = np.zeros([148, 148, 148])

        # 设置算法的配置
        if self.algorithm == 'FDK':
            cfg = astra.astra_dict('FDK_CUDA')
            cfg['ReconstructionDataId'] = astra.data3d.create('-vol', vol_geom, data=initial_voxel)
            cfg['ProjectionDataId'] = proj_id
        elif self.algorithm == 'SIRT':
            cfg = astra.astra_dict('SIRT3D_CUDA')
            cfg['ReconstructionDataId'] = astra.data3d.create('-vol', vol_geom, data=initial_voxel)
            cfg['ProjectionDataId'] = proj_id
            cfg['option'] = {}
            cfg['option']['MinConstraint'] = 0

        # 创建并运行算法
        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id, self.iterations)

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

    def test_recons(self):
        """
        This function is designed to test the proposed code improvement without reconstructing the whole dataset.
        :return: N/A
        """
        result_path = './test/result'
        slice_path = os.path.join(result_path, "slices")
        voxel_path = os.path.join(result_path, "voxel")
        os.makedirs(voxel_path, exist_ok=True)
        os.makedirs(slice_path, exist_ok=True)

        #  load just 1 projection for testing
        projections = self.load_projections()
        voxel_raw = np.load('./data/voxel/train_LIDC_128.npz')['data']
        voxel_truth = np.transpose(voxel_raw[0], (2, 1, 0))
        voxels = []
        # calculate the difference between ground truth and each iteration
        difference = []
        ssim_total = []
        for index in tqdm(range(2)):
            recons_voxel = self.astra_reconstruct_3d(projections)
            voxels.append(recons_voxel)

            self.result_show(voxels[index], 'Recons')
            self.result_show(voxel_truth, 'Truth')

            # calculate MSE/SSIM
            # recons_voxel_normalized = (voxels[index]*255).astype(np.int64)
            # truth_voxel_normalized = (voxel_truth*255).astype(np.int64)
            # mse = mean_squared_error(recons_voxel_normalized.flatten(), truth_voxel_normalized.flatten())
            mse = mean_squared_error(voxels[index].flatten(), voxel_truth.flatten())
            # ssim = ssim3D.ssim3d(voxels[index], voxel_truth)
            # save slices
            plt.imsave(os.path.join(slice_path, f"{self.num_angles}_std={self.std}_{index}.png"),
                       voxels[index][:, :, 64], cmap='gray')
            # save evaluations
            difference.append(mse)
            # ssim_total.append(ssim)
        np.savez(result_path, voxels)
        np.savetxt('dif.txt', difference, fmt='%f')

    def Recon_process(self):
        os.makedirs('./result/voxel', exist_ok=True)
        os.makedirs('./result/voxel/logs', exist_ok=True)
        projections_train, projections_val = self.load_projections()

        reconstruction_train = []
        reconstruction_val = []
        mse_total_individual = []

        voxel_raw_train = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Train_LIDC_128.npz')['arr_0']
        voxel_raw_val = np.load('/home/mrb2/experiments/graduation_project/shared_data/voxel/raw/Val_LIDC_128.npz')['arr_0']
        print('Shape of train projections:', np.shape(projections_train))
        print('Shape of val projections:', np.shape(projections_val))

        # for index in tqdm(range(len(projections_train)), desc='Reconstructing train data', leave=True):
        #     voxel_truth = voxel_raw_train[index]
        #     proj = projections_train[index]
        #     reconstructed_voxel = self.astra_reconstruct_3d(proj)
        #     reconstruction_train.append(reconstructed_voxel)

        #     # # evaluate
        #     # fig, axi = plt.subplots(1,2)
        #     # plt.subplots_adjust(wspace=0.4, hspace=0)
        #     # axi[0].imshow(reconstruction[index][:, 64, :], cmap='gray')
        #     # axi[0].set_title(f'Reconstructed Slice')
        #     # axi[1].imshow(voxel_raw[index][:, 64, :], cmap='gray')
        #     # axi[1].set_title(f'Raw Slice')
        #     # plt.savefig(f'./test/{index}.png', dpi=500)
        #     # plt.close()
            
        #     mse_individual = mse(reconstruction_train[index], voxel_truth)
        #     mse_total_individual.append(mse_individual)

        #     # save matrix for debug
        #     if self.if_numpy:
        #         np.save(f'./result/voxel/unzipped/{index}', reconstructed_voxel)
        # np.savetxt(f'./result/voxel/logs/total_mse_train_{self.algorithm}_{self.num_angles}_angles.txt', mse_total_individual, fmt='%f')
        # np.savez(f'./result/voxel/Train_Recons_{self.algorithm}_{self.num_angles}_angles.npz', reconstruction_train)
        # # self.result_show(reconstruction_train[0], 'recons')

        for index in tqdm(range(len(projections_val)), desc='Reconstructing val data', leave=True):
            voxel_truth = voxel_raw_val[index]
            proj = projections_val[index]
            reconstructed_voxel = self.astra_reconstruct_3d(proj)
            reconstruction_val.append(reconstructed_voxel)

            # # evaluate
            # fig, axi = plt.subplots(1,2)
            # plt.subplots_adjust(wspace=0.4, hspace=0)
            # axi[0].imshow(reconstruction[index][:, 64, :], cmap='gray')
            # axi[0].set_title(f'Reconstructed Slice')
            # axi[1].imshow(voxel_raw[index][:, 64, :], cmap='gray')
            # axi[1].set_title(f'Raw Slice')
            # plt.savefig(f'./test/{index}.png', dpi=500)
            # plt.close()
            
            mse_individual = mse(reconstruction_val[index], voxel_truth)
            mse_total_individual.append(mse_individual)

            # save matrix for debug
            if self.if_numpy:
                np.save(f'./result/voxel/unzipped/{index}', reconstructed_voxel)
        np.savetxt(f'./result/voxel/logs/total_mse_train_{self.algorithm}_{self.num_angles}_angles.txt', mse_total_individual, fmt='%f')
        np.savez(f'./result/voxel/Val_Recons_{self.algorithm}_{self.num_angles}_angles.npz', reconstruction_val)
        # self.result_show(reconstruction_train[0], 'recons')