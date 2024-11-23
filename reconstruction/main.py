import argparse
import CBCT_Recons
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    # geom parameters
    parser.add_argument('--operation', type=str, default='projection',
                        help='choose to do projection or reconstruction')
    parser.add_argument('--vol_size', type=int, default=148,
                        help='size of reconstruction volume')
    parser.add_argument('--n_angle', type=int, default=16,
                        help='number of projection angle')
    parser.add_argument('--d_s2o', type=int, default=1000,
                        help='Source-to-Origin Distance')
    parser.add_argument('--d_o2d', type=int, default=50,
                        help='Origin-to-Detector Distance')

    # recons parameters
    parser.add_argument('--projection_path_train', type=str, default='/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Train_Recons_SIRT_8_angles.npz',
                        help='the path to the projection data')
    parser.add_argument('--projection_path_val', type=str, default='/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Val_Recons_SIRT_8_angles.npz',
                        help='the path to the projection data')
    parser.add_argument('--re_algorithm', type=str, default='SIRT',
                        help='choose which algorithm to do reconstruction (FDK/SIRT)')
    parser.add_argument('--iterations', type=int, default=500, 
                        help='how many rounds you want to run the SIRT')
    parser.add_argument('--if_noise', type=bool, default=False,
                        help='if you want to add random noise to the initial value of recons. process')

    # projection parameters
    parser.add_argument('--raw_voxel_path_train', type=str, default='/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Train_Recons_SIRT_8_angles.npz',
                        help='the path to the voxel data')
    parser.add_argument('--raw_voxel_path_val', type=str, default='/home/mrb2/experiments/graduation_project/shared_data/voxel/recons/raw/Val_Recons_SIRT_8_angles.npz',
                        help='the path to the voxel data')
    parser.add_argument('--proj_result_path', type=str, default='./result/projection',
                        help='the path to the voxel data')

    # results parameter
    # Set the two options below as True will lead to performance decrease.
    parser.add_argument('--if_image', type=bool, default=False,
                        help='if you want to save the projection image to debug, choose True (not work when recons)')
    parser.add_argument('--if_numpy', type=bool, default=False,
                        help='if you want to save the projection matrix to debug, choose True')

    # test parameters
    parser.add_argument('--if_test', type=bool, default=False,
                        help='if you want to test the recons results by using projection function')
    parser.add_argument('--std', type=float, default=0.1,
                        help='set the standard division of the initial value')
    parser.add_argument('--test_folder', type=str, default='./test/data/Recons_SIRT_360_angles.npz',
                        help='folder path of test data')
    params = parser.parse_args()

    if params.operation == 'reconstruction':
        if params.if_test:
            test_recons = CBCT_Recons.Recons(params)
            test_recons.test_recons()
        else:
            reconstruction = CBCT_Recons.Recons(params)
            reconstruction.Recon_process()
    else:
        projection = CBCT_Recons.Projection(params)
        projection.projection_process()


if __name__ == '__main__':
    main()
