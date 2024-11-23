import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance
from sklearn.metrics import mutual_info_score

def random_sample_and_compare(file_path1, file_path2, sample_size=50, intensity_threshold=20):
    # 加载 .npz 数据
    data1 = np.load(file_path1)['arr_0']
    data2 = np.load(file_path2)['arr_0']
    
    # 确保数据维度正确
    assert data1.ndim == 4, "数据的维度必须是 (N, H, W, D)，当前为 {}".format(data1.shape)
    assert data2.ndim == 4, "数据的维度必须是 (N, H, W, D)，当前为 {}".format(data2.shape)
    num_samples1, height1, width1, depth1 = data1.shape
    num_samples2, height2, width2, depth2 = data2.shape

    # 随机选择样本
    indices1 = np.random.choice(num_samples1, sample_size, replace=False)
    sampled_data1 = data1[indices1]  # 选中的 50 个样本

    indices2 = np.random.choice(num_samples2, 6, replace=False)
    sampled_data2 = data2[indices2]
    
    # 合并所有投影图像为单个分布，并筛选像素强度
    projection1 = sampled_data1[0].reshape(-1)
    projection2 = sampled_data2[1].reshape(-1)
    
    # 筛选像素强度小于等于阈值的像素
    projection1 = projection1[projection1 <= intensity_threshold]
    projection2 = projection2[projection2 <= intensity_threshold]
    
    # 计算直方图（限制范围在 [0, intensity_threshold]）
    hist1 = np.histogram(projection1, bins=intensity_threshold + 1, range=(0, intensity_threshold), density=True)[0]
    hist2 = np.histogram(projection2, bins=intensity_threshold + 1, range=(0, intensity_threshold), density=True)[0]
    
    # 计算 Wasserstein 距离
    wd = wasserstein_distance(hist1, hist2)
    print(f"Wasserstein 距离: {wd}")
    
    # 计算互信息（Mutual Information）
    mi = mutual_info_score(hist1, hist2)
    print(f"互信息: {mi}")
    
    # 可视化直方图
    plt.figure(figsize=(10, 5))
    plt.plot(hist1, label='Sample 1', color='blue')
    plt.plot(hist2, label='Sample 2', color='orange')
    plt.title(f'Pixel Value Distributions (Intensity <= {intensity_threshold})')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Normalized Frequency')
    plt.legend()
    plt.grid()
    plt.show()

# 输入数据路径和键
file_path1 = "/home/mrb2/experiments/graduation_project/shared_data/exp/EXP_2024.10.1.3/train.npz"  # 替换为实际文件路径
file_path2 = '/home/mrb2/experiments/graduation_project/shared_data/projection/dl/projections_train_8-16.npz'

random_sample_and_compare(file_path1, file_path2)
