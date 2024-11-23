import numpy as np
import matplotlib.pyplot as plt

data1 = np.load('./model_output.npz')['arr_0']
data1 = data1.squeeze(1)
data2 = np.load('./model_input.npz')['arr_0']
data2 = data2.squeeze(1)
print(np.shape(data1))
print(np.shape(data2))

fig, axs = plt.subplots(2, 1)  # 调整figsize和dpi以满足需求

        # 调整子图之间的间距
plt.subplots_adjust(wspace=0.1, hspace=0.2)  # 减小宽度间距和高度间距

        
        # 对于GPU上的数据，确保先将其移动到CPU，并转换为numpy数组
input = data1[0, :, :]
target = data2[0, :, :]

        # 展示输入图像的切片
axs[0].imshow(input, cmap='gray')
axs[0].set_title(f'Input #{1}')
axs[0].axis('off')  # 不显示坐标轴

        # 展示目标图像的切片
axs[1].imshow(target, cmap='gray')
axs[1].set_title(f'Target #{1}')
axs[1].axis('off')  # 不显示坐标轴

        # 给整个大图设置标题
fig.suptitle(111)
        
        # 调整整个大图的布局，确保标题和子图之间的间距足够
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # 保存图像，调整DPI以提高保存图像的清晰度
plt.savefig('111.png', dpi=500)

plt.imsave('input.png',data1[0], cmap='gray')
plt.imsave('target.png',data2[0], cmap='gray')