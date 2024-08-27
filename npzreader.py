import numpy as np

np.set_printoptions(threshold=np.inf)

# 加载npz文件
data = np.load("C:\\cza\\python_work\\自学作品\\深度学习\\自动驾驶\\DPM_AAAI\\KITTI-mini\\07\\0\\0.npz")

print(len(data['lidar_pcd'][0]))
print(len(data['lidar_seg'][0]))
