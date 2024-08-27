import numpy as np
from utils.visualization import show_pcd



colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 0], [0, 0.5, 0.5], [0.5, 0, 0.5], [0.5, 0.5, 0]]

npz_file = r"C:\Users\HUAWEI\Desktop\npz\0.npz"

with np.load(npz_file, allow_pickle=True) as npz:  # 使用numpy的load函数加载npz文件，并允许文件内对象被pickle序列化
    xyz = npz['lidar_pcd']  # 从npz文件中读取点云数据xyz，对应键为'lidar_pcd'，其形状为(N, 3)，数据类型为f32（32位浮点数）
    label = npz['labels']


seg_label_set = np.unique(label)

pcd_per_cls = []
color_per_cls = []

for cls in seg_label_set:
    cur_pcd = xyz[label == cls]
    cur_color = colors[cls % len(colors)]
    pcd_per_cls.append(cur_pcd)
    color_per_cls.append(cur_color)

show_pcd(pcd_per_cls, color_per_cls)

