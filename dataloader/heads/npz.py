import os  # 导入os模块，用于文件路径操作
import numpy as np  # 导入numpy库并命名为np，用于数值计算和数组操作
from dataloader.heads.basic import PointCloudReader  # 从dataloader.heads.basic模块中导入PointCloudReader类，这是一个点云读取器的基类


class NPZReader(PointCloudReader):  # 定义一个继承自PointCloudReader的类NPZReader，用于读取npz格式的点云数据
    optional_type = ['npz']  # 定义一个类变量optional_type，指定可以处理的文件类型为'npz'

    def __init__(self):  # 定义初始化方法
        super().__init__()  # 调用父类PointCloudReader的初始化方法

    '''
    def _load_pcd(self, file_path):  # 定义一个私有方法_load_pcd，用于从文件中加载点云数据
        """从源文件读取"""  # 方法的文档字符串，简单描述方法的功能
        file_type = os.path.splitext(file_path)[-1][1:]  # 获取文件的扩展名，os.path.splitext返回一个包含文件路径和扩展名的元组，[-1][1:]用于提取扩展名并去掉前面的'.'
        assert file_type in self.optional_type, f'Only type of the file in {self.optional_type} is optional, ' \
                                                f'not \'{file_type}\''  # 断言文件类型在optional_type列表中，如果不在，抛出带有错误信息的异常
        with np.load(file_path, allow_pickle=True) as npz:  # 使用numpy的load函数加载npz文件，并允许文件内对象被pickle序列化
            npz_keys = npz.files  # 获取npz文件中的所有键名
            assert 'lidar_pcd' in npz_keys, 'pcd file must contains \'lidar_pcd\''  # 断言'lidar_pcd'键必须存在，否则抛出异常
            xyz = npz['lidar_pcd']  # 从npz文件中读取点云数据xyz，对应键为'lidar_pcd'，其形状为(N, 3)，数据类型为f32（32位浮点数）
            rotation = npz['ego_rotation'] if 'ego_rotation' in npz_keys else None  # 从npz文件中读取自车旋转矩阵rotation，键为'ego_rotation'，如果键不存在则为None，形状为(3, 3)，f32
            translation = npz['ego_translation'] if 'ego_translation' in npz_keys else None  # 从npz文件中读取自车平移向量translation，键为'ego_translation'，如果键不存在则为None，形状为(3, 1)，f32
            norm = npz['lidar_norm'] if 'lidar_norm' in npz_keys else None  # 从npz文件中读取点云法线数据norm，键为'lidar_norm'，如果键不存在则为None，形状为(N, 3)，f32
            label = npz['lidar_seg'] if 'lidar_seg' in npz_keys else None  # 从npz文件中读取点云分割标签label，键为'lidar_seg'，如果键不存在则为None，形状为(N, 3)，f32
            image = npz['image'] if 'image' in npz_keys else None  # 从npz文件中读取图像数据image，键为'image'，如果键不存在则为None
            uvd = npz['lidar_proj'] if 'lidar_proj' in npz_keys else None  # 从npz文件中读取点云投影数据uvd，键为'lidar_proj'，如果键不存在则为None

        return xyz, rotation, translation, norm, label, image, uvd  # 返回从npz文件中读取的所有数据
    '''
