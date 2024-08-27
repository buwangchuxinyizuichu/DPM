import colorlog as logging  # 导入colorlog模块，并将其命名为logging，用于彩色日志记录。
logging.basicConfig(level=logging.INFO)  # 设置日志记录的基本配置，日志级别为INFO。
logger = logging.getLogger(__name__)  # 获取一个名为当前模块名的logger对象，用于日志记录。
logger.setLevel(logging.INFO)  # 设置logger对象的日志级别为INFO。

import random  # 导入random模块，用于生成随机数。
import pickle  # 导入pickle模块，用于对象的序列化和反序列化。
import numpy as np  # 导入numpy模块，并将其命名为np，用于数值计算。
import numpy.linalg as linalg  # 导入numpy的线性代数模块，并命名为linalg，用于矩阵运算。
import torch  # 导入PyTorch库，用于深度学习任务。
import torch.nn as nn  # 从PyTorch中导入神经网络模块，并命名为nn，用于构建神经网络模型。
import torch.nn.functional as F  # 导入PyTorch的函数式接口模块，命名为F，用于调用各种神经网络功能。
from torch import Tensor as Tensor  # 从PyTorch中导入Tensor类型，并保持名称不变，代表张量数据结构。
from typing import Tuple, Dict  # 导入Tuple和Dict类型提示，用于函数参数和返回值的类型注释。
from utils.pose import rt_global_to_relative  # 从utils.pose模块中导入rt_global_to_relative函数，用于转换全局姿态到相对姿态。

class DeepPointModelPipeline(nn.Module):  # 定义一个名为DeepPointModelPipeline的类，继承自PyTorch的nn.Module类，用于封装DeepPoint网络模型的训练流程。
    """
    DeepPoint网络模型的训练流程封装
    """

    def __init__(self, args, encoder: nn.Module, decoder: nn.Module, criterion: nn.Module):
        super().__init__()  # 调用父类nn.Module的构造函数，初始化基础组件。
        self.args = args  # 保存传入的参数args，这通常包含训练和模型的超参数。
        self.encoder = encoder  # 保存传入的编码器模型，用于特征提取。
        self.decoder = decoder  # 保存传入的解码器模型，用于特征配对和配准。
        self.criterion = criterion  # 保存传入的损失函数，用于计算训练过程中的损失。
        self._forward_method = None  # 初始化_forward_method为空，稍后会用于设置模型的前向传播方法。
        self.registration()  # 调用registration方法，可能用于设置模型的配准方法或初始化。
        self.refined_SE3_cache = dict()  # 初始化一个字典，用于缓存精化后的SE3变换矩阵。

    def forward(self, *args) -> Tuple[Tensor, dict]:
        return self._forward_method(*args)  # 调用之前设置的_forward_method方法进行前向传播，并返回结果。

    def _train_registration(self, pcd: Tensor, R: Tensor, T: Tensor, padding_mask: Tensor, calib: Tensor, info: dict)\
            -> Tuple[Tensor, dict]:  # 定义私有方法_train_registration，用于训练配准模块，返回损失和评价指标字典。
        """
        :param pcd: (S, 3, N)  # 点云数据，形状为(S, 3, N)，S表示帧数，N表示点的数量。
        :param R: (S, 3, 3)  # 旋转矩阵，形状为(S, 3, 3)。
        :param T: (S, 3, 1)  # 平移向量，形状为(S, 3, 1)。
        :param calib: (S, 4, 4)  # 校准矩阵，形状为(S, 4, 4)，表示相对于原始姿态的变换SE3。
        :param padding_mask: (S, 3)  # 填充掩码，用于指示点云的有效区域。
        :param info: 数据信息，指明地图帧数量、索引等信息。
        """
        coor, fea, mask = self.encoder(pcd, padding_mask)  # 使用编码器对输入点云和掩码进行编码，提取特征和掩码。
        S, _, N = coor.shape  # 获取coor的形状，S表示帧数，N表示点的数量。
        B = info['num_map']  # 从info字典中提取地图数量B。
        S = S // B  # 重新计算每个地图中的帧数S。
        coor = coor * self.args.slam_system.coor_scale  # 对坐标进行缩放，以匹配SLAM系统的尺度。
        pcd_index = np.asarray([dsf_index[2] for dsf_index in info['dsf_index']])  # 提取点云索引，长度为点云帧数。
        refined_SE3_file = info['refined_SE3_file']  # 获取精化后的SE3变换文件路径，长度为B。

        # 拆分成batch形式
        fea = fea.reshape(B, S, -1, N)  # 将特征重新reshape为batch形式。
        coor = coor.reshape(B, S, -1, N)  # 将坐标重新reshape为batch形式。
        R = R.reshape(B, S, 3, 3)  # 将旋转矩阵重新reshape为batch形式。
        T = T.reshape(B, S, 3, 1)  # 将平移向量重新reshape为batch形式。
        mask = mask.reshape(B, S, -1)  # 将掩码重新reshape为batch形式。
        pcd_index = pcd_index.reshape((B, S))  # 将点云索引重新reshape为batch形式。
        calib = calib.reshape(B, S, 4, 4)  # 将校准矩阵重新reshape为batch形式。

        # 50%概率s2m，50%概率m2m，但map帧数至多不超过map_size_max
        map_size_max = self.args.train.registration.map_size_max  # 从训练参数中获取最大地图大小。
        if S <= map_size_max:  # 如果帧数不超过最大地图大小
            if random.random() < 0.5:  # 以50%的概率选择1或随机拆分点
                S1 = 1
            else:
                S1 = random.randint(1, S - 1)  # 随机拆分帧数，确保地图帧数不超过限制。
        else:
            S1 = random.randint(S - map_size_max, map_size_max)  # 如果超过最大地图大小，确保拆分后的帧数不超过限制。
        S2 = S - S1  # 计算剩余帧数，形成另一部分。

        # 场景内所有帧被拆分为两张地图内
        src_coor, dst_coor = coor[:, :S1], coor[:, S1:]  # 将坐标数据拆分为源地图和目标地图。
        src_fea, dst_fea = fea[:, :S1], fea[:, S1:]  # 将特征数据拆分为源地图和目标地图。
        src_R, dst_R = R[:, :S1], R[:, S1:]  # 将旋转矩阵拆分为源地图和目标地图。
        src_T, dst_T = T[:, :S1], T[:, S1:]  # 将平移向量拆分为源地图和目标地图。
        src_mask, dst_mask = mask[:, :S1], mask[:, S1:]  # 将掩码拆分为源地图和目标地图。
        src_index, dst_index = pcd_index[:, :S1], pcd_index[:, S1:]  # 将点云索引拆分为源地图和目标地图。
        src_calib, dst_calib = calib[:, :S1], calib[:, S1:]  # 将校准矩阵拆分为源地图和目标地图。

        # 取各地图的第一帧为中心帧，其他帧与中心帧对齐后拉平为一帧 [B, S, *, N] -> [B, *, S*N]
        R, T = self._get_accurate_RT(src_index=src_index[:, 0], dst_index=dst_index[:, 0],
                                     src_R=src_R[:, 0], src_T=src_T[:, 0], src_calib=src_calib[:, 0],
                                     dst_R=dst_R[:, 0], dst_T=dst_T[:, 0], dst_calib=dst_calib[:, 0],
                                     refined_SE3_file=refined_SE3_file)  # 计算源地图与目标地图第一帧的准确旋转和平移矩阵。

        if S1 > 1:  # 如果源地图中有超过一帧
            map1_relative_R, map1_relative_T = \
                self._get_accurate_RT(src_index=src_index[:, 1:], dst_index=src_index[:, 0],
                                      src_R=src_R[:, 1:], src_T=src_T[:, 1:], src_calib=src_calib[:, 1:],
                                      dst_R=src_R[:, 0], dst_T=src_T[:, 0], dst_calib=src_calib[:, 0],
                                      refined_SE3_file=refined_SE3_file)  # 计算源地图其他帧相对于第一帧的相对旋转和平移矩阵。

            src_coor[:, 1:] = map1_relative_R @ src_coor[:, 1:] + map1_relative_T  # 应用相对旋转和平移，将其他帧对齐到第一帧。

        if S2 > 1:  # 如果目标地图中有超过一帧
            map2_relative_R, map2_relative_T = \
                self._get_accurate_RT(src_index=dst_index[:, 1:], dst_index=dst_index[:, 0],
                                      src_R=dst_R[:, 1:], src_T=dst_T[:, 1:], src_calib=dst_calib[:, 1:],
                                      dst_R=dst_R[:, 0], dst_T=dst_T[:, 0], dst_calib=dst_calib[:, 0],
                                      refined_SE3_file=refined_SE3_file,
                                      bridge_index=src_index[:, 0])  # 计算目标地图其他帧相对于第一帧的相对旋转和平移矩阵，可能需要依赖源地图的中心帧作为中继。

            dst_coor[:, 1:] = map2_relative_R @ dst_coor[:, 1:] + map2_relative_T  # 应用相对旋转和平移，将其他帧对齐到第一帧。

        src_coor = src_coor.transpose(1, 2).reshape(B, -1, S1 * N)  # 转置并拉平源地图的坐标，使其形状为(B, *, S1*N)。
        src_fea = src_fea.transpose(1, 2).reshape(B, -1, S1 * N)  # 转置并拉平源地图的特征，使其形状为(B, *, S1*N)。
        src_mask = src_mask.reshape(B, -1)  # 拉平源地图的掩码。
        dst_coor = dst_coor.transpose(1, 2).reshape(B, -1, S2 * N)  # 转置并拉平目标地图的坐标，使其形状为(B, *, S2*N)。
        dst_fea = dst_fea.transpose(1, 2).reshape(B, -1, S2 * N)  # 转置并拉平目标地图的特征，使其形状为(B, *, S2*N)。
        dst_mask = dst_mask.reshape(B, -1)  # 拉平目标地图的掩码。
        src_global_coor = R @ src_coor + T  # 应用全局旋转和平移矩阵，将源地图的坐标转换到全局坐标系中。
        dst_global_coor = dst_coor.clone()  # 克隆目标地图的坐标，作为全局坐标的参考。

        _DEBUG = False  # 定义一个调试标志，初始值为False。
        if _DEBUG:  # 如果调试模式开启
            from utils.visualization import show_pcd  # 导入点云可视化工具。
            for src, dst, src_global, dst_global in zip(src_coor, dst_coor, src_global_coor, dst_global_coor):
                show_pcd([src.T, dst.T], [[1, 0, 0], [0, 1, 0]],
                         window_name=f'src={src.shape[1]}, dst={dst.shape[1]} | local')  # 显示局部坐标系下的点云配准结果。
                show_pcd([src_global.T, dst_global.T], [[1, 0, 0], [0, 1, 0]],
                         window_name=f'src={src.shape[1]}, dst={dst.shape[1]} | gt')  # 显示全局坐标系下的点云配准结果。

        src_pairing_fea, dst_pairing_fea, src_coarse_pairing_fea, dst_coarse_pairing_fea, \
        src_offset_res, dst_offset_res = \
            self.decoder(
                torch.cat([src_fea, src_coor], dim=1),
                torch.cat([dst_fea, dst_coor], dim=1),
                src_padding_mask=src_mask,
                dst_padding_mask=dst_mask,
                gt_Rt=(R, T)
            )  # 使用解码器对源和目标特征及坐标进行配准，返回配对特征和粗配对特征，以及偏移残差。

        loss, top1_pairing_acc, loss_pairing, loss_coarse_pairing, loss_offset = \
            self.criterion(
                src_global_coor=src_global_coor, dst_global_coor=dst_global_coor,
                src_padding_mask=src_mask, dst_padding_mask=dst_mask,
                src_pairing_fea=src_pairing_fea, dst_pairing_fea=dst_pairing_fea,
                src_coarse_pairing_fea=src_coarse_pairing_fea, dst_coarse_pairing_fea=dst_coarse_pairing_fea,
                src_offset_res=src_offset_res, dst_offset_res=dst_offset_res
            )  # 使用定义的损失函数计算配准过程中的损失和评价指标。

        offset_err = (torch.norm(src_offset_res.detach(), p=2, dim=1).mean() +
                      torch.norm(dst_offset_res.detach(), p=2, dim=1).mean()).item() / 2  # 计算偏移残差的L2范数平均值作为误差度量。

        metric_dict = {  # 创建一个字典，存储各种损失和评价指标。
            'loss_regis': loss.item(),  # 配准损失。
            'loss_p': loss_pairing.item(),  # 配对损失。
            'loss_c': loss_coarse_pairing.item(),  # 粗配对损失。
            'loss_o': loss_offset.item(),  # 偏移损失。
            'top1_acc': top1_pairing_acc,  # 配对的Top-1准确率。
            'offset_err': offset_err  # 偏移误差。
        }
        return loss, metric_dict  # 返回损失值和指标字典。

    def _train_loop_detection(self, src_pcd: Tensor, src_R: Tensor, src_T: Tensor, src_mask: Tensor, src_calib: Tensor,
                              dst_pcd: Tensor, dst_R: Tensor, dst_T: Tensor, dst_mask: Tensor, dst_calib: Tensor
                              ) -> Tuple[Tensor, dict]:
        B = src_pcd.shape[0]  # 获取源点云的第一个维度大小B，通常表示batch大小。
        stacked_pcd = torch.cat([src_pcd, dst_pcd], dim=0)  # 将源点云和目标点云在第一个维度上拼接，形成(2B, C, N)的张量。
        stacked_mask = torch.cat([src_mask, dst_mask], dim=0)  # 将源掩码和目标掩码在第一个维度上拼接，形成(2B, N)的张量。

        # 特征提取
        coor, fea, mask = self.encoder(stacked_pcd, stacked_mask)  # 使用编码器对拼接后的点云和掩码进行编码，提取坐标、特征和掩码。
        coor = coor * self.args.slam_system.coor_scale  # 对坐标进行缩放，以匹配SLAM系统的尺度。
        src_coor, dst_coor = coor[:B], coor[B:]  # 将编码后的坐标按照batch大小B分割成源坐标和目标坐标。
        src_fea, dst_fea = fea[:B], fea[B:]  # 将编码后的特征按照batch大小B分割成源特征和目标特征。
        src_mask, dst_mask = mask[:B], mask[B:]  # 将编码后的掩码按照batch大小B分割成源掩码和目标掩码。

        # 预测回环
        src_descriptor = torch.cat([src_fea, src_coor], dim=1)  # 将源特征和源坐标在特征维度上拼接，形成源描述符。
        dst_descriptor = torch.cat([dst_fea, dst_coor], dim=1)  # 将目标特征和目标坐标在特征维度上拼接，形成目标描述符。
        loop_pred = self.decoder.loop_detection_forward(
            src_descriptor=src_descriptor, dst_descriptor=dst_descriptor,
            src_padding_mask=src_mask, dst_padding_mask=dst_mask,
        )  # 使用解码器进行回环检测，输出回环预测结果。

        # 基于配对距离判断是否回环
        dis = torch.norm((src_T - dst_T).squeeze(-1), p=2, dim=-1)  # 计算源点云和平移向量之间的欧氏距离。
        loop_gt = (dis <= self.args.train.loop_detection.distance).float()  # 如果距离小于设定的回环检测距离阈值，则认为发生回环。
        loop_loss = F.binary_cross_entropy(input=loop_pred, target=loop_gt)  # 使用二元交叉熵损失计算回环检测损失。

        # 指标，总体精度，召回率，假阳性率
        loop_pred_binary = loop_pred > 0.5  # 将回环预测结果二值化，阈值为0.5。
        loop_gt_binary = loop_gt.bool()  # 将回环标签转换为布尔类型。
        precision = (torch.sum(loop_pred_binary == loop_gt_binary) / loop_pred_binary.shape[0]).item()  # 计算总体精度。
        if loop_gt_binary.sum() > 0:  # 如果回环标签中有正例
            recall = torch.sum(loop_pred_binary[loop_gt_binary]) / loop_gt_binary.sum()  # 计算召回率。
            recall = recall.item()
        else:
            recall = 1.0  # 如果没有正例，则召回率设为1.0。
        negative_gt_mask = ~loop_gt_binary  # 计算负例掩码。
        if negative_gt_mask.sum() > 0:  # 如果负例掩码中有负例
            false_positive = torch.sum(loop_pred_binary[negative_gt_mask]) / negative_gt_mask.sum()  # 计算假阳性率。
            false_positive = false_positive.item()
        else:
            false_positive = 0.0  # 如果没有负例，则假阳性率设为0.0。

        metric_dict = {  # 创建一个字典，存储损失和评价指标。
            'loss_loop': loop_loss.item(),  # 回环检测损失。
            'loop_precision': precision,  # 回环检测精度。
            'loop_recall': recall,  # 回环检测召回率。
            'loop_false_positive': false_positive  # 回环检测假阳性率。
        }
        return loop_loss, metric_dict  # 返回回环检测损失和指标字典。

    def registration(self):
        self._forward_method = self._train_registration  # 设置模型的前向传播方法为_train_registration。
        # 训练配准时冻结回环检测部分
        for name, param in self.named_parameters():
            if 'loop' in name:  # 如果参数名称中包含'loop'，表示属于回环检测部分
                param.requires_grad = False  # 冻结该部分参数，即不更新其梯度。
            else:
                param.requires_grad = True  # 启用非回环检测部分的参数梯度更新。

    def loop_detection(self):
        self._forward_method = self._train_loop_detection  # 设置模型的前向传播方法为_train_loop_detection。
        # 训练回环检测时冻结其他网络
        for name, param in self.named_parameters():
            if 'loop' in name:  # 如果参数名称中包含'loop'，表示属于回环检测部分
                param.requires_grad = True  # 启用回环检测部分的参数梯度更新。
            else:
                param.requires_grad = False  # 冻结非回环检测部分的参数，即不更新其梯度。

    def _get_accurate_RT(self, src_index: np.ndarray, dst_index: np.ndarray, refined_SE3_file: str,
                         src_R: Tensor, src_T: Tensor, src_calib: Tensor,
                         dst_R: Tensor, dst_T: Tensor, dst_calib: Tensor, bridge_index=None,
                         src_pcd=None, dst_pcd=None) -> Tuple[Tensor, Tensor]:
        assert len(src_index) == len(dst_index) == len(refined_SE3_file) == len(src_R) == len(src_T) == len(src_calib) \
               == len(dst_R) == len(dst_T) == len(dst_calib)  # 检查输入的各个数组和张量的长度是否一致。
        B = len(src_index)  # 获取batch大小B，即源索引数组的长度。
        device = src_calib.device  # 获取校准张量的设备（CPU或GPU）。
        if bridge_index is None:
            bridge_index = [None] * B  # 如果未提供桥接索引，则初始化为None。
        else:
            assert len(bridge_index) == B  # 如果提供了桥接索引，检查其长度是否与batch大小一致。
        use_squeeze = src_index.ndim == 1 and dst_index.ndim == 1  # 判断源索引和目标索引是否是一维数组，以确定是否需要压缩维度。
        # 调整维度至 (B, S, *)
        src_index = src_index[:, np.newaxis] if src_index.ndim == 1 else src_index  # 如果源索引是一维数组，扩展其维度。
        dst_index = dst_index[:, np.newaxis] if dst_index.ndim == 1 else dst_index  # 如果目标索引是一维数组，扩展其维度。
        src_R = src_R.unsqueeze(1) if src_R.ndim == 3 else src_R  # 如果源旋转矩阵是3维张量，扩展其维度。
        src_T = src_T.unsqueeze(1) if src_T.ndim == 3 else src_T  # 如果源平移向量是3维张量，扩展其维度。
        src_calib = src_calib.unsqueeze(1) if src_calib.ndim == 3 else src_calib  # 如果源校准矩阵是3维张量，扩展其维度。
        dst_R = dst_R.unsqueeze(1) if dst_R.ndim == 3 else dst_R  # 如果目标旋转矩阵是3维张量，扩展其维度。
        dst_T = dst_T.unsqueeze(1) if dst_T.ndim == 3 else dst_T  # 如果目标平移向量是3维张量，扩展其维度。
        dst_calib = dst_calib.unsqueeze(1) if dst_calib.ndim == 3 else dst_calib  # 如果目标校准矩阵是3维张量，扩展其维度。

        S = max(src_index.shape[1], dst_index.shape[1])  # 计算源索引和目标索引的最大帧数。
        if src_index.shape[1] < S and src_index.shape[1] == 1:  # 如果源索引帧数小于最大帧数且为1
            src_index = src_index.repeat(repeats=S, axis=1)  # 重复源索引以匹配最大帧数。
            src_R = src_R.repeat(1, S, 1, 1)  # 重复源旋转矩阵以匹配最大帧数。
            src_T = src_T.repeat(1, S, 1, 1)  # 重复源平移向量以匹配最大帧数。
            src_calib = src_calib.repeat(1, S, 1, 1)  # 重复源校准矩阵以匹配最大帧数。

        if dst_index.shape[1] < S and dst_index.shape[1] == 1:  # 如果目标索引帧数小于最大帧数且为1
            dst_index = dst_index.repeat(repeats=S, axis=1)  # 重复目标索引以匹配最大帧数。
            dst_R = dst_R.repeat(1, S, 1, 1)  # 重复目标旋转矩阵以匹配最大帧数。
            dst_T = dst_T.repeat(1, S, 1, 1)  # 重复目标平移向量以匹配最大帧数。
            dst_calib = dst_calib.repeat(1, S, 1, 1)  # 重复目标校准矩阵以匹配最大帧数。

        # 逐场景校准
        _DEBUG, show_index = src_pcd is not None, 0  # 如果提供了源点云，则开启调试模式，并初始化显示索引。
        R_list, T_list = [], []  # 初始化旋转矩阵列表和平移向量列表，用于存储每个batch的校准结果。
        for b_src_i, b_src_R, b_src_T, b_src_c, b_dst_i, b_dst_R, b_dst_T, b_dst_c, file, bridge in \
                zip(src_index, src_R, src_T, src_calib, dst_index, dst_R, dst_T, dst_calib, refined_SE3_file,
                    bridge_index):
            SE3_dict = self._load_refined_SE3(file)  # 从文件中加载精化后的SE3变换矩阵。
            if SE3_dict is not None:  # 如果SE3字典不为空
                b_SE3 = []
                for i, (s, d, s_calib, d_calib) in enumerate(zip(b_src_i, b_dst_i, b_src_c, b_dst_c)):
                    try:
                        icp_SE3 = torch.from_numpy(get_SE3_from_dict(SE3_dict, s, d, bridge)).float().to(
                            device)  # 从SE3字典中获取ICP精化后的SE3变换矩阵。
                        current_SE3 = d_calib @ icp_SE3 @ s_calib.inverse()  # 计算当前的SE3变换矩阵。
                    except:
                        r, t = rt_global_to_relative(center_R=b_dst_R[i], center_T=b_dst_T[i],
                                                     other_R=b_src_R[i],
                                                     other_T=b_src_T[i])  # 如果获取SE3矩阵失败，使用全局姿态计算相对旋转和平移矩阵。
                        current_SE3 = torch.eye(4, dtype=torch.float32, device=device)  # 初始化4x4单位矩阵作为SE3变换矩阵。
                        current_SE3[:3, :3] = r  # 设置相对旋转矩阵。
                        current_SE3[:3, 3:] = t  # 设置相对平移向量。
                        import os
                        src_SE3 = torch.eye(4, dtype=torch.float32, device=device)  # 初始化源SE3矩阵。
                        dst_SE3 = torch.eye(4, dtype=torch.float32, device=device)  # 初始化目标SE3矩阵。
                        src_SE3[:3, :3] = b_src_R[i]  # 设置源旋转矩阵。
                        src_SE3[:3, 3:] = b_src_T[i]  # 设置源平移向量。
                        dst_SE3[:3, :3] = b_dst_R[i]  # 设置目标旋转矩阵。
                        dst_SE3[:3, 3:] = b_dst_T[i]  # 设置目标平移向量。
                        gt_ori_relative_SE3 = d_calib.inverse() @ dst_SE3.inverse() @ src_SE3 @ s_calib  # 计算真实的相对SE3矩阵。
                        dist = torch.norm(gt_ori_relative_SE3[:3, -1]).item()  # 计算平移向量的L2范数作为距离度量。
                        logger.warning(f'Found a pair without icp_SE3, in {os.path.dirname(file)}:({s}, {d}), '
                                       f'{dist=:.2f}, use gt instead')  # 记录警告信息，提示未找到ICP精化的SE3变换矩阵，使用真实值代替。
                    b_SE3.append(current_SE3)  # 将当前的SE3变换矩阵添加到列表中。
                b_SE3 = torch.stack(b_SE3, dim=0)  # 将SE3变换矩阵列表堆叠成张量。
                R_list.append(b_SE3[:, :3, :3])  # 提取旋转矩阵并添加到旋转矩阵列表中。
                T_list.append(b_SE3[:, :3, 3:])  # 提取平移向量并添加到平移向量列表中。
            else:  # 如果没有SE3字典，直接基于真实位姿计算
                R, T = rt_global_to_relative(center_R=b_dst_R, center_T=b_dst_T, other_R=b_src_R,
                                             other_T=b_src_T)  # 使用全局姿态计算相对旋转和平移矩阵。
                R_list.append(R)  # 添加计算得到的旋转矩阵到列表中。
                T_list.append(T)  # 添加计算得到的平移向量到列表中。

            if _DEBUG:  # 如果调试模式开启
                gt_Rs, gt_Ts = rt_global_to_relative(center_R=dst_R[show_index], center_T=dst_T[show_index],
                                                     other_R=src_R[show_index],
                                                     other_T=src_T[show_index])  # 计算真实的相对旋转和平移矩阵。
                from utils.visualization import show_pcd  # 导入点云可视化工具。
                src_xyz, dst_xyz = src_pcd[show_index], dst_pcd[show_index]  # 获取当前批次的源点云和目标点云。
                src_xyz = src_xyz[None].repeat(S, 1, 1) if src_xyz.ndim == 2 else src_xyz  # 如果源点云是二维张量，重复其维度。
                dst_xyz = dst_xyz[None].repeat(S, 1, 1) if dst_xyz.ndim == 2 else dst_xyz  # 如果目标点云是二维张量，重复其维度。
                for src, dst, gt_R, gt_T, icp_R, icp_T in zip(src_xyz, dst_xyz, gt_Rs, gt_Ts, R_list[-1],
                                                              T_list[-1]):  # 逐对显示点云配准结果。
                    src_gt = (gt_R @ src + gt_T).T  # 计算真实的配准结果。
                    src_icp = (icp_R @ src + icp_T).T  # 计算ICP精化后的配准结果。
                    src, dst = src.T, dst.T  # 转置源点云和目标点云。
                    show_pcd([src, dst], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'local | {S=} | {file}')  # 显示局部坐标系下的点云配准结果。
                    show_pcd([src_gt, dst], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'gt | {S=} | {file}')  # 显示真实坐标系下的点云配准结果。
                    show_pcd([src_icp, dst], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'icp | {S=} | {file}')  # 显示ICP精化后坐标系下的点云配准结果。
                if S > 1:  # 如果帧数大于1
                    src_xyz_gt = (gt_Rs @ src_xyz + gt_Ts).transpose(1, 2).reshape(-1, 3)  # 将真实的配准结果拉平成点云。
                    src_xyz_icp = (R_list[-1] @ src_xyz + T_list[-1]).transpose(1, 2).reshape(-1,
                                                                                              3)  # 将ICP精化后的配准结果拉平成点云。
                    dst_xyz_map = dst_xyz.transpose(1, 2).reshape(-1, 3)  # 将目标点云拉平成地图。
                    show_pcd([src_xyz_gt, dst_xyz_map], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'gt map | {S=} | {file}')  # 显示真实地图坐标系下的配准结果。
                    show_pcd([src_xyz_icp, dst_xyz_map], [[1, 0, 0], [0, 1, 0]],
                             window_name=f'icp map | {S=} | {file}')  # 显示ICP精化后的地图坐标系下的配准结果。
                show_index += 1  # 增加显示索引。

        R, T = torch.stack(R_list, dim=0), torch.stack(T_list, dim=0)  # 将旋转矩阵列表和平移向量列表堆叠成张量。
        if use_squeeze:  # 如果使用压缩
            R, T = R.squeeze(1), T.squeeze(1)  # 压缩旋转矩阵和平移向量的维度。
        return R, T  # 返回旋转矩阵和平移向量。

    def _load_refined_SE3(self, file):
        if file not in self.refined_SE3_cache.keys():  # 如果文件未被缓存
            if file != '':
                with open(file, 'rb') as f:
                    refined_SE3: Dict[Tuple[int, int], np.ndarray] = pickle.load(f)  # 从文件中加载精化后的SE3变换矩阵。
            else:
                refined_SE3 = None  # 如果文件为空，设置refined_SE3为None。
            self.refined_SE3_cache[file] = refined_SE3  # 将加载的SE3变换矩阵缓存到字典中。
        return self.refined_SE3_cache[file]  # 返回缓存的SE3变换矩阵。


def get_SE3_from_dict(SE3_dict: Dict[Tuple[int, int], np.ndarray], s: int, d: int, bridge=None) -> np.ndarray:
    if s == d:  # 如果源点和目标点相同，则变换矩阵为单位矩阵。
        SE3 = np.eye(4)
    elif s < d:  # 如果源点的索引小于目标点的索引
        SE3 = SE3_dict.get((s, d), None)  # 从字典中获取s到d的SE3变换矩阵。
        if SE3 is not None:
            SE3 = linalg.inv(SE3)  # 如果找到变换矩阵，则计算其逆矩阵。
    else:
        SE3 = SE3_dict.get((d, s), None)  # 如果源点的索引大于目标点的索引，则直接获取d到s的SE3变换矩阵。

    if SE3 is None:  # 如果SE3矩阵为None，表示字典中没有找到直接的变换矩阵。
        SE3_s2b = get_SE3_from_dict(SE3_dict, s, bridge, None)  # 递归调用自己，从源点到桥接点的SE3变换矩阵。
        SE3_b2d = get_SE3_from_dict(SE3_dict, bridge, d, None)  # 递归调用自己，从桥接点到目标点的SE3变换矩阵。
        SE3 = SE3_b2d @ SE3_s2b  # 计算最终的变换矩阵为桥接点变换矩阵的组合。

    return SE3  # 返回计算得到的SE3变换矩阵。

