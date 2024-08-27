import torch.nn as nn
from torch import Tensor as Tensor
from typing import List
from network.encoder.pointnext import Stage, FeaturePropagation

class Encoder(nn.Module):
    """
    基于 PointNeXt 和 FPN (特征金字塔网络) 的骨干网络，用于特征提取
    """
    def __init__(self, args):
        super().__init__()

        # ======================== 输入参数 ========================
        self.args = args
        self.encoder_cfg = self.args.encoder

        self.in_channel = self.encoder_cfg.in_channel
        self.out_channel = self.encoder_cfg.out_channel
        self.downsample_layers = len(self.encoder_cfg.npoint)
        self.upsample_layers = self.encoder_cfg.upsample_layers
        width = self.encoder_cfg.width
        norm = self.encoder_cfg.get('norm', 'LN').lower()
        bias = self.encoder_cfg.get('bias', True)

        # ======================== 构建网络 ========================
        self.point_mlp0 = nn.Conv1d(in_channels=self.in_channel, out_channels=width, kernel_size=1)
        # 第一个 MLP，一个 1D 卷积层，用于把每个点的特征维度改为 width

        self.downsampler = nn.ModuleList()  # 列表 downsampler：下采样层
        self.upsampler = nn.ModuleList()  # 列表 upsampler：上采样层

        for i in range(self.downsample_layers):  # 创建指定个数下采样层（源于输入参数）
            self.downsampler.append(
                Stage(npoint=self.encoder_cfg.npoint[i],
                      radius_list=self.encoder_cfg.radius_list[i],
                      nsample_list=self.encoder_cfg.nsample_list[i],
                      in_channel=width,
                      sample=self.encoder_cfg.sample[i],
                      expansion=self.encoder_cfg['expansion'],
                      norm=norm,
                      bias=bias))
            width *= 2  # 每经过一次下采样，特征通道数翻倍，用于捕获更复杂的特征

        upsampler_in = width  # 上采样层的输入通道数初始化为下采样的最后一层输出通道数
        for i in range(self.upsample_layers):
            upsampler_out = max(self.out_channel, width // 2)  # 确保上采样的输出通道数不低于 out_channel
            self.upsampler.append(
                FeaturePropagation(in_channel=[upsampler_in, width // 2],
                                   mlp=[upsampler_out, upsampler_out],
                                   norm=norm,
                                   bias=bias))
            width = width // 2  # 每经过一次上采样，特征通道数减半
            upsampler_in = upsampler_out  # 更新上采样层的输入通道数

    def forward(self, points: Tensor, points_padding: Tensor) -> List[Tensor]:
        # 将输入点云的坐标和特征分开处理
        l0_coor, l0_fea, l0_padding = points[:, :3, :].clone(), points[:, :self.in_channel, :].clone(), points_padding
        l0_fea = self.point_mlp0(l0_fea)  # 通过 MLP 处理初始特征，生成宽度为 width 的特征图
        recorder = [[l0_coor, l0_fea, l0_padding]]  # 记录初始层的坐标、特征和填充掩码

        # 下采样过程
        for layer in self.downsampler:
            # 不断下采样特征图，并记录每一层的特征图
            new_coor, new_fea, new_padding = layer(*recorder[-1])
            recorder.append([new_coor, new_fea, new_padding])

        # 上采样过程
        for i, layer in enumerate(self.upsampler):
            # 从深层特征图到浅层特征图，进行逐步上采样，并整合特征
            points_coor1, points_fea1, points_padding1 = recorder[self.downsample_layers - i - 1]  # 获取对应的浅层特征
            points_coor2, points_fea2, points_padding2 = recorder[-1]  # 获取当前的深层特征
            new_points_fea1 = layer(points_coor1, points_coor2, points_fea1, points_fea2, points_padding2)
            recorder.append([points_coor1.clone(), new_points_fea1, points_padding1.clone()])

        return recorder[-1]  # 返回最终上采样后的特征，包括坐标、特征和填充掩码
