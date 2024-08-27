import os
import sys
import yaml
from parameters import *

args = parser.parse_args()  # 解析命令行参数并保存到args对象中
assert args.use_ddp == False  # 确保分布式数据并行（DDP）未被启用
if args.use_cuda:  # 如果使用CUDA（GPU计算）
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index  # 设置环境变量，指定使用的GPU索引
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import colorlog as logging  # 用于生成带颜色的日志输出

logging.basicConfig(level=logging.INFO)  # 设置基本配置，日志级别为INFO
logger = logging.getLogger(__name__)  # 创建一个日志记录器实例，名称为当前模块的名称
logger.setLevel(logging.INFO)  # 设置日志记录器的日志级别为INFO

import warnings

warnings.filterwarnings("ignore")  # 忽略所有的警告信息

import torch

torch.manual_seed(42)  # 设置随机种子为42，以保证实验的可重复性
torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程数据共享策略为'file_system'

from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from system.core import SlamSystem
from dataloader.body import BasicAgent
from dataloader.transforms import PointCloudTransforms
from network.encoder.encoder import Encoder
from network.decoder.decoder import Decoder


def main():
    # 加载YAML配置文件并准备平台
    global args
    if not os.path.exists(args.yaml_file):  # 检查配置文件是否存在
        raise FileNotFoundError(f'yaml_file is not found: {args.yaml_file}')
    logger.info(f'Loading config from \'{args.yaml_file}\'...')
    with open(args.yaml_file, 'r', encoding='utf-8') as f:  # 打开YAML配置文件
        cfg = yaml.load(f, yaml.FullLoader)  # 使用YAML加载器加载配置文件内容
    args = update_args(args, cfg)  # 使用配置文件内容更新命令行参数
    if not args.thread_safety:  # 如果未启用线程安全
        torch.multiprocessing.set_start_method('spawn')  # 设置多进程启动方式为'spawn'
        logger.warning(f'The start method of torch.multiprocessing has been set to \'spawn\'')
    if args.use_cuda and torch.cuda.is_available():  # 如果使用CUDA且CUDA可用
        args.device = torch.device('cuda')  # 设置设备为CUDA
        gpus = list(range(torch.cuda.device_count()))  # 获取所有可用GPU的索引列表
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))  # 设置当前使用的GPU为第一个可用的GPU
    else:
        args.device = torch.device('cpu')

    # 初始化模型并加载权重
    logger.info('Preparing model...')
    encoder = Encoder(args=args)
    decoder = Decoder(args=args)
    if (os.path.exists(args.weight) == False):
        logger.warning(f'weight file not exists: {args.weight}, model will be random initialized.')
    else:  # 如果权重文件存在
        logger.info(f'Load weight from \'{args.weight}\'')
        weights = torch.load(args.weight, map_location='cpu')  # 加载权重文件，将其映射到CPU
        encoder.load_state_dict(weights['encoder'], strict=True)  # 加载编码器的权重
        decoder.load_state_dict(weights['decoder'], strict=False)  # 加载解码器的权重（非严格模式）
        logger.info(f'Initialization completed, device = \'{args.device}\'')  # 记录日志，提示初始化完成并显示使用的设备
    if args.half:  # 如果使用半精度浮点数
        logger.critical('USE FP16')  # 记录严重日志，提示正在使用FP16
        encoder = encoder.half()  # 将编码器转换为半精度
        decoder = decoder.half()  # 将解码器转换为半精度

    # Init data-transform  # 初始化数据转换
    logger.info('Preparing data...')  # 记录日志，显示正在准备数据
    transforms = PointCloudTransforms(args=args, mode='infer')  # 初始化点云数据的转换器，模式为推理模式

    # For each sequence...  # 对每个数据序列进行处理
    for i, data_root in enumerate(args.infer_src):  # 遍历推理数据源
        if (isinstance(data_root, str)):  # 如果数据源是字符串
            data_root = [data_root]  # 将字符串转换为列表
        str_list = list(filter(lambda dir: os.path.exists(dir), data_root))  # 过滤出存在的目录
        if (len(str_list) == 0):  # 如果不存在任何有效的目录
            logger.error(f"dir in source '{data_root}' ({i}) not found, SKIP!")  # 记录错误日志，提示未找到目录并跳过当前序列
            continue  # 跳过当前循环
        else:  # 如果存在有效目录
            logger.info(f"loading data '{data_root}' ({i})")  # 记录日志，显示正在加载的数据源路径
            data_root = str_list[0]  # 使用第一个有效目录作为数据根目录
        dataset = BasicAgent(root=data_root, reader='auto')  # 初始化数据集代理对象
        dataset.set_independent(data_transforms=transforms)  # 设置数据集的独立数据转换器

        # Create result dir and save yaml  # 创建结果目录并保存配置文件
        save_root = os.path.join(args.infer_tgt, f'Seq{i:02}')  # 生成结果保存路径
        os.makedirs(save_root, exist_ok=True)  # 创建结果目录，如果目录不存在则创建
        with open(os.path.join(save_root, 'settings.yaml'), 'w+', encoding='utf-8') as arg_file:  # 打开一个YAML文件用于写入配置
            args_dict = sorted(args._get_kwargs())  # 将args的键值对转化为排序后的列表
            for k, v in args_dict:  # 遍历键值对
                arg_file.write(f'{k}: {v}\n')  # 将每个键值对写入文件

        # Prepare dataloader  # 准备数据加载器
        dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        # 初始化数据加载器，设置批量大小为1，不打乱顺序，使用指定数量的工作线程，并启用内存锁定
        infer_loop = tqdm(dataloader, total=len(dataloader), dynamic_ncols=True, desc=f'{"dataloader":<12s}')
        # 使用tqdm创建一个带有进度条的数据加载循环

        # Init SLAM system  # 初始化SLAM系统
        slam_system = SlamSystem(args=args, dpm_encoder=encoder, dpm_decoder=decoder, system_id=0, logger_dir=save_root)
        # 初始化SLAM系统，传入编码器、解码器、系统ID和日志目录
        if args.multi_thread:  # 如果启用了多线程
            # Feed data! (Multi-Thread / MT)  # 以多线程方式输入数据
            slam_system.MT_Init()  # 初始化多线程SLAM系统
            for frame_id, data in enumerate(infer_loop):  # 遍历推理数据
                slam_system.MT_Step(data)  # 多线程处理数据
            slam_system.MT_Done()  # 完成多线程处理
            slam_system.MT_Wait()  # 等待所有线程完成

        else:  # 如果使用单线程
            # Feed data! (Single Thread)  # 以单线程方式输入数据
            for frame_id, data in enumerate(infer_loop):  # 遍历推理数据
                if args.half:  # 如果使用半精度浮点数
                    data = [i.half() if isinstance(i, torch.Tensor) and i.dtype == torch.float else i for i in data]
                    # 将数据中的浮点数转换为半精度
                code = slam_system.step(data)  # 处理当前数据帧
                infer_loop.set_description_str(f"infer: [{code}]" + ", ".join([f"{name}:{time[0]:.3f}s" for name, time in slam_system.result_logger.log_time(window=50).items()]))
                # 更新进度条描述，显示推理状态和各操作的时间
        slam_system.result_logger.save_trajectory('trajectory')  # 保存SLAM系统的轨迹
        # slam_system.result_logger.save_posegraph('trajectory')  # （注释掉）保存SLAM系统的姿态图
        slam_system.result_logger.draw_trajectory('trajectory', draft=False)  # 绘制并保存轨迹图
        slam_system.result_logger.save_map('trajectory')  # 保存生成的地图
        logger.info(f'Sequence {i} End, Time = ' + ", ".join([f"{name}:{time[0]:.3f}/{time[1]:.3f}s" for name, time in slam_system.result_logger.log_time().items()]))
        # 记录日志，显示当前序列处理完毕以及各部分处理时间


if __name__ == "__main__":
    main()
    logger.info('Done.')
