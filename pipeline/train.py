import os
import sys
import yaml
from parameters import *

args = parser.parse_args()  # args 储存命令行的输入参数
if not args.use_ddp and args.use_cuda:  # 如果没有使用分布式数据并行且使用了CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index  # 设置CUDA可见的设备，根据gpu_index选择GPU
sys.path.insert(1, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))  # 将父目录添加到Python路径中，以便模块导入

import colorlog as logging  # 导入colorlog模块，用于带有颜色的日志记录

logging.basicConfig(level=logging.INFO)  # 配置日志记录器，日志级别为INFO
logger = logging.getLogger(__name__)  # 创建一个日志记录器对象，名称为当前模块名
logger.setLevel(logging.INFO)  # 设置日志记录器的日志级别为INFO
import warnings  # 导入warnings模块，用于控制警告信息的显示

warnings.filterwarnings("ignore")  # 忽略所有警告信息

import torch  # 导入PyTorch框架
import torch.distributed as dist  # 导入PyTorch分布式模块
import torch.multiprocessing  # 导入PyTorch多进程模块

torch.multiprocessing.set_sharing_strategy('file_system')  # 设置多进程共享策略为文件系统，以避免文件句柄共享问题

from dataloader.body import SlamDatasets  # 从dataloader.body模块中导入SlamDatasets类，用于加载SLAM数据集
from dataloader.transforms import PointCloudTransforms  # 从dataloader.transforms模块中导入PointCloudTransforms类，用于点云数据变换
from network.encoder.encoder import Encoder  # 从network.encoder.encoder模块中导入Encoder类，用于定义编码器网络结构
from network.decoder.decoder import Decoder  # 从network.decoder.decoder模块中导入Decoder类，用于定义解码器网络结构
from network.loss import RegistrationLoss  # 从network.loss模块中导入RegistrationLoss类，用于定义注册损失函数
from modules.model_pipeline import DeepPointModelPipeline  # 从modules.model_pipeline模块中导入DeepPointModelPipeline类，用于构建整个模型流水线
from modules.trainer import Trainer  # 从modules.trainer模块中导入Trainer类，用于定义训练流程


def main():
    """任务入口"""
    '''参数解析与设置'''
    global args
    if not os.path.exists(args.yaml_file):
        raise FileNotFoundError(f'yaml_file \'{args.yaml_file}\' is not found!')
    logger.info(f'Loading config from \'{args.yaml_file}\'...')
    with open(args.yaml_file, 'r', encoding='utf-8') as f:
        cfg = yaml.load(f, yaml.FullLoader)
    args = update_args(args, cfg)
    if not args.thread_safety:
        torch.multiprocessing.set_start_method('spawn')
        logger.warning(f'The start method of torch.multiprocessing has been set to \'spawn\'')
    if args.use_ddp and torch.cuda.is_available():
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.device = torch.device('cuda', args.local_rank)
        dist.init_process_group(backend='nccl', rank=args.local_rank, world_size=args.word_size)
        torch.cuda.set_device(args.device)
    elif args.use_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        gpus = list(range(torch.cuda.device_count()))
        torch.cuda.set_device('cuda:{}'.format(gpus[0]))
    else:
        args.device = torch.device('cpu')
    '''数据变换、加载数据集'''
    logger.info('Preparing data...')
    transforms = PointCloudTransforms(args=args, mode='train')  # 包含了配置文件中transforms下的所有变换的列表
    if args.use_ddp:  # 如果启用分布式数据并行
        if args.local_rank == 0:
            dataset = SlamDatasets(args=args, data_transforms=transforms)
            print(dataset)
        dist.barrier()
        if args.local_rank != 0:
            dataset = SlamDatasets(args=args, data_transforms=transforms)
    else:  # 如果未启用分布式数据并行
        dataset = SlamDatasets(args=args, data_transforms=transforms)  # dataset用于储存所有的数据集
        print(dataset)
    '''模型与损失函数'''
    logger.info('Preparing model...')  # 记录日志
    encoder = Encoder(args=args)  # 创建Encoder对象，定义编码器网络结构
    decoder = Decoder(args=args)  # 创建Decoder对象，定义解码器网络结构
    criterion = RegistrationLoss(args=args)  # 创建RegistrationLoss对象，定义损失函数
    model = DeepPointModelPipeline(args=args, encoder=encoder, decoder=decoder, criterion=criterion)  # 创建DeepPointModelPipeline对象，构建整个模型流水线
    '''训练流程'''
    logger.info('Launching trainer...')
    trainer = Trainer(args=args, dataset=dataset, model=model)  # 创建Trainer对象，传入参数、数据集和模型，定义训练流程
    trainer.run()  # 运行训练流程


if __name__ == "__main__":
    main()
    logger.info('Done.')
