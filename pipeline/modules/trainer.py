import colorlog as logging  # 导入colorlog模块，并将其命名为logging，用于彩色日志记录。

logging.basicConfig(level=logging.INFO)  # 设置日志记录的基本配置，日志级别为INFO。
logger = logging.getLogger(__name__)  # 获取一个名为当前模块名的logger对象，用于日志记录。
logger.setLevel(logging.INFO)  # 设置logger对象的日志级别为INFO。

import os  # 导入os模块，用于操作系统相关的功能，如环境变量和文件路径操作。
import time  # 导入time模块，用于处理时间相关的操作，如延迟和时间戳。
import zipfile  # 导入zipfile模块，用于处理zip压缩文件的创建、读取和解压缩。
import torch  # 导入PyTorch库，用于深度学习任务。
import torch.distributed as dist  # 导入PyTorch的分布式计算模块，用于多GPU训练。
from torch.nn.parallel import DistributedDataParallel  # 从PyTorch中导入分布式数据并行模块，用于将模型并行到多个GPU上。
from torch.utils.data import DataLoader  # 从PyTorch中导入DataLoader类，用于加载数据集。
from torch.utils.data.distributed import DistributedSampler  # 从PyTorch中导入分布式采样器，用于在分布式训练中分配数据。
from torch.utils.tensorboard import SummaryWriter  # 从PyTorch中导入SummaryWriter类，用于将训练日志写入TensorBoard。
from torch.cuda.amp import autocast as autocast  # 从PyTorch中导入autocast，用于自动混合精度训练，减少显存使用并加快计算。
from glob import glob  # 导入glob模块，用于查找符合特定模式的文件路径。
from tqdm import tqdm  # 导入tqdm模块，用于显示循环进度条。
from collections import OrderedDict  # 从collections模块中导入OrderedDict类，用于创建有序字典。
from pipeline.modules.utils import Recorder, fakecast, Optimizer, Scheduler, try_load_state_dict  # 从pipeline.modules.utils中导入多个实用工具类和函数。
from pipeline.modules.model_pipeline import DeepPointModelPipeline  # 从pipeline.modules.model_pipeline中导入DeepPointModelPipeline类，用于构建深度学习模型流水线。
from utils.device import move_to_device  # 从utils.device中导入move_to_device函数，用于将数据或模型移动到指定的设备（如GPU）。
from dataloader.body import SlamDatasets  # 从dataloader.body中导入SlamDatasets类，用于加载SLAM数据集。

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 设置环境变量`CUDA_LAUNCH_BLOCKING`为1，确保CUDA错误发生时能够精确定位，便于调试。
torch.manual_seed(42)  # 设置随机种子为42，以确保结果的可重复性。


class Trainer:  # 定义一个名为Trainer的类，用于封装训练过程。
    """
    训练器，输入待训练的模型、参数，封装训练过程
    """

    def __init__(self, args, dataset: SlamDatasets, model: DeepPointModelPipeline):
        self.args = args  # 保存传入的参数args，这些通常包括训练配置和超参数。
        self.train_cfg = args.train  # 从args中提取训练配置，并保存到train_cfg中。
        self.dataset = dataset  # 保存传入的数据集对象，通常用于加载和处理训练数据。
        self.model = model  # 保存传入的模型对象，通常是需要训练的神经网络模型。
        self.stage_epoch = (self.train_cfg.registration.num_epochs, self.train_cfg.loop_detection.num_epochs)
        # 保存训练的阶段和对应的epoch数，包括配准和回环检测阶段。

        # 训练器件与参数
        self.optimizer = None  # 初始化优化器为None，稍后会设置。
        self.scheduler = None  # 初始化学习率调度器为None，稍后会设置。
        self.dataloader = None  # 初始化数据加载器为None，稍后会设置。
        self.sampler = None  # 初始化数据采样器为None，稍后会设置。
        self.writer = None  # 初始化TensorBoard记录器为None，稍后会设置。
        self.log_interval = None  # 初始化日志记录间隔为None，稍后会设置。
        self.epoch = 1  # 初始化当前epoch为1。
        self.step = 1  # 初始化当前步数为1。
        if self.train_cfg.auto_cast:
            self.cast = autocast  # 如果启用了自动混合精度训练，则使用autocast。
        else:
            self.cast = fakecast  # 否则，使用fakecast，这是一个不进行混合精度的占位符。
        self.log = f'{self.args.name}{self.args.version}_config={os.path.split(self.args.yaml_file)[1]}'
        # 设置日志记录的文件名，通常包含实验名称、版本号和配置文件名。
        self.save_root = os.path.join('log_train', self.log)  # 设置保存训练日志和模型的根目录。
        self.is_main_process = not (self.args.use_ddp and self.args.local_rank != 0)
        # 判断当前进程是否为主进程，在分布式训练中，只有主进程执行某些操作。

        # 初始化模型参数，from checkpoint else from scratch
        if args.checkpoint != '':
            self.load_checkpoint(args.checkpoint)  # 如果提供了checkpoint，则加载checkpoint继续训练。
        elif args.weight != '':
            self.load_weight(args.weight)  # 如果提供了预训练权重，则加载权重进行训练。
        else:
            self.init_scratch()  # 如果没有提供checkpoint或权重，则从头初始化模型。

        # 保存训练参数与代码文件
        if self.is_main_process:  # 只有主进程执行以下操作
            os.makedirs(self.save_root, exist_ok=True)  # 创建保存目录，如果目录不存在则创建。
            logger.info(f'save root = \'{self.save_root}\'')  # 记录保存目录的日志。
            args_dict = sorted(args._get_kwargs())  # 获取并排序所有的训练参数。
            with open(os.path.join(self.save_root, 'settings.yaml'), 'w+', encoding='utf-8') as arg_file:
                for k, v in args_dict:
                    arg_file.write(f'{k}: {v}\n')  # 将训练参数保存到settings.yaml文件中。
            code_files = [f for f in sorted(glob('./**/*.py', recursive=True)) if not os.path.basename(f).startswith('__')]
            # 获取所有Python代码文件的路径，排除以"__"开头的文件。
            zfile = zipfile.ZipFile(os.path.join(self.save_root, 'codes.zip'), mode='w', compression=zipfile.ZIP_DEFLATED, compresslevel=9)
            # 创建一个压缩文件，将代码文件压缩保存。
            for f in code_files:
                zfile.write(f)  # 将每个代码文件写入压缩包。
            zfile.close()  # 关闭压缩文件。
        s = f'Initialization completed, device = \'{self.args.device}\''  # 记录初始化完成的日志信息。
        if self.is_main_process:
            s += ' [MAIN PROCESS]'  # 如果是主进程，附加标识信息。
        logger.info(s)  # 输出日志信息。
        if self.args.use_ddp:
            dist.barrier()  # 在分布式训练中，所有进程等待同步。

    def run(self):
        if self.epoch <= self.stage_epoch[0]:  # 如果当前epoch小于等于配准阶段的epoch数
            self.dataset.registration()  # 设置数据集为配准模式。
            batch_size = self.train_cfg.registration.batch_size  # 获取配准阶段的batch大小。
        else:
            self.dataset.loop_detection()  # 设置数据集为回环检测模式。
            batch_size = self.train_cfg.loop_detection.batch_size  # 获取回环检测阶段的batch大小。

        if self.args.use_ddp:  # 如果使用分布式训练
            self.sampler = DistributedSampler(self.dataset)  # 使用分布式采样器分配数据。
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=self.args.num_workers, sampler=self.sampler, collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
            # 使用采样器创建数据加载器，pin_memory用于加速数据传输，drop_last用于舍弃不完整的最后一个batch。
        else:
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
            # 如果不使用分布式训练，创建一个普通的数据加载器，shuffle=True表示打乱数据。

        if self.is_main_process:  # 只有主进程执行以下操作
            self.writer = SummaryWriter(os.path.join('log_tb', self.log))  # 创建TensorBoard记录器，用于记录训练日志。
            train_record = Recorder()  # 创建一个Recorder对象，用于记录训练过程中的指标。

        start_epoch = self.epoch  # 设置开始训练的epoch为当前的epoch。
        for ep in range(start_epoch, sum(self.stage_epoch) + 1):  # 循环训练每个epoch，直到所有阶段的epoch结束。
            self._epoch_begin(ep)  # 调用_epoch_begin方法，执行每个epoch开始时的操作。

            train_metric = self.train_one_epoch()  # 训练一个epoch，并返回训练指标。

            self.scheduler.step()  # 更新学习率。

            if self.is_main_process:  # 只有主进程执行以下操作
                train_record.add_dict(train_metric)  # 将本次epoch的训练指标添加到记录器中。

                if ep % self.train_cfg.save_cycle == 0:  # 如果当前epoch是保存周期
                    self.save()  # 保存模型和训练状态。

            self.epoch += 1  # 增加epoch计数。

        if self.is_main_process:  # 只有主进程执行以下操作
            self.save(finish=True)  # 保存最终的模型和训练状态。
            print(train_record.tostring())  # 输出训练记录的总结信息。
        if self.args.use_ddp:
            dist.barrier()  # 在分布式训练中，所有进程等待同步。

    def _epoch_begin(self, ep):
        """每个epoch开始前的操作"""
        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            dist.barrier()  # 同步所有进程，确保所有进程都准备好开始新一轮的训练。

        if ep == self.stage_epoch[0] + 1:  # 如果当前epoch等于配准阶段结束后进入的回环检测阶段
            self._next_stage()  # 切换至训练回环检测的阶段。

        if ep <= self.stage_epoch[0]:  # 如果当前epoch在配准阶段
            registration_cfg = self.train_cfg.registration  # 获取配准阶段的训练配置。
            if 'K_0' in registration_cfg.keys():  # 如果配准配置中有K_0参数
                K_0 = registration_cfg['K_0']  # 获取初始配准尺度K_0。
                K_mult = registration_cfg['K_mult']  # 获取配准尺度的乘数K_mult。
                mult_epoch = registration_cfg['mult_epoch']  # 获取需要调整配准尺度的epoch列表。
                times = 0
                for i in mult_epoch:  # 遍历每个mult_epoch
                    if ep >= i:  # 如果当前epoch大于或等于指定的epoch
                        times += 1  # 增加调整次数
                registration_cfg['K'] = K_0 * (K_mult**times)  # 更新配准尺度K。
            batch_size = registration_cfg.batch_size  # 设置当前阶段的batch size。
            if self.is_main_process:  # 如果是主进程
                self.writer.add_scalar("runtime/K", registration_cfg['K'], ep)  # 将当前配准尺度记录到TensorBoard日志中。
        else:  # 如果当前epoch在回环检测阶段
            batch_size = self.train_cfg.loop_detection.batch_size  # 设置回环检测阶段的batch size。

        if self.is_main_process:  # 如果是主进程
            self.writer.add_scalar("runtime/learning_rate", self.optimizer.param_groups[0]['lr'], ep)  # 将当前学习率记录到TensorBoard日志中。

        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            self.sampler.set_epoch(ep)  # 设置数据采样器的epoch，以确保每个进程在不同epoch中采样不同的数据。
            log_interval = (self.train_cfg.log_cycle / self.args.world_size) // batch_size  # 计算日志记录的间隔（每隔多少个batch记录一次日志）。
        else:
            log_interval = self.train_cfg.log_cycle // batch_size  # 在单GPU模式下，计算日志记录间隔。
        self.log_interval = int(max(log_interval, 1))  # 确保日志记录间隔至少为1。

    def train_one_epoch(self):
        start_time = time.time()  # 记录当前epoch的开始时间。
        self.model.train()  # 将模型设置为训练模式。
        step_count = 0  # 初始化步数计数器。
        log_interval = self.log_interval  # 获取日志记录间隔。
        epoch_metrics = dict()  # 初始化字典，用于存储每个epoch的训练指标。

        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            dist.barrier()  # 同步所有进程，确保所有进程在同一时间开始训练。

        loop = tqdm(self.dataloader, total=len(self.dataloader), leave=False, dynamic_ncols=True)  # 使用tqdm创建一个带进度条的数据加载器。
        loop.set_description('train')  # 设置进度条的描述为'train'。
        for data in loop:  # 逐批次加载数据
            step_count += 1  # 增加步数计数器。
            data = move_to_device(data, device=self.args.device, non_blocking=True)  # 将数据移动到指定的设备上（如GPU），并启用非阻塞传输。

            # 前向传播与反向传播
            with self.cast():  # 使用自动混合精度（如果启用）
                loss, metric = self.model(*data)  # 前向传播，计算损失和训练指标。

            self.optimizer.zero_grad()  # 清除梯度。
            loss.backward()  # 反向传播计算梯度。
            self.optimizer.step()  # 更新模型参数。

            loop.set_postfix_str(' | '.join(f'{k}={v:2.4f}' for k, v in metric.items()))  # 在进度条中显示当前的训练指标。

            if self.is_main_process:  # 如果是主进程
                for metric_name, metric_value in metric.items():
                    epoch_metrics.setdefault(metric_name, []).append(metric_value)  # 将每步的训练指标添加到字典中。

                if step_count % log_interval == 0:  # 每隔log_interval步记录一次日志
                    for label, metric_list in epoch_metrics.items():
                        self.writer.add_scalar(f"train/step_{label}", sum(metric_list[-log_interval:]) / log_interval, self.step)  # 记录到TensorBoard日志中。
                self.step += 1  # 增加步数计数器。

        if not self.is_main_process:  # 如果不是主进程，直接返回。
            return None

        summary_str = ''
        summary_metric = {}
        for label, metric_list in epoch_metrics.items():  # 计算每个指标的平均值，并记录到TensorBoard日志中。
            self.writer.add_scalar(f"train/epoch_{label}", sum(metric_list) / len(metric_list), self.epoch)
            summary_str += f'{label} = {sum(metric_list) / len(metric_list):6.4f} | '
            summary_metric[label] = sum(metric_list) / len(metric_list)

        cost_time = time.time() - start_time  # 计算训练一个epoch所用的时间。
        cost_m, cost_s = divmod(cost_time, 60)  # 将时间转换为分钟和秒。
        cost_h, cost_m = divmod(cost_m, 60)  # 将分钟转换为小时。
        logger.info(f'Train Epoch {self.epoch:>4d} | ' + summary_str + f'Time = {int(cost_h)}h:{int(cost_m):02d}m:{cost_s:04.1f}s')  # 记录每个epoch的日志信息。
        return summary_metric  # 返回每个epoch的训练指标。

    def save(self, finish=False):
        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            encoder_state_dict = self.model.module.encoder.state_dict()  # 获取模型编码器的状态字典。
            decoder_state_dict = self.model.module.decoder.state_dict()  # 获取模型解码器的状态字典。
        else:
            encoder_state_dict = self.model.encoder.state_dict()  # 获取模型编码器的状态字典。
            decoder_state_dict = self.model.decoder.state_dict()  # 获取模型解码器的状态字典。
        if not finish:  # 如果不是最终保存
            state = {  # 创建一个包含模型状态、优化器状态、学习率调度器状态、当前epoch和步数的字典。
                'encoder': encoder_state_dict,
                'decoder': decoder_state_dict,
                'optimizer': self.optimizer.state_dict(),
                'scheduler': self.scheduler.state_dict(),
                'epoch': self.epoch,
                'step': self.step,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}{self.args.version}_epoch{self.epoch}.ckpt')  # 设置checkpoint保存路径。
        else:  # 如果是最终保存
            state = {  # 只保存模型的编码器和解码器状态字典。
                'encoder': encoder_state_dict,
                'decoder': decoder_state_dict,
            }
            file_path = os.path.join(self.save_root, f'{self.args.name}{self.args.version}.pth')  # 设置模型保存路径。
        torch.save(state, file_path)  # 保存状态字典到文件。

    def init_scratch(self):
        optimizer = Optimizer(self.train_cfg.registration.optimizer)  # 初始化优化器。
        scheduler = Scheduler(self.train_cfg.registration.scheduler)  # 初始化学习率调度器。
        self.model.registration()  # 设置模型为配准阶段。
        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            self.model = DistributedDataParallel(self.model.cuda(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank)  # 使用DistributedDataParallel包装模型，并将模型移动到指定GPU上。
        else:
            self.model = self.model.to(self.args.device)  # 将模型移动到指定设备（如GPU）。
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))  # 设置优化器，只优化需要梯度的参数。
        self.scheduler = scheduler(self.optimizer)  # 设置学习率调度器。
        if self.is_main_process:  # 如果是主进程
            logger.info(f'Training from scratch')  # 记录日志，表示从头开始训练。

    def load_checkpoint(self, checkpoint: str):
        if not os.path.exists(checkpoint):  # 检查checkpoint文件是否存在
            raise FileNotFoundError(f'checkpoint file \'{checkpoint}\' is not found.')  # 如果不存在，则抛出文件未找到的异常。
        checkpoint_file_path = checkpoint  # 保存checkpoint文件的路径。

        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            checkpoint = torch.load(checkpoint, map_location=f'cuda:{self.args.local_rank}')  # 加载checkpoint文件，并将其映射到当前进程的GPU上。
        else:
            checkpoint = torch.load(checkpoint, map_location=self.args.device)  # 加载checkpoint文件，并将其映射到指定的设备（如GPU）。

        # Load model
        self.epoch = checkpoint['epoch'] + 1  # 从checkpoint中获取epoch，并将当前epoch设置为加载的epoch加1。
        if self.is_main_process:  # 如果是主进程
            logger.info(f"Load epoch, current = {self.epoch}")  # 记录当前加载的epoch信息。
        self.step = checkpoint['step']  # 从checkpoint中获取步数，并设置当前步数。
        if self.is_main_process:  # 如果是主进程
            logger.info(f"Load step, current = {self.step}")  # 记录当前加载的步数信息。
        encoder_state_dict = checkpoint['encoder']  # 从checkpoint中获取编码器的状态字典。
        try_load_state_dict(self.model.encoder, encoder_state_dict, 'encoder', log=self.is_main_process)  # 尝试加载编码器的状态字典。
        decoder_state_dict = checkpoint['decoder']  # 从checkpoint中获取解码器的状态字典。
        try_load_state_dict(self.model.decoder, decoder_state_dict, 'decoder', log=self.is_main_process)  # 尝试加载解码器的状态字典。

        # 根据训练轮数判定当前训练阶段
        if self.epoch <= self.stage_epoch[0]:  # 如果当前epoch在配准阶段
            self.model.registration()  # 设置模型为配准阶段。
            optimizer = Optimizer(self.train_cfg.registration.optimizer)  # 初始化配准阶段的优化器。
            scheduler = Scheduler(self.train_cfg.registration.scheduler)  # 初始化配准阶段的学习率调度器。
        else:  # 如果当前epoch在回环检测阶段
            self.model.loop_detection()  # 设置模型为回环检测阶段。
            optimizer = Optimizer(self.train_cfg.loop_detection.optimizer)  # 初始化回环检测阶段的优化器。
            scheduler = Scheduler(self.train_cfg.loop_detection.scheduler)  # 初始化回环检测阶段的学习率调度器。
        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            self.model = DistributedDataParallel(self.model.cuda(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank)  # 使用DistributedDataParallel包装模型，并将其移动到指定GPU上。
        else:
            self.model = self.model.to(self.args.device)  # 将模型移动到指定设备（如GPU）。
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))  # 设置优化器，只优化需要梯度的参数。
        self.scheduler = scheduler(self.optimizer)  # 设置学习率调度器。

        # 恰好为阶段转换的轮次时无需加载optimizer和scheduler，其余轮次则从checkpoint中加载
        if self.epoch != self.stage_epoch[0] + 1:  # 如果当前epoch不在阶段转换时（即不是配准阶段到回环检测阶段的过渡）
            try_load_state_dict(self.optimizer, checkpoint['optimizer'], 'optimizer', log=self.is_main_process)  # 尝试加载优化器的状态字典。
            try_load_state_dict(self.scheduler, checkpoint['scheduler'], 'scheduler', log=self.is_main_process)  # 尝试加载学习率调度器的状态字典。
        if self.is_main_process:  # 如果是主进程
            logger.info(f'Load checkpoint done. \'{checkpoint_file_path}\'')  # 记录加载checkpoint完成的信息。

    def load_weight(self, weight: str):
        if not os.path.exists(weight):  # 检查weight文件是否存在
            raise FileNotFoundError(f'weight file \'{weight}\' is not found.')  # 如果不存在，则抛出文件未找到的异常。
        weight_file_path = weight  # 保存weight文件的路径。

        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            weight = torch.load(weight, map_location=f'cuda:{self.args.local_rank}')  # 加载weight文件，并将其映射到当前进程的GPU上。
        else:
            weight = torch.load(weight, map_location=self.args.device)  # 加载weight文件，并将其映射到指定的设备（如GPU）。

        encoder_state_dict = weight['encoder']  # 从weight中获取编码器的状态字典。
        try_load_state_dict(self.model.encoder, encoder_state_dict, 'encoder', log=self.is_main_process)  # 尝试加载编码器的状态字典。
        decoder_state_dict = weight['decoder']  # 从weight中获取解码器的状态字典。
        try_load_state_dict(self.model.decoder, decoder_state_dict, 'decoder', log=self.is_main_process)  # 尝试加载解码器的状态字典。
        self.init_scratch()  # 从头初始化模型的优化器和学习率调度器。
        if self.is_main_process:  # 如果是主进程
            logger.info(f'Load specific weight from \'{weight_file_path}\'')  # 记录加载weight完成的信息。

    def _next_stage(self):
        # 切换训练阶段至回环检测
        self.dataset.loop_detection()  # 设置数据集为回环检测模式。
        batch_size = self.train_cfg.loop_detection.batch_size  # 获取回环检测阶段的batch size。
        if self.args.use_ddp:  # 如果使用分布式数据并行（DDP）
            model = self.model.module  # 获取模型的模块。
            model.loop_detection()  # 设置模型为回环检测阶段。
            self.model = DistributedDataParallel(model.cuda(self.args.local_rank), device_ids=[self.args.local_rank], output_device=self.args.local_rank)  # 使用DistributedDataParallel包装模型，并将其移动到指定GPU上。
            self.sampler = DistributedSampler(self.dataset)  # 设置分布式采样器。
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=self.args.num_workers, sampler=self.sampler, collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
            # 创建数据加载器，使用分布式采样器，pin_memory用于加速数据传输，drop_last用于舍弃不完整的最后一个batch。
        else:
            self.model.loop_detection()  # 设置模型为回环检测阶段。
            self.dataloader = DataLoader(dataset=self.dataset, batch_size=batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=self.dataset.collate_fn, pin_memory=True, drop_last=True)
            # 创建一个普通的数据加载器，shuffle=True表示打乱数据。
        optimizer = Optimizer(self.train_cfg.loop_detection.optimizer)  # 初始化回环检测阶段的优化器。
        scheduler = Scheduler(self.train_cfg.loop_detection.scheduler)  # 初始化回环检测阶段的学习率调度器。
        self.optimizer = optimizer(filter(lambda p: p.requires_grad, self.model.parameters()))  # 设置优化器，只优化需要梯度的参数。
        self.scheduler = scheduler(self.optimizer)  # 设置学习率调度器。
        if self.is_main_process:  # 如果是主进程
            logger.info(f'Convert the training stage from registration to loop-detection')  # 记录阶段转换的信息。

    @staticmethod
    def add_module(state_dict):
        new_state_dict = OrderedDict()  # 创建一个有序字典，用于存储新的状态字典。
        for k, v in state_dict.items():
            if not k.startswith('module.'):  # 如果状态字典的键不以'module.'开头
                k = 'module.' + k  # 为键添加'module.'前缀。
            new_state_dict[k] = v  # 将键值对添加到新的状态字典中。
        return new_state_dict  # 返回新的状态字典。

    @staticmethod
    def remove_module(state_dict):
        new_state_dict = OrderedDict()  # 创建一个有序字典，用于存储新的状态字典。
        for k, v in state_dict.items():
            if k.startswith('module.'):  # 如果状态字典的键以'module.'开头
                k = k[7:]  # 去掉'module.'前缀。
            new_state_dict[k] = v  # 将键值对添加到新的状态字典中。
        return new_state_dict  # 返回新的状态字典。
