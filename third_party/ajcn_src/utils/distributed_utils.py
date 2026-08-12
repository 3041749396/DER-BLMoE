import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import torch.backends.cudnn as cudnn

def setup_distributed(rank, world_size, backend='nccl', init_method='env://'):
    """
    设置分布式训练环境
    
    参数:
        rank: 当前进程的排名
        world_size: 总进程数
        backend: 分布式后端 ('nccl' 或 'gloo')
        init_method: 初始化方法
    """
    # 设置环境变量
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    
    # 初始化进程组
    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前设备
    torch.cuda.set_device(rank)
    
    # 启用cudnn benchmark以提高性能
    cudnn.benchmark = True

def cleanup_distributed():
    """
    清理分布式训练环境
    """
    if dist.is_initialized():
        dist.destroy_process_group()

def get_device_count():
    """
    获取可用GPU数量
    
    返回:
        int: 可用GPU数量
    """
    return torch.cuda.device_count()

def model_to_distributed(model, rank=None, parallel_mode='ddp', sync_bn=True):
    """
    将模型转换为分布式模型
    
    参数:
        model: 要转换的模型
        rank: 当前进程的排名
        parallel_mode: 并行模式 ('dp' 或 'ddp')
        sync_bn: 是否使用同步批归一化
    
    返回:
        分布式模型
    """
    # 如果启用了同步批归一化，则转换模型的批归一化层
    if sync_bn and parallel_mode == 'ddp':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 根据并行模式转换模型
    if parallel_mode == 'ddp':
        if rank is None:
            raise ValueError("使用DDP时必须提供rank参数")
        model = model.cuda(rank)
        model = DDP(model, device_ids=[rank])
    elif parallel_mode == 'dp':
        model = model.cuda()
        model = DP(model)
    else:
        raise ValueError(f"不支持的并行模式: {parallel_mode}")
    
    return model

def is_main_process(rank):
    """
    检查当前进程是否为主进程
    
    参数:
        rank: 当前进程的排名
    
    返回:
        bool: 如果是主进程则为True，否则为False
    """
    return rank == 0

def all_gather(tensor):
    """
    从所有进程收集张量
    
    参数:
        tensor: 要收集的张量
    
    返回:
        list: 所有进程的张量列表
    """
    output = [torch.zeros_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(output, tensor)
    return output

def reduce_dict(input_dict, average=True):
    """
    减少字典中的所有值（在多进程之间）
    
    参数:
        input_dict: 输入字典
        average: 是否对结果取平均值
    
    返回:
        dict: 减少后的字典
    """
    world_size = dist.get_world_size()
    if world_size < 2:
        return input_dict
    
    with torch.no_grad():
        names = []
        values = []
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])
            
        values = torch.stack(values, dim=0)
        dist.all_reduce(values)
        
        if average:
            values /= world_size
            
        reduced_dict = {k: v for k, v in zip(names, values)}
        
    return reduced_dict

def synchronize():
    """
    同步所有进程
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def run_distributed(fn, world_size, config, *args, **kwargs):
    """
    运行分布式训练函数
    
    参数:
        fn: 要运行的函数
        world_size: 总进程数
        config: 配置
        *args, **kwargs: 传递给fn的其他参数
    """
    if world_size == 1:
        # 单GPU模式
        return fn(0, world_size, config, *args, **kwargs)
    else:
        # 多GPU模式
        mp.spawn(fn, args=(world_size, config, *args), nprocs=world_size, join=True) 