import os
import torch
import numpy as np
import yaml
import argparse
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

# 导入自定义模块
from models.base_model import BaseModel
from utils.data_processor import DataProcessor
from rl.distributed_environment import DistributedCompressionEnv
from rl.distributed_ppo_agent import DistributedPPOAgent
from utils.evaluation import evaluate_compression
from utils.visualization import plot_training_curves, plot_compression_results, plot_layer_compression
from utils.distributed_utils import setup_distributed, cleanup_distributed, get_device_count, is_main_process, run_distributed

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def distributed_train_rl_compression(rank, world_size, config):
    """
    分布式训练RL压缩策略
    
    参数:
        rank: 当前进程的排名
        world_size: 总进程数
        config: 配置字典
    """
    # 设置分布式训练环境
    setup_distributed(
        rank, 
        world_size, 
        backend=config['distributed']['backend'],
        init_method=config['distributed']['init_method']
    )
    
    # 设置设备
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if is_main_process(rank):
        print(f"进程 {rank}: 使用设备 {device}")
    
    # 设置随机种子
    set_seed(42 + rank)  # 每个进程使用不同的种子
    
    # 创建输出目录
    if is_main_process(rank):
        os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    # 加载数据
    if is_main_process(rank):
        print(f"进程 {rank}: 加载ADS-B数据...")
    
    # 增加num_workers参数到DataProcessor
    data_processor = DataProcessor(
        batch_size=config['dataset']['batch_size'],
        data_dir=config['dataset']['data_dir']
    )
    train_loader, val_loader, test_loader = data_processor.load_data()
    
    if is_main_process(rank):
        print(f"训练集大小: {len(train_loader.dataset)}")
        print(f"验证集大小: {len(val_loader.dataset)}")
        print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 获取数据集的类别数
    num_classes = len(torch.unique(train_loader.dataset.tensors[1]))
    if is_main_process(rank):
        print(f"类别数: {num_classes}")
    
    # 创建模型
    if is_main_process(rank):
        print(f"进程 {rank}: 创建模型...")
    
    model = BaseModel(num_classes=num_classes)
    
    # 加载预训练模型（如果有）
    if config['model']['checkpoint_path']:
        # 加载到CPU，然后转移到正确的设备
        model.load_state_dict(torch.load(config['model']['checkpoint_path'], map_location='cpu'))
        if is_main_process(rank):
            print(f"加载预训练模型: {config['model']['checkpoint_path']}")
    
    # 创建分布式压缩环境
    if is_main_process(rank):
        print(f"进程 {rank}: 创建压缩环境...")
    
    env = DistributedCompressionEnv(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        rank=rank,
        world_size=world_size,
        target_accuracy=config['compression']['target_accuracy'],
        target_compression=config['compression']['target_compression'],
        alpha=config['compression']['reward_weights']['accuracy'],
        beta=config['compression']['reward_weights']['model_size'],
        gamma=config['compression']['reward_weights']['inference_time']
    )
    
    # 创建分布式PPO代理
    if is_main_process(rank):
        print(f"进程 {rank}: 创建PPO代理...")
    
    action_dims = [
        len(config['rl']['action_space']['prune_ratios']),
        len(config['rl']['action_space']['convert_options'])
    ]
    
    agent = DistributedPPOAgent(
        state_dim=env.state_dim,
        action_dims=action_dims,
        device=device,
        rank=rank,
        world_size=world_size,
        parallel_mode=config['distributed']['parallel_mode'],
        lr=config['rl']['lr'],
        gamma=config['rl']['gamma'],
        clip_ratio=config['rl']['clip_ratio'],
        value_coef=config['rl']['value_coef'],
        entropy_coef=config['rl']['entropy_coef']
    )
    
    # 训练参数
    num_episodes = config['rl']['num_episodes']
    save_interval = config['output']['save_interval']
    log_interval = config['output']['log_interval']
    
    # 记录训练数据（仅主进程）
    if is_main_process(rank):
        episode_rewards = []
        episode_accuracies = []
        episode_params_ratios = []
        episode_time_ratios = []
        
        # 创建结果记录文件
        results_file = os.path.join(config['output']['save_dir'], 'compression_results.txt')
        with open(results_file, 'w') as f:
            f.write("# 网络压缩实验结果\n")
            f.write("# 格式: Episode, Reward, Accuracy(%), Accuracy_Ratio, Params(M), Params_Ratio, FLOPs(M), FLOPs_Ratio, Inference_Time(ms), Time_Ratio\n")
            f.write(f"# 原始模型 - 精度: {env.original_accuracy:.2f}%, 参数量: {env.original_params:.4f}M, FLOPs: {env.original_flops:.4f}M, 推理时间: {env.original_time:.4f}ms\n\n")
        
        # 记录每个层的压缩决策
        compression_decisions_file = os.path.join(config['output']['save_dir'], 'compression_decisions.txt')
        with open(compression_decisions_file, 'w') as f:
            f.write("# 网络压缩决策记录\n")
            f.write("# 格式: Episode, Layer, Prune_Ratio, Convert_To_Depthwise\n\n")
        
        # 记录最佳模型的详细结构
        best_model_structure_file = os.path.join(config['output']['save_dir'], 'best_model_structure.txt')
    
    # 最佳模型跟踪（所有进程共享）
    best_reward = float('-inf')
    best_model = None
    best_prune_ratios = None
    best_convert_blocks = None
    
    # 开始训练
    if is_main_process(rank):
        print(f"进程 {rank}: 开始训练，共{num_episodes}轮...")
        start_time = time.time()
    
    for episode in range(num_episodes):
        # 重置环境
        state = env.reset()
        episode_reward = 0
        episode_decisions = []  # 记录本轮的决策
        
        # 执行一个episode
        done = False
        while not done:
            # 选择动作
            action, log_prob = agent.select_action(state)
            
            # 记录决策（仅主进程）
            if is_main_process(rank):
                layer_idx = env.current_layer_idx
                layer_name = env.layers[layer_idx] if layer_idx < len(env.layers) else "unknown"
                prune_ratio = env.prune_ratio_options[action[0]]
                convert_action = action[1]
                episode_decisions.append((layer_name, prune_ratio, convert_action))
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储经验
            agent.store_transition(state, action, log_prob, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
        
        # 更新策略
        agent.update(
            batch_size=config['rl']['batch_size'],
            epochs=config['rl']['update_epochs']
        )
        
        # 评估当前模型
        current_accuracy = env.evaluate(env.current_model)
        current_flops, current_params = env.estimate_model_complexity(env.current_model)
        current_time = env.get_inference_time(env.current_model)
        
        # 计算比率
        accuracy_ratio = current_accuracy / env.original_accuracy
        params_ratio = current_params / env.original_params
        flops_ratio = current_flops / env.original_flops
        time_ratio = current_time / env.original_time
        
        # 主进程记录训练数据
        if is_main_process(rank):
            episode_rewards.append(episode_reward)
            episode_accuracies.append(accuracy_ratio)
            episode_params_ratios.append(params_ratio)
            episode_time_ratios.append(time_ratio)
            
            # 记录本轮结果
            with open(results_file, 'a') as f:
                f.write(f"{episode+1}, {episode_reward:.4f}, {current_accuracy:.4f}, {accuracy_ratio:.4f}, "
                       f"{current_params:.4f}, {params_ratio:.4f}, {current_flops:.4f}, {flops_ratio:.4f}, "
                       f"{current_time:.4f}, {time_ratio:.4f}\n")
            
            # 记录本轮决策
            with open(compression_decisions_file, 'a') as f:
                f.write(f"# Episode {episode+1}\n")
                for i, (layer, prune_ratio, convert) in enumerate(episode_decisions):
                    f.write(f"{episode+1}, {layer}, {prune_ratio:.2f}, {convert}\n")
                f.write("\n")
            
            # 打印日志
            if (episode + 1) % log_interval == 0:
                elapsed_time = time.time() - start_time
                print(f"Episode {episode+1}/{num_episodes} | 奖励: {episode_reward:.4f} | "
                     f"精度: {current_accuracy:.2f}% ({accuracy_ratio:.2f}x) | "
                     f"参数: {current_params:.4f}M ({params_ratio:.2f}x) | "
                     f"推理时间: {current_time:.4f}ms ({time_ratio:.2f}x) | "
                     f"用时: {elapsed_time:.2f}s")
        
        # 更新最佳模型（所有进程）
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model = env.current_model
            best_prune_ratios = env.prune_ratios.copy()
            best_convert_blocks = env.convert_blocks.copy()
            
            # 主进程保存最佳模型
            if is_main_process(rank):
                best_results = {
                    'accuracy': current_accuracy,
                    'accuracy_ratio': accuracy_ratio,
                    'params': current_params,
                    'params_ratio': params_ratio,
                    'flops': current_flops,
                    'flops_ratio': flops_ratio,
                    'time': current_time,
                    'time_ratio': time_ratio
                }
                
                # 保存最佳模型
                torch.save(best_model.state_dict(), os.path.join(config['output']['save_dir'], 'best_compressed_model.pth'))
                
                # 保存最佳压缩策略
                np.save(os.path.join(config['output']['save_dir'], 'best_prune_ratios.npy'), best_prune_ratios)
                np.save(os.path.join(config['output']['save_dir'], 'best_convert_blocks.npy'), best_convert_blocks)
                
                # 记录最佳模型结构
                with open(best_model_structure_file, 'w') as f:
                    f.write("# 最佳压缩模型结构\n\n")
                    f.write(str(best_model))
                    f.write("\n\n# 压缩决策\n")
                    for i, layer in enumerate(env.layers):
                        if i < len(best_prune_ratios):
                            prune_ratio = best_prune_ratios[i]
                            convert = "是" if best_convert_blocks[i] == 1 else "否"
                            f.write(f"层 {layer}: 剪枝率={prune_ratio:.2f}, 转换为深度可分离卷积={convert}\n")
                    f.write("\n# 压缩效果\n")
                    f.write(f"原始精度: {env.original_accuracy:.4f}%, 压缩后精度: {current_accuracy:.4f}% ({accuracy_ratio:.2f}x)\n")
                    f.write(f"原始参数量: {env.original_params:.4f}M, 压缩后参数量: {current_params:.4f}M ({params_ratio:.2f}x)\n")
                    f.write(f"原始FLOPs: {env.original_flops:.4f}M, 压缩后FLOPs: {current_flops:.4f}M ({flops_ratio:.2f}x)\n")
                    f.write(f"原始推理时间: {env.original_time:.4f}ms, 压缩后推理时间: {current_time:.4f}ms ({time_ratio:.2f}x)\n")
        
        # 定期保存模型和可视化结果（仅主进程）
        if is_main_process(rank) and (episode + 1) % save_interval == 0:
            # 保存当前模型和代理
            torch.save(env.current_model.state_dict(), 
                      os.path.join(config['output']['save_dir'], f'model_episode_{episode+1}.pth'))
            agent.save(os.path.join(config['output']['save_dir'], f'agent_episode_{episode+1}.pth'))
            
            # 保存可视化数据
            if config['output']['visualize']:
                generate_visualization_data(
                    config['output']['save_dir'],
                    episode_rewards,
                    episode_accuracies,
                    episode_params_ratios,
                    episode_time_ratios
                )
    
    # 训练结束，保存最终结果（仅主进程）
    if is_main_process(rank):
        elapsed_time = time.time() - start_time
        print(f"训练完成！总用时: {elapsed_time:.2f}秒")
        
        # 保存最终模型
        torch.save(env.current_model.state_dict(), 
                  os.path.join(config['output']['save_dir'], 'final_compressed_model.pth'))
        agent.save(os.path.join(config['output']['save_dir'], 'final_agent.pth'))
        
        # 保存最终可视化数据
        if config['output']['visualize']:
            generate_visualization_data(
                config['output']['save_dir'],
                episode_rewards,
                episode_accuracies,
                episode_params_ratios,
                episode_time_ratios
            )
            
            # 绘制图表
            plot_training_curves(
                config['output']['save_dir'],
                episode_rewards,
                episode_accuracies,
                episode_params_ratios,
                episode_time_ratios
            )
    
    # 清理分布式环境
    cleanup_distributed()

def generate_visualization_data(save_dir, rewards, accuracies, params_ratios, time_ratios):
    """
    生成可视化数据
    
    参数:
        save_dir: 保存目录
        rewards: 奖励列表
        accuracies: 精度比例列表
        params_ratios: 参数比例列表
        time_ratios: 时间比例列表
    """
    os.makedirs(os.path.join(save_dir, 'visualization'), exist_ok=True)
    
    # 保存数据为numpy数组
    np.save(os.path.join(save_dir, 'visualization', 'rewards.npy'), np.array(rewards))
    np.save(os.path.join(save_dir, 'visualization', 'accuracies.npy'), np.array(accuracies))
    np.save(os.path.join(save_dir, 'visualization', 'params_ratios.npy'), np.array(params_ratios))
    np.save(os.path.join(save_dir, 'visualization', 'time_ratios.npy'), np.array(time_ratios))

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分布式RL网络压缩训练")
    parser.add_argument("--config", type=str, default="configs/distributed_config.yaml", help="配置文件路径")
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 获取可用GPU数量
    num_gpus = get_device_count()
    
    if num_gpus == 0:
        print("错误: 找不到可用的GPU。请确保已安装CUDA并且有可用的GPU。")
        exit(1)
    
    # 确定进程数量
    world_size = config['distributed']['num_processes']
    if world_size == -1 or world_size > num_gpus:
        world_size = num_gpus
    
    print(f"检测到 {num_gpus} 个GPU，将使用 {world_size} 个进程进行训练")
    
    # 运行分布式训练
    if world_size > 1:
        run_distributed(distributed_train_rl_compression, world_size, config)
    else:
        # 单GPU模式
        distributed_train_rl_compression(0, 1, config)

if __name__ == "__main__":
    main() 