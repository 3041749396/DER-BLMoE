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
from rl.environment import CompressionEnv
from rl.ppo_agent import PPOAgent
from utils.evaluation import evaluate_compression
from utils.visualization import plot_training_curves, plot_compression_results, plot_layer_compression

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

def train_rl_compression(config):
    """
    使用RL训练网络压缩策略
    
    参数:
        config: 配置字典
    """
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 设置随机种子
    set_seed(42)
    
    # 创建输出目录
    os.makedirs(config['output']['save_dir'], exist_ok=True)
    
    # 加载数据
    print("加载ADS-B数据...")
    data_processor = DataProcessor(
        batch_size=config['dataset']['batch_size'],
        data_dir=config['dataset']['data_dir']
    )
    train_loader, val_loader, test_loader = data_processor.load_data()
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 获取数据集的类别数
    num_classes = len(torch.unique(train_loader.dataset.tensors[1]))
    print(f"类别数: {num_classes}")
    
    # 创建模型
    print("创建模型...")
    model = BaseModel(num_classes=num_classes)
    model.load_state_dict(torch.load("base_model.pth"))
    # print(model)
    #exit()  # 注释掉这行，让程序继续执行
    
    # 加载预训练模型（如果有）
    if config['model']['checkpoint_path']:
        model.load_state_dict(torch.load(config['model']['checkpoint_path']))
        print(f"加载预训练模型: {config['model']['checkpoint_path']}")
    
    # 在创建模型后添加以下代码
    print("Model structure:")
    for name, module in model.named_modules():
        print(f"{name}: {module}")

    print("\nModel state dict keys:")
    for key in model.state_dict().keys():
        print(key)
    
    # 创建压缩环境
    print("创建压缩环境...")
    env = CompressionEnv(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        target_accuracy=config['compression']['target_accuracy'],
        target_compression=config['compression']['target_compression'],
        alpha=config['compression']['reward_weights']['accuracy'],
        beta=config['compression']['reward_weights']['model_size'],
        gamma=config['compression']['reward_weights']['inference_time']
    )
    
    # 创建PPO代理
    print("创建PPO代理...")
    action_dims = [
        len(config['rl']['action_space']['prune_ratios']),
        len(config['rl']['action_space']['convert_options'])
    ]
    
    agent = PPOAgent(
        state_dim=env.state_dim,
        action_dims=action_dims,
        device=device,
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
    
    # 记录训练数据
    episode_rewards = []
    episode_accuracies = []
    episode_params_ratios = []
    episode_time_ratios = []
    
    best_reward = float('-inf')
    best_model = None
    best_prune_ratios = None
    best_convert_blocks = None
    
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
    
    # 开始训练
    print(f"开始训练，共{num_episodes}轮...")
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
            
            # 记录决策
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
        
        # 记录训练数据
        episode_rewards.append(episode_reward)
        
        # 评估当前模型
        current_accuracy = env.evaluate(env.current_model)
        current_flops, current_params = env.estimate_model_complexity(env.current_model)
        current_time = env.get_inference_time(env.current_model)
        
        # 计算比率
        accuracy_ratio = current_accuracy / env.original_accuracy
        params_ratio = current_params / env.original_params
        flops_ratio = current_flops / env.original_flops
        time_ratio = current_time / env.original_time
        
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
        
        # 更新最佳模型
        if episode_reward > best_reward:
            best_reward = episode_reward
            best_model = env.current_model
            best_prune_ratios = env.prune_ratios.copy()
            best_convert_blocks = env.convert_blocks.copy()
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
            np.save(os.path.join(config['output']['save_dir'], 'best_prune_ratios_1.npy'), best_prune_ratios)
            np.save(os.path.join(config['output']['save_dir'], 'best_convert_blocks_1.npy'), best_convert_blocks)
            
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
        
        # 打印日志
        if (episode + 1) % log_interval == 0:
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.4f}, Accuracy: {current_accuracy:.2f}%")
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            agent.save(os.path.join(config['output']['save_dir'], f'agent_episode_{episode+1}.pth'))
            
            # 绘制训练曲线
            if config['output']['visualize']:
                plot_training_curves(
                    episode_rewards,
                    episode_accuracies,
                    episode_params_ratios,
                    episode_time_ratios,
                    save_path=os.path.join(config['output']['save_dir'], f'training_curves_episode_{episode+1}.png')
                )
        
        # 记录训练数据
        episode_accuracies.append(current_accuracy)
        episode_params_ratios.append(params_ratio)
        episode_time_ratios.append(time_ratio)
    
    # 训练结束
    end_time = time.time()
    training_time = end_time - start_time
    print(f"训练完成，耗时: {training_time:.2f}秒")
    
    # 训练结束，记录总结信息
    with open(os.path.join(config['output']['save_dir'], 'compression_summary.txt'), 'w') as f:
        f.write("# 网络压缩实验总结\n\n")
        f.write(f"训练轮数: {num_episodes}\n")
        f.write(f"训练时间: {(time.time() - start_time)/60:.2f}分钟\n\n")
        
        f.write("## 原始模型\n")
        f.write(f"精度: {env.original_accuracy:.4f}%\n")
        f.write(f"参数量: {env.original_params:.4f}M\n")
        f.write(f"FLOPs: {env.original_flops:.4f}M\n")
        f.write(f"推理时间: {env.original_time:.4f}ms\n\n")
        
        f.write("## 最佳压缩模型\n")
        f.write(f"精度: {best_results['accuracy']:.4f}% ({best_results['accuracy_ratio']:.2f}x)\n")
        f.write(f"参数量: {best_results['params']:.4f}M ({best_results['params_ratio']:.2f}x)\n")
        f.write(f"FLOPs: {best_results['flops']:.4f}M ({best_results['flops_ratio']:.2f}x)\n")
        f.write(f"推理时间: {best_results['time']:.4f}ms ({best_results['time_ratio']:.2f}x)\n\n")
        
        f.write("## 压缩决策\n")
        for i, layer in enumerate(env.layers):
            if i < len(best_prune_ratios):
                prune_ratio = best_prune_ratios[i]
                convert = "是" if best_convert_blocks[i] == 1 else "否"
                f.write(f"层 {layer}: 剪枝率={prune_ratio:.2f}, 转换为深度可分离卷积={convert}\n")
    
    # 生成可视化数据
    generate_visualization_data(config['output']['save_dir'], episode_rewards, episode_accuracies, 
                               episode_params_ratios, episode_time_ratios)
    
    return best_model, best_prune_ratios, best_convert_blocks, best_results

def generate_visualization_data(save_dir, rewards, accuracies, params_ratios, time_ratios):
    """生成用于可视化的数据文件"""
    # 奖励曲线数据
    with open(os.path.join(save_dir, 'reward_curve.txt'), 'w') as f:
        f.write("# Episode, Reward\n")
        for i, reward in enumerate(rewards):
            f.write(f"{i+1}, {reward:.4f}\n")
    
    # 精度曲线数据
    with open(os.path.join(save_dir, 'accuracy_curve.txt'), 'w') as f:
        f.write("# Episode, Accuracy_Ratio\n")
        for i, acc in enumerate(accuracies):
            f.write(f"{i+1}, {acc:.4f}\n")
    
    # 参数量曲线数据
    with open(os.path.join(save_dir, 'params_curve.txt'), 'w') as f:
        f.write("# Episode, Params_Ratio\n")
        for i, params in enumerate(params_ratios):
            f.write(f"{i+1}, {params:.4f}\n")
    
    # 推理时间曲线数据
    with open(os.path.join(save_dir, 'time_curve.txt'), 'w') as f:
        f.write("# Episode, Time_Ratio\n")
        for i, time in enumerate(time_ratios):
            f.write(f"{i+1}, {time:.4f}\n")
    
    # 综合性能数据
    with open(os.path.join(save_dir, 'performance_data.txt'), 'w') as f:
        f.write("# Episode, Reward, Accuracy_Ratio, Params_Ratio, Time_Ratio\n")
        for i in range(len(rewards)):
            f.write(f"{i+1}, {rewards[i]:.4f}, {accuracies[i]:.4f}, {params_ratios[i]:.4f}, {time_ratios[i]:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用RL训练网络压缩策略")
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='配置文件路径')
    args = parser.parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 训练压缩策略
    train_rl_compression(config) 