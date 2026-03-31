# experiments/exp_hyperparam_sensitivity.py
import os
import sys
import copy
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------
# Formatting Settings (对齐顶刊质感)
# ---------------------------------------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config_moe_bls import get_moe_config
from utils.normal_data_utils import load_normal_data, stft_features_from_raw, calculate_global_crop_indices, compute_bandwidth_importance
from models.moe_entropy_gate import MoEBLSEntropyResidualGate
from experiments.exp_test import count_params, _calc_pca_metrics

def setup_logger(log_file):
    logger = logging.getLogger("SensitivityLogger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def run_single_setting(cfg, X_raw_tr, Y_tr, X_raw_te, Y_te, crop_indices):
    """
    运行单次超参数配置：重算 STFT -> 拆分频段 -> PCA -> 训练 -> 预测 -> 统计参数量
    """
    # 1. 重算 STFT (因为 NUM_LOGICAL_EXPERTS 改变会影响 Padding)
    X_stft_tr = stft_features_from_raw(X_raw_tr, cfg, crop_indices=crop_indices)
    X_stft_te = stft_features_from_raw(X_raw_te, cfg, crop_indices=crop_indices)

    # 2. 拆分频段
    num_bands = cfg.NUM_LOGICAL_EXPERTS
    X_tr_bands_raw = np.array_split(X_stft_tr, num_bands, axis=1)
    X_te_bands_raw = np.array_split(X_stft_te, num_bands, axis=1)

    band_weights = compute_bandwidth_importance(X_tr_bands_raw, method="variance")

    # 3. PCA 拟合与转换
    pca_list = []
    X_tr_bands, X_te_bands = [], []
    for i in range(num_bands):
        if cfg.USE_PCA:
            pca = PCA(n_components=cfg.PCA_DIM_MOE, svd_solver="randomized", random_state=cfg.GLOBAL_SEED + i)
            X_tr_bands.append(pca.fit_transform(X_tr_bands_raw[i]))
            X_te_bands.append(pca.transform(X_te_bands_raw[i]))
            pca_list.append(pca)
        else:
            X_tr_bands.append(X_tr_bands_raw[i])
            X_te_bands.append(X_te_bands_raw[i])
            pca_list.append(None)

    # 4. 初始化与训练模型
    model = MoEBLSEntropyResidualGate(
        input_dims_per_band=[b.shape[1] for b in X_tr_bands],
        num_classes=cfg.NUM_CLASSES,
        total_expert_feature_win_num=cfg.TOTAL_WIN_NUM,
        expert_feature_nodes_per_win=cfg.NODES_PER_WIN,
        total_expert_enhance_nodes=cfg.TOTAL_ENHANCE_NUM,
        expert_reg_lambda=cfg.REG_LAMBDA,
        gate_alpha=2.0, top_k=cfg.MOE_TOP_K,
        importance_scores=band_weights,
        random_state=cfg.GLOBAL_SEED
    )
    model.fit(X_tr_bands, Y_tr)

    # 5. 测试与统计
    preds = model.predict(X_te_bands)
    acc = accuracy_score(Y_te, preds)

    # 计算参数量
    pca_params, _ = _calc_pca_metrics(pca_list)
    model_params = count_params(model, "moe")
    total_params = pca_params + model_params

    return acc, total_params

def plot_2x2_sensitivity(df, output_root, logger):
    """
    绘制极具学术风格的 2x2 双Y轴图表 (统一图例)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    # 定义四个子图对应的参数名和横轴标签
    params = [
        ("NUM_LOGICAL_EXPERTS", "Number of Logical Experts ($N$)"),
        ("TOTAL_WIN_NUM", "Total Feature Windows ($N_{win}$)"),
        ("NODES_PER_WIN", "Nodes per Window ($N_{nodes}$)"),
        ("TOTAL_ENHANCE_NUM", "Total Enhance Nodes ($N_{enh}$)")
    ]

    # 用于存放统一图例的句柄和标签
    lines_for_legend = []
    labels_for_legend = []

    for i, (param_key, x_label) in enumerate(params):
        ax1 = axes[i]
        sub_df = df[df["Param_Type"] == param_key].sort_values("Param_Value")
        
        x_vals = sub_df["Param_Value"].values
        acc_vals = sub_df["Accuracy"].values
        param_vals = sub_df["Params"].values / 1e3 # 转换为 K (千)
        
        # 绘制主轴 Accuracy (柱状图/折线图)
        color_acc = "black" # NPG black
        ax1.set_xlabel(x_label, fontweight='bold', fontsize=12)
        ax1.set_ylabel("Accuracy", color=color_acc, fontweight='bold', fontsize=16)
        line1 = ax1.plot(x_vals, acc_vals, marker='o', color=color_acc, linewidth=2.5, markersize=8, label='Accuracy')
        ax1.tick_params(axis='y', labelcolor=color_acc)
        ax1.set_ylim(min(acc_vals)-0.02, max(acc_vals)+0.01)
        ax1.grid(alpha=0.3)
        ax1.set_xticks(x_vals) # 强制显示离散的 X 轴刻度
        
        # 绘制次轴 Parameters
        ax2 = ax1.twinx()
        color_param = "#3C5488" # NPG Blue
        ax2.set_ylabel("Parameters (K)", color=color_param, fontweight='bold', fontsize=16)
        line2 = ax2.plot(x_vals, param_vals, marker='s', color=color_param, linestyle='--', linewidth=2.5, markersize=8, label='Parameters (K)')
        ax2.tick_params(axis='y', labelcolor=color_param)
        
        # 提取图例信息（只在第一个子图提取一次即可，因为样式一致）
        if i == 0:
            lines_for_legend = line1 + line2
            labels_for_legend = [l.get_label() for l in lines_for_legend]

    # 紧凑布局
    plt.tight_layout()
    
    # 统一添加全局图例：放置在图表顶部外侧正中间，并设置为两列 (ncol=2)
    fig.legend(lines_for_legend, labels_for_legend, 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=2, 
               frameon=True, edgecolor='black', 
               prop={'family':'Times New Roman', 'size':12})
    
    # 微调顶部边距，防止全局图例与子图的 Title 或坐标轴重叠
    fig.subplots_adjust(top=0.92)

    save_path = os.path.join(output_root, "hyperparam_sensitivity_2x2.png")
    # bbox_inches='tight' 会确保保存的图片包含超出原 bounding box 的元素（比如我们放在顶部的图例）
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    logger.info(f"    -> Plot saved to {save_path}")

def main():
    base_cfg = get_moe_config()
    
    # 生成输出目录
    now = datetime.now()
    output_root = os.path.join(base_cfg.RESULTS_ROOT or PROJECT_ROOT, f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{base_cfg.NUM_CLASSES}Class_Sensitivity")
    os.makedirs(output_root, exist_ok=True)
    logger = setup_logger(os.path.join(output_root, "sensitivity.log"))
    logger.info("============================================================")
    logger.info("  Start Hyperparameter Sensitivity Analysis")
    logger.info("============================================================")

    # 1. 加载数据 (只取 Main 数据部分，且因为我们要动态算 STFT，这里拿到 RAW 即可)
    # 调用现有的 load_normal_data 会顺便把 base_cfg 下的 STFT 算了，但没关系，我们提取出 RAW 数据自己用
    (tr_stft, tr_raw, Y_tr), (te_stft, te_raw, Y_te), _, _ = load_normal_data(base_cfg)
    Y_tr = Y_tr.astype(int).ravel()
    Y_te = Y_te.astype(int).ravel()
    
    # 提前计算裁剪边界，保持全局一致
    crop_indices = calculate_global_crop_indices(tr_raw, base_cfg, energy_ratio=0.95)

    # 2. 定义搜索网格
    search_grids = {
        "NUM_LOGICAL_EXPERTS": [2, 4, 6, 8, 10, 12],
        "TOTAL_WIN_NUM": [4, 8, 12, 16, 20, 24],
        "NODES_PER_WIN": [5, 10, 15, 20, 25, 30],
        "TOTAL_ENHANCE_NUM": [400, 800, 1200, 1600, 2000, 2400]
    }

    results = []

    # 3. 开始执行控制变量实验
    for param_key, param_values in search_grids.items():
        logger.info(f"\n[Search] Testing {param_key} ...")
        for val in param_values:
            # 深拷贝配置，每次修改单个参数
            cfg = copy.deepcopy(base_cfg)
            setattr(cfg, param_key, val)
            
            # 【特殊逻辑】：如果专家数改变，强行保证 PCA 降维不超过基频段特征维度的限制
            if param_key == "NUM_LOGICAL_EXPERTS":
                pass # 现在的代码逻辑可以自适应
            
            try:
                acc, params = run_single_setting(cfg, tr_raw, Y_tr, te_raw, Y_te, crop_indices)
                logger.info(f"  -> {param_key}={val:4d} | Acc={acc:.4f} | Params={params/1e3:.1f}K")
                results.append({
                    "Param_Type": param_key,
                    "Param_Value": val,
                    "Accuracy": acc,
                    "Params": params
                })
            except Exception as e:
                logger.error(f"  -> {param_key}={val:4d} failed: {e}")

    # 4. 保存与绘图
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_root, "sensitivity_results.csv"), index=False)
    plot_2x2_sensitivity(df, output_root, logger)
    logger.info("\n[DONE] Sensitivity Analysis completed.")

if __name__ == "__main__":
    main()