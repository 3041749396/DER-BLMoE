# experiments/exp_model_evaluation.py
import os
import sys
import time
import pickle
import logging
from math import pi
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn

# ---------------------------------------------------------
# Formatting Settings 
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
from models.bls import BLSClassifier
from models.resnet_1d import ResNet18_1D
from models.cnn_1d import MobileNetV2_1D
from models.sfebln import SFEBLNClassifier
from utils.normal_data_utils import load_normal_data

def setup_logger(log_file):
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# ---------------------------------------------------------
# Helper Functions for Params and FLOPs 
# ---------------------------------------------------------
def _calc_pca_metrics(pca_list):
    params, flops = 0, 0
    if not pca_list: return 0, 0
    for pca in pca_list:
        if pca is not None and hasattr(pca, "components_"):
            n_comp, n_feat = pca.components_.shape
            params += (n_comp * n_feat) + n_feat
            flops += n_feat + (2 * n_comp * n_feat)
    return int(params), int(flops)

def _count_bls_params(model: BLSClassifier) -> int:
    total = 0
    W_feat, b_feat = getattr(model, "W_feature", None), getattr(model, "b_feature", None)
    if W_feat is not None and b_feat is not None:
        for w, b in zip(W_feat, b_feat): total += w.size + b.size
    W_enh, b_enh = getattr(model, "W_enhance", None), getattr(model, "b_enhance", None)
    if W_enh is not None and b_enh is not None: total += W_enh.size + b_enh.size
    if getattr(model, "beta", None) is not None: total += model.beta.size
    return int(total)

def count_params(model, model_type: str) -> int:
    if "pytorch" in model_type: return int(sum(p.numel() for p in model.parameters()))
    if model_type == "sfebln":
        cnt = sum(beta.size for beta in getattr(model, "Beta1OfEachWindow", []))
        if getattr(model, "weightOfEnhanceLayer", None) is not None: cnt += model.weightOfEnhanceLayer.size
        if getattr(model, "weightOfSPLayer", None) is not None: cnt += model.weightOfSPLayer.size
        if getattr(model, "OutputWeight", None) is not None: cnt += model.OutputWeight.size
        return int(cnt)
    if "moe" in model_type: return int(sum(_count_bls_params(exp) for exp in getattr(model, "experts", [])))
    if hasattr(model, "beta"): return _count_bls_params(model)
    return 0

def _calc_linear_flops(in_features, out_features): return 2 * in_features * out_features + out_features
def _calc_conv1d_flops(layer, input_len):
    out_len = (input_len + 2 * layer.padding[0] - layer.dilation[0] * (layer.kernel_size[0] - 1) - 1) // layer.stride[0] + 1
    weight_ops = layer.in_channels * layer.kernel_size[0] * layer.out_channels // layer.groups
    flops = 2 * weight_ops * out_len + (layer.out_channels * out_len if layer.bias is not None else 0)
    return flops, out_len

def _count_bls_flops(model: BLSClassifier, input_sample: np.ndarray) -> int:
    flops = 0
    if getattr(model, "W_feature", None):
        for W in model.W_feature: flops += _calc_linear_flops(input_sample.size, W.shape[1]) + 4 * W.shape[1]
    N_feature = model.feature_win_num * model.feature_nodes_per_win
    if getattr(model, "W_enhance", None) is not None:
        flops += _calc_linear_flops(N_feature, model.W_enhance.shape[1]) + 4 * model.W_enhance.shape[1]
    if getattr(model, "beta", None) is not None:
        flops += _calc_linear_flops(N_feature + model.enhance_nodes, model.num_classes)
    return int(flops)

def count_flops(model, model_type: str, input_sample) -> int:
    if "pytorch" in model_type:
        flops, current_len = 0, input_sample.shape[-1] // 2
        for m in model.modules():
            if isinstance(m, nn.Conv1d): f, current_len = _calc_conv1d_flops(m, current_len); flops += f
            elif isinstance(m, nn.Linear): flops += _calc_linear_flops(m.in_features, m.out_features)
            elif isinstance(m, nn.BatchNorm1d): flops += 4 * m.num_features * current_len
        return int(flops)
    if model_type == "sfebln":
        flops = sum(_calc_linear_flops(b.shape[0], b.shape[1]) for b in getattr(model, "Beta1OfEachWindow", []))
        for w_name in ["weightOfEnhanceLayer", "weightOfSPLayer"]:
            w = getattr(model, w_name, None)
            if w is not None: flops += _calc_linear_flops(w.shape[0], w.shape[1]) + 4 * w.shape[1]
        if getattr(model, "OutputWeight", None) is not None:
            flops += _calc_linear_flops(model.OutputWeight.shape[0], model.OutputWeight.shape[1])
        return int(flops)
    if "moe" in model_type:
        total_flops = sum(_count_bls_flops(exp, input_sample[i] if isinstance(input_sample, list) else input_sample) for i, exp in enumerate(getattr(model, "experts", [])))
        if hasattr(model, "num_classes"):
            total_flops += model.num_experts * (model.num_classes * 4 + 5 + model.num_classes * 2)
        return int(total_flops)
    if hasattr(model, "beta"): return _count_bls_flops(model, input_sample)
    return 0

# ---------------------------------------------------------
# Evaluation Core
# ---------------------------------------------------------
def evaluate_run(run_path, cfg, X_test_stft, X_test_raw, Y_test, device, logger):
    logger.info(f"\n[Eval] Directory: {run_path}")
    training_times = {}
    try:
        with open(os.path.join(run_path, "training_times.pkl"), "rb") as f:
            training_times = pickle.load(f)
    except FileNotFoundError:
        logger.warning("  [Warn] training_times.pkl not found.")

    try:
        with open(os.path.join(run_path, "pca_transformers_baseline.pkl"), "rb") as f: pca_list_bl = pickle.load(f)
        with open(os.path.join(run_path, "pca_transformers_moe.pkl"), "rb") as f: pca_list_moe = pickle.load(f)
    except FileNotFoundError:
        pca_list_bl = [None] * cfg.NUM_LOGICAL_EXPERTS
        pca_list_moe = [None] * cfg.NUM_LOGICAL_EXPERTS

    num_bands = cfg.NUM_LOGICAL_EXPERTS
    X_bands_stft = np.array_split(X_test_stft, num_bands, axis=1)

    t0 = time.perf_counter()
    X_bands_moe = [pca.transform(b) if pca else b for pca, b in zip(pca_list_moe, X_bands_stft)]
    moe_pca_lat = (time.perf_counter() - t0) * 1000.0 / max(1, X_test_stft.shape[0])
    moe_pca_params, moe_pca_flops = _calc_pca_metrics(pca_list_moe)

    t0 = time.perf_counter()
    X_bands_bl = [pca.transform(b) if pca else b for pca, b in zip(pca_list_bl, X_bands_stft)]
    X_bl_concat = np.concatenate(X_bands_bl, axis=1)
    bl_pca_lat = (time.perf_counter() - t0) * 1000.0 / max(1, X_test_stft.shape[0])
    bl_pca_params, bl_pca_flops = _calc_pca_metrics(pca_list_bl)

    models_to_eval = [
        ("BLS", "run_bls_model.pkl", "bls", X_bl_concat, X_bl_concat[0]),
        ("SFEBLN", "run_sfebln_model.pkl", "sfebln", X_test_raw, X_test_raw[0]),
        ("DER-BLMoE (Ours)", "run_flagship_moe_model.pkl", "moe_f", X_bands_moe, [b[0] for b in X_bands_moe]),
        ("ResNet18", "run_resnet18_best.pth", "pytorch_res", X_test_raw, X_test_raw[0]),
        ("MobileNetV2", "run_mobilenetv2_best.pth", "pytorch_mob", X_test_raw, X_test_raw[0]),
    ]
    
    results = {}
    run_reps = {}

    for m_name, m_file, m_type, X_input, X_sample_flops in models_to_eval:
        m_path = os.path.join(run_path, m_file)
        if not os.path.exists(m_path): continue

        if "pytorch" in m_type:
            model = ResNet18_1D(num_classes=cfg.NUM_CLASSES).to(device) if m_type == "pytorch_res" else MobileNetV2_1D(num_classes=cfg.NUM_CLASSES, input_len=X_test_raw.shape[1]//2).to(device)
            model.load_state_dict(torch.load(m_path, map_location=device, weights_only=True))
            model.eval()
        else:
            with open(m_path, "rb") as f: model = pickle.load(f)

        t0 = time.perf_counter()
        if "pytorch" in m_type:
            L = X_input.shape[1] // 2
            X_torch = torch.tensor(np.stack([X_input[:, :L], X_input[:, L:]], axis=1), dtype=torch.float32).to(device)
            with torch.no_grad(): 
                out = model(X_torch)
                preds = out.argmax(dim=1).cpu().numpy()
                reps = out.cpu().numpy()
            if device.type == "cuda": torch.cuda.synchronize()
        else:
            reps = model.predict_proba(X_input)
            preds = reps.argmax(axis=1)

        run_reps[m_name] = reps

        acc = accuracy_score(Y_test, preds)
        core_lat = (time.perf_counter() - t0) * 1000.0 / max(1, len(Y_test))
        core_params = count_params(model, m_type)
        core_flops = count_flops(model, m_type, X_sample_flops)

        overhead_lat = moe_pca_lat if m_type == "moe_f" else (bl_pca_lat if m_type == "bls" else 0.0)
        overhead_params = moe_pca_params if m_type == "moe_f" else (bl_pca_params if m_type == "bls" else 0)
        overhead_flops = moe_pca_flops if m_type == "moe_f" else (bl_pca_flops if m_type == "bls" else 0)
        
        train_t = training_times.get(m_name, 0.0)
        test_total_time = (core_lat + overhead_lat) * len(Y_test) / 1000.0
        total_time = train_t + test_total_time

        results[m_name] = {
            "Accuracy": acc,
            "Params_Core": core_params, "Params_Overhead": overhead_params, "Params_Total": core_params + overhead_params,
            "FLOPs_Core": core_flops, "FLOPs_Overhead": overhead_flops, "FLOPs_Total": core_flops + overhead_flops,
            "Latency_Core": core_lat, "Latency_Overhead": overhead_lat, "Latency_Total": core_lat + overhead_lat,
            "Training_Time": train_t, "Total_Time": total_time
        }
        logger.info(f"   - {m_name:14s} | Acc={acc:.4f} | TotalTime={total_time:.2f}s | Params(Core)={core_params/1e3:.1f}K | Lat(Core)={core_lat:.4f}ms")

    return results, run_reps

# ---------------------------------------------------------
# Plotting
# ---------------------------------------------------------
def plot_tsne_clusters(reps_dict, Y_true, output_root, logger):
    logger.info("[Plot] Generating t-SNE clustering plots...")
    
    max_samples = 1500
    if len(Y_true) > max_samples:
        np.random.seed(42)
        idx = np.random.choice(len(Y_true), max_samples, replace=False)
    else:
        idx = np.arange(len(Y_true))
        
    Y_sub = Y_true[idx]
    
    # NPG Colors
    colors = ['#E64B35', '#4DBBD5', '#00A087', '#3C5488', '#F39B7F', 
              '#8491B4', '#91D1C2', '#DC0000', '#7E6148', '#B09C85']
    
    num_models = len(reps_dict)
    cols = 3
    rows = int(np.ceil(num_models / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.5*rows))
    if num_models == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (m_name, rep) in enumerate(reps_dict.items()):
        ax = axes[i]
        rep_sub = rep[idx]
        
        tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
        emb = tsne.fit_transform(rep_sub)
        
        for c in range(10): 
            c_idx = (Y_sub == c)
            ax.scatter(emb[c_idx, 0], emb[c_idx, 1], color=colors[c%len(colors)], 
                       label=f'Class {c}' if i==0 else "", alpha=0.8, s=15, edgecolors='none')
        
        ax.set_xlabel("t-SNE Dim 1", fontname="Times New Roman", fontsize=10)
        ax.set_ylabel("t-SNE Dim 2", fontname="Times New Roman", fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=8)
        
        ax.text(0.05, 0.95, m_name, transform=ax.transAxes, 
                fontsize=11, fontweight='bold', fontname='Times New Roman',
                va='top', ha='left', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        if i == 0:
            ax.legend(prop={'family': 'Times New Roman', 'size': 8},
                      frameon=True, edgecolor='black', loc='best', markerscale=0.8)
                      
    for j in range(num_models, len(axes)):
        fig.delaxes(axes[j])
        
    plt.tight_layout()
    save_path = os.path.join(output_root, "final_tsne_clusters.png")
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    logger.info(f"    -> Saved t-SNE plot to {save_path}")


def plot_efficiency_radar_chart(df, output_root, logger):
    labels = df["Model"].values
    acc = df["Accuracy"].values
    eps = 1e-9
    eff_params = acc / (df["Params_Total"].values + eps)
    eff_flops = acc / (df["FLOPs_Total"].values + eps)
    eff_infer = acc / (df["Latency_Total"].values + eps)
    eff_total_t = 1.0 / (df["Total_Time"].values + eps) 

    def normalize_metric(values):
        _min, _max = np.min(values), np.max(values)
        val_processed = np.log10(values + 1e-12) if (_max / (_min + 1e-12) > 50) else values.copy()
        v_min, v_max = val_processed.min(), val_processed.max()
        if v_max - v_min < 1e-12: return np.ones_like(values)
        return (val_processed - v_min) / (v_max - v_min) * 0.9 + 0.1

    data = np.array([normalize_metric(m) for m in [acc, eff_params, eff_flops, eff_infer, eff_total_t]])
    categories = ['Accuracy', 'Param Efficiency\n(Acc/TotalParam)', 'FLOPs Efficiency\n(Acc/TotalFLOPs)', 'Inference Efficiency\n(Acc/TotalLat)', 'Total Speed\n(1/Time)']
    
    angles = [n / float(len(categories)) * 2 * pi for n in range(len(categories))]
    angles += angles[:1]
    palette = sns.color_palette("tab10", n_colors=len(labels))
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '<', '>']

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.grid(color='#AAAAAA', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.tick_params(axis='x', pad=25)
    ax.set_yticklabels([]) 

    for i, label in enumerate(labels):
        values = data[:, i].tolist()
        values += values[:1]
        lw, zorder = (4.0, 10) if "Ours" in label or "MoE" in label else (2.5, 5)
        ax.plot(angles, values, linewidth=lw, linestyle='-', color=palette[i], label=label, marker=markers[i%len(markers)], markersize=8, zorder=zorder)
        ax.fill(angles, values, color=palette[i], alpha=0.1)

    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    plt.xticks(angles[:-1], categories, size=10, fontweight='bold', color='#333333', fontname='Times New Roman')
    ax.set_rlabel_position(0)
    plt.yticks([0.1, 0.3, 0.5, 0.7, 0.9], [""]*5, color="grey", size=8)
    plt.ylim(0, 1.05)

    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), prop={'family':'Times New Roman', 'size':8}, frameon=True, shadow=False, edgecolor='black')
    
    save_path = os.path.join(output_root, "final_radar_efficiency.png")
    plt.tight_layout()
    plt.savefig(save_path, dpi=1200, bbox_inches='tight')
    logger.info(f"[Output] Efficiency Radar Chart saved to: {save_path}")


def save_and_plot_results(all_data, output_root, logger):
    if not all_data: return
    rows = [{"Run": r, "Model": m, **v} for r, res in all_data for m, v in res.items()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(output_root, "final_model_comparison_stacked.csv"), index=False)

    model_order = [m for m in ["BLS", "SFEBLN", "ResNet18", "MobileNetV2", "DER-BLMoE (Ours)"] if m in df["Model"].unique()]
    df_mean = df.groupby("Model", as_index=False).mean(numeric_only=True).set_index("Model").reindex(model_order).reset_index()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    ax_acc = axes[0]
    sns.barplot(data=df_mean, x="Model", y="Accuracy", hue="Model", legend=False, ax=ax_acc, palette="viridis")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.set_ylabel("Classification Accuracy", fontweight='bold', fontname='Times New Roman')
    for i, m in enumerate(model_order):
        if m in df_mean["Model"].values:
            val = df_mean.loc[df_mean["Model"] == m, "Accuracy"].values[0]
            ax_acc.text(i, val + 0.01, f"{val:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold', fontname='Times New Roman')

    def plot_stacked(ax, col_core, col_overhead, ylabel, unit_scale=1, unit_suffix=""):
        indices = range(len(model_order))
        c_val, o_val = df_mean[col_core].values / unit_scale, df_mean[col_overhead].values / unit_scale
        ax.bar(indices, c_val, label='Trainable/Core', color='#2c7fb8', alpha=0.9)
        ax.bar(indices, o_val, bottom=c_val, label='Static/Overhead', color='#a1dab4', alpha=0.9)
        ax.set_ylabel(ylabel, fontweight='bold', fontname='Times New Roman')
        ax.set_xticks(indices)
        ax.set_xticklabels(model_order, rotation=30)
        ax.legend(frameon=True, edgecolor='black', prop={'family': 'Times New Roman', 'size': 16})
        for i, (c, o) in enumerate(zip(c_val, o_val)):
            total = c + o
            if total > 0:
                ax.text(i, total * 1.02, f"{total:.1f}{unit_suffix}", ha='center', va='bottom', fontsize=8, fontweight='bold', fontname='Times New Roman')

    plot_stacked(axes[1], "Params_Core", "Params_Overhead", "Params (x1000)", 1e3, "K")
    plot_stacked(axes[2], "Latency_Core", "Latency_Overhead", "Latency (ms)", 1.0, "ms")
    plot_stacked(axes[3], "FLOPs_Core", "FLOPs_Overhead", "FLOPs (x10^6)", 1e6, "M")

    ax_train = axes[4]
    sns.barplot(data=df_mean, x="Model", y="Total_Time", hue="Model", legend=False, ax=ax_train, palette="magma")
    ax_train.set_ylabel("Total Time (seconds)", fontweight='bold', fontname='Times New Roman')
    ax_train.set_xticklabels(model_order, rotation=30)
    for i, m in enumerate(model_order):
        if m in df_mean["Model"].values:
            val = df_mean.loc[df_mean["Model"] == m, "Total_Time"].values[0]
            ax_train.text(i, val + (df_mean["Total_Time"].max()*0.02), f"{val:.2f}s", ha='center', va='bottom', fontsize=8, fontweight='bold', fontname='Times New Roman')
            
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, "final_stacked_bars.png"), dpi=1200)

    plot_efficiency_radar_chart(df_mean, output_root, logger)
    logger.info("\n[Summary - Detailed Breakdown]")
    logger.info("\n" + df_mean[["Model", "Accuracy", "Params_Total", "FLOPs_Total", "Latency_Total", "Total_Time"]].to_string(float_format=lambda x: "{:.4f}".format(x)))

def main():
    cfg = get_moe_config()
    
    # 1. 寻找 checkpoints 目录下最新的模型权重文件夹
    base_dir = os.path.join(PROJECT_ROOT, "checkpoints")
    if not os.path.exists(base_dir):
        print(f"Checkpoints dir not found: {base_dir}")
        return
        
    exp_dirs = [d for d in os.listdir(base_dir) if "Class" in d and os.path.isdir(os.path.join(base_dir, d))]
    if not exp_dirs:
        print("No experiment directories found in checkpoints/")
        return
    
    latest_exp = max(exp_dirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    exp_path = os.path.join(base_dir, latest_exp)

    # 2. 在外层 compare_results (即 cfg.RESULTS_ROOT) 目录中生成带有时间戳的专属结果保存路径
    now = datetime.now()
    eval_folder_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{cfg.NUM_CLASSES}Class_Eval"
    
    if cfg.RESULTS_ROOT:
        output_root = os.path.join(cfg.RESULTS_ROOT, eval_folder_name)
    else:
        output_root = os.path.join(PROJECT_ROOT, "compare_results", eval_folder_name)
        
    os.makedirs(output_root, exist_ok=True)

    # 3. 日志及后续产出都放在这个外部目录下
    logger = setup_logger(os.path.join(output_root, "test.log"))
    logger.info("============================================================")
    logger.info(f"  Evaluating Checkpoints from: {latest_exp}")
    logger.info(f"  Results and Logs saving to: {output_root}")
    logger.info("============================================================")

    _, (te_stft, te_raw, te_y), _, _ = load_normal_data(cfg)
    Y_test = te_y.astype(int).ravel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = sorted([d for d in os.listdir(exp_path) if d.startswith("run")])
    all_results = []
    final_reps = None

    for d in run_dirs:
        run_path = os.path.join(exp_path, d)
        if os.path.isdir(run_path):
            res, reps = evaluate_run(run_path, cfg, te_stft, te_raw, Y_test, device, logger)
            all_results.append((d, res))
            final_reps = reps  

    save_and_plot_results(all_results, output_root, logger)
    
    if final_reps is not None:
        plot_tsne_clusters(final_reps, Y_test, output_root, logger)

if __name__ == "__main__":
    main()