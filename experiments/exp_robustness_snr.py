# experiments/exp_robustness_snr.py
import os
import sys
import pickle
import glob
import logging
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from tqdm import tqdm
from sklearn.metrics import accuracy_score

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
from utils.normal_data_utils import load_normal_data, stft_features_from_raw, calculate_global_crop_indices
from utils.noise_data_utils import add_awgn_iq_batch

# DL Models
from models.resnet_1d import ResNet18_1D
from models.cnn_1d import MobileNetV2_1D

# Explicit import to ensure pickle finds the class definition
from models.moe_entropy_gate import MoEBLSEntropyResidualGate


def setup_logger(log_file):
    logger = logging.getLogger("SNRLogger")
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


def load_models_and_pca(run_path, cfg, device, input_len, logger):
    """
    加载指定 Run 文件夹下的所有模型权重和 PCA 变换器
    """
    # 1. Load PCA
    try:
        with open(os.path.join(run_path, "pca_transformers_baseline.pkl"), "rb") as f:
            pca_list_bl = pickle.load(f)
        with open(os.path.join(run_path, "pca_transformers_moe.pkl"), "rb") as f:
            pca_list_moe = pickle.load(f)
    except FileNotFoundError:
        logger.warning(f"[Warn] PCA not found in {run_path}, skipping PCA transform.")
        pca_list_bl = [None] * cfg.NUM_LOGICAL_EXPERTS
        pca_list_moe = [None] * cfg.NUM_LOGICAL_EXPERTS

    transformers = {"bl": pca_list_bl, "moe": pca_list_moe}

    # 2. Load Models
    models = {}
    model_files = [
        ("BLS", "run_bls_model.pkl", "sklearn"),
        ("SFEBLN", "run_sfebln_model.pkl", "sklearn"),
        ("DER-BLMoE (Ours)", "run_flagship_moe_model.pkl", "sklearn"),
        ("ResNet18", "run_resnet18_best.pth", "pytorch_res"),
        ("MobileNetV2", "run_mobilenetv2_best.pth", "pytorch_mob"),
    ]

    for m_name, m_fname, m_type in model_files:
        m_path = os.path.join(run_path, m_fname)
        if not os.path.exists(m_path):
            continue

        if "pytorch" in m_type:
            if m_type == "pytorch_res":
                model = ResNet18_1D(num_classes=cfg.NUM_CLASSES).to(device)
            else:
                model = MobileNetV2_1D(num_classes=cfg.NUM_CLASSES, input_len=input_len).to(device)

            state_dict = torch.load(m_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict, strict=False)
            model.eval()
            models[m_name] = (model, m_type)
        else:
            with open(m_path, "rb") as f:
                models[m_name] = (pickle.load(f), "sklearn")

    return models, transformers


def prepare_noisy_inputs(X_raw_clean, snr, cfg, transformers, crop_indices=None):
    """
    生成带噪数据，并执行必要的预处理
    """
    X_raw_noisy = add_awgn_iq_batch(
        X_raw_clean,
        snr_db=snr,
        per_sample=True,
        seed=None,
    )
    X_dl_raw = X_raw_noisy.astype(np.float32)
    
    X_stft_noisy = stft_features_from_raw(X_raw_noisy, cfg, crop_indices=crop_indices)

    X_bands_stft = np.array_split(X_stft_noisy, cfg.NUM_LOGICAL_EXPERTS, axis=1)
    X_bands_moe = []
    X_bands_bl = []

    for i, band in enumerate(X_bands_stft):
        pca_moe = transformers["moe"][i] if transformers["moe"] else None
        if pca_moe is not None:
            X_bands_moe.append(pca_moe.transform(band))
        else:
            X_bands_moe.append(band)

        pca_bl = transformers["bl"][i] if transformers["bl"] else None
        if pca_bl is not None:
            X_bands_bl.append(pca_bl.transform(band))
        else:
            X_bands_bl.append(band)

    X_bl_concat = np.concatenate(X_bands_bl, axis=1).astype(np.float32)

    return {
        "raw": X_dl_raw,
        "moe_bands": X_bands_moe,
        "bls_concat": X_bl_concat
    }


def visualize_snr_academic(df, output_root, logger):
    plt.figure(figsize=(9, 7))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.5)

    # NPG (Nature Publishing Group) 顶刊配色
    palette = {
        "DER-BLMoE (Ours)": "#E64B35", 
        "SFEBLN": "#4DBBD5", 
        "BLS": "#00A087", 
        "ResNet18": "#3C5488", 
        "MobileNetV2": "#F39B7F"
    }
    markers = {"DER-BLMoE (Ours)": "o", "SFEBLN": "^", "BLS": "s", "ResNet18": "v", "MobileNetV2": "D"}
    linestyles = {"DER-BLMoE (Ours)": "-", "SFEBLN": "-.", "BLS": "--", "ResNet18": ":", "MobileNetV2": ":"}

    for m in df["Model"].unique():
        subset = df[df["Model"] == m].sort_values("SNR")
        plt.plot(subset["SNR"], subset["Mean"], label=m, 
                 color=palette.get(m, "black"), marker=markers.get(m, "o"), 
                 linestyle=linestyles.get(m, "-"), linewidth=2.5, markersize=8)
        plt.fill_between(subset["SNR"], subset["Mean"] - subset["CI95"], subset["Mean"] + subset["CI95"], 
                         color=palette.get(m, "black"), alpha=0.15)

    plt.xlabel("Signal-to-Noise Ratio (dB)", fontweight="bold", fontname='Times New Roman')
    plt.ylabel("Classification Accuracy", fontweight="bold", fontname='Times New Roman')
    plt.ylim(0, 1.05)
    plt.xlim(4, 26)
    
    # 图例：缩小、加框、新罗马字体
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 10}, 
               framealpha=0.9, edgecolor="black", frameon=True)
    
    plt.tight_layout()
    save_path = os.path.join(output_root, "robustness_snr_academic_plot.png")
    # 强制将生成图像 DPI 设置为 1200
    plt.savefig(save_path, dpi=1200)
    logger.info(f"[Output] SNR Robustness Chart saved to: {save_path}")


def run_snr_experiment():
    cfg = get_moe_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 自动定位最新的模型文件目录
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

    # 2. 生成外部结果保存文件夹
    now = datetime.now()
    eval_folder_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{cfg.NUM_CLASSES}Class_SNR_Eval"
    
    if cfg.RESULTS_ROOT:
        output_root = os.path.join(cfg.RESULTS_ROOT, eval_folder_name)
    else:
        output_root = os.path.join(PROJECT_ROOT, "compare_results", eval_folder_name)
    os.makedirs(output_root, exist_ok=True)

    logger = setup_logger(os.path.join(output_root, "snr_test.log"))
    logger.info("============================================================")
    logger.info(f"  Academic Robustness Experiment: SNR [5, 25] dB")
    logger.info(f"  Evaluating Checkpoints from: {latest_exp}")
    logger.info(f"  Results saving to: {output_root}")
    logger.info("============================================================")

    (tr_stft, tr_raw, tr_y), (te_stft, te_raw, te_y), _, _ = load_normal_data(cfg)
    
    crop_indices = calculate_global_crop_indices(tr_raw, cfg, energy_ratio=0.95)
    logger.info(f"[Info] Recovered Crop Indices: {crop_indices}")

    X_test_raw_clean = te_raw
    Y_test = te_y.astype(int).ravel()
    iq_len = X_test_raw_clean.shape[1] // 2
    X_test_clean_bkp = X_test_raw_clean.copy()

    run_dirs = sorted([os.path.join(exp_path, d) for d in os.listdir(exp_path) if d.startswith("run")])

    if not run_dirs:
        logger.error(f"[Error] No model runs found in {exp_path}")
        return

    snr_levels = list(range(5, 26, 5))
    all_results = []

    for run_path in tqdm(run_dirs, desc="Evaluating Runs"):
        run_name = os.path.basename(run_path)
        logger.info(f"Evaluating {run_name}...")
        
        models, transformers = load_models_and_pca(run_path, cfg, device, input_len=iq_len, logger=logger)

        for snr in snr_levels:
            inputs = prepare_noisy_inputs(X_test_clean_bkp, snr, cfg, transformers, crop_indices=crop_indices)

            for m_name, (model, m_type) in models.items():
                if m_name == "BLS":
                    preds = model.predict(inputs["bls_concat"])
                elif "MoE" in m_name:
                    preds = model.predict(inputs["moe_bands"])
                elif m_name == "SFEBLN":
                    preds = model.predict(inputs["raw"])
                elif "pytorch" in m_type:
                    X_in = inputs["raw"]
                    L = X_in.shape[1] // 2
                    X_reshaped = np.stack([X_in[:, :L], X_in[:, L:]], axis=1)
                    X_torch = torch.tensor(X_reshaped, dtype=torch.float32).to(device)
                    with torch.no_grad():
                        preds = model(X_torch).argmax(dim=1).cpu().numpy()
                else:
                    continue

                acc = accuracy_score(Y_test, preds)
                all_results.append({
                    "Run": run_name,
                    "SNR": snr,
                    "Model": m_name,
                    "Accuracy": acc,
                })

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(output_root, "robustness_snr_raw.csv"), index=False)

    def ci95(x):
        return 1.96 * np.std(x) / np.sqrt(len(x))

    df_agg = df.groupby(["Model", "SNR"])["Accuracy"].agg(Mean="mean", Std="std", CI95=ci95).reset_index()
    df_agg.to_csv(os.path.join(output_root, "robustness_snr_academic_agg.csv"), index=False)
    
    visualize_snr_academic(df_agg, output_root, logger)
    logger.info("[DONE] Academic SNR Experiment completed successfully.")


if __name__ == "__main__":
    run_snr_experiment()