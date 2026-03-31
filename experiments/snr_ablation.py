# experiments/snr_ablation.py
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
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from contextlib import contextmanager

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

# Explicit imports for pickle
from models.bls import BLSClassifier
from models.moe_entropy_gate import MoEBLSEntropyResidualGate

# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------

def setup_logger(log_file):
    logger = logging.getLogger("AblationLogger")
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

@contextmanager
def temporary_model_config(model, **kwargs):
    """
    上下文管理器：临时修改模型属性，退出时自动还原。
    """
    original_state = {}
    for k in kwargs.keys():
        if hasattr(model, k):
            original_state[k] = getattr(model, k)
        
    for k, v in kwargs.items():
        if hasattr(model, k):
            setattr(model, k, v)

    try:
        yield model
    finally:
        for k, v in original_state.items():
            setattr(model, k, v)


def load_best_model(target_dir, logger):
    logger.info(f"[Info] Loading ablation target from: {target_dir}")

    # Load PCA
    try:
        with open(os.path.join(target_dir, "pca_transformers_baseline.pkl"), "rb") as f:
            pca_bl = pickle.load(f)
        with open(os.path.join(target_dir, "pca_transformers_moe.pkl"), "rb") as f:
            pca_moe = pickle.load(f)
    except Exception as e:
        logger.warning(f"[Warn] PCA load failed ({e}), using Identity.")
        pca_bl, pca_moe = None, None

    transformers = {"bl": pca_bl, "moe": pca_moe}

    # Load Models
    path_bls = os.path.join(target_dir, "run_bls_model.pkl")
    path_moe = os.path.join(target_dir, "run_flagship_moe_model.pkl")

    if not os.path.exists(path_bls) or not os.path.exists(path_moe):
        raise FileNotFoundError(f"Model pickle files missing in {target_dir}")

    with open(path_bls, "rb") as f:
        model_bls = pickle.load(f)
    with open(path_moe, "rb") as f:
        model_moe = pickle.load(f)

    return model_bls, model_moe, transformers

def prepare_data_batch(X_raw_clean, snr, cfg, transformers, crop_indices=None):
    """
    增加 crop_indices 参数，确保特征维度与训练时一致
    """
    # 1. Add Noise
    X_raw_noisy = add_awgn_iq_batch(X_raw_clean.copy(), snr_db=snr, per_sample=True)
    
    # 2. STFT (Apply Crop)
    X_stft = stft_features_from_raw(X_raw_noisy, cfg, crop_indices=crop_indices)
    
    # 3. Prepare MoE Input (List of Bands)
    X_bands_raw = np.array_split(X_stft, cfg.NUM_LOGICAL_EXPERTS, axis=1)
    X_bands_moe = []
    
    for i, band in enumerate(X_bands_raw):
        pca = transformers["moe"][i] if transformers["moe"] else None
        if pca:
            X_bands_moe.append(pca.transform(band))
        else:
            X_bands_moe.append(band)

    # 4. Prepare BLS Input (Concatenated)
    X_bands_bl = []
    for i, band in enumerate(X_bands_raw):
        pca = transformers["bl"][i] if transformers["bl"] else None
        if pca:
            X_bands_bl.append(pca.transform(band))
        else:
            X_bands_bl.append(band)
            
    X_bl_concat = np.concatenate(X_bands_bl, axis=1)

    return X_bands_moe, X_bl_concat


def plot_results(df, save_root, logger):
    plt.figure(figsize=(10, 7))
    sns.set_style("whitegrid")
    sns.set_context("paper", font_scale=1.4)

    # NPG (Nature Publishing Group) 顶刊配色
    palette = {
        "Ours (Entropy Gate)": "#E64B35",      
        "Ablation: Uniform (Average All)": "#4DBBD5",   
        "Ablation: Hard Switch (Top-1)": "#00A087", 
        "Baseline: Single BLS": "#3C5488"
    }
    
    markers = {
        "Ours (Entropy Gate)": "o",
        "Ablation: Uniform (Average All)": "X",
        "Ablation: Hard Switch (Top-1)": "^",
        "Baseline: Single BLS": "s"
    }

    sns.lineplot(
        data=df, x="SNR", y="Accuracy", hue="Method", style="Method",
        palette=palette, markers=markers, markersize=9, linewidth=2.5,
        dashes=False 
    )

    plt.ylabel("Accuracy", fontweight="bold", fontname='Times New Roman')
    plt.xlabel("SNR (dB)", fontweight="bold", fontname='Times New Roman')
    plt.xlim(-1, 26)
    plt.ylim(0.5, 1.02)
    
    # 图例：缩小、加框、新罗马字体
    plt.legend(loc="lower right", prop={'family': 'Times New Roman', 'size': 11}, 
               framealpha=0.9, edgecolor="black", frameon=True)
               
    plt.tight_layout()
    
    out_file = os.path.join(save_root, "ablation_study_plot.png")
    # 强制 DPI 为 1200
    plt.savefig(out_file, dpi=1200, bbox_inches='tight')
    logger.info(f"[Done] Ablation plot saved to: {out_file}")

# ---------------------------------------------------------
# Main Logic
# ---------------------------------------------------------
def run_ablation_experiment():
    cfg = get_moe_config()
    
    # 1. 自动寻找最新模型目录
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
    
    run_dirs = sorted([os.path.join(exp_path, d) for d in os.listdir(exp_path) if d.startswith("run")])
    if not run_dirs:
        print(f"[Error] No runs found in {exp_path}")
        return
    
    # 默认对最后一次 run 进行消融实验
    target_dir = run_dirs[-1]

    # 2. 生成外部结果保存文件夹
    now = datetime.now()
    eval_folder_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{cfg.NUM_CLASSES}Class_Ablation_Eval"
    
    if cfg.RESULTS_ROOT:
        output_root = os.path.join(cfg.RESULTS_ROOT, eval_folder_name)
    else:
        output_root = os.path.join(PROJECT_ROOT, "compare_results", eval_folder_name)
    os.makedirs(output_root, exist_ok=True)

    logger = setup_logger(os.path.join(output_root, "ablation_test.log"))
    logger.info("============================================================")
    logger.info(f"  Ablation Experiment: Gate Mechanisms")
    logger.info(f"  Evaluating Target: {target_dir}")
    logger.info(f"  Results saving to: {output_root}")
    logger.info("============================================================")

    # 3. Load Resources
    try:
        model_bls, model_moe, transformers = load_best_model(target_dir, logger)
    except Exception as e:
        logger.error(f"[Error] {e}")
        return

    # 4. Load Data & Calculate Crop Indices
    (tr_stft, tr_raw, tr_y), (te_stft, te_raw, te_y), _, _ = load_normal_data(cfg)
    crop_indices = calculate_global_crop_indices(tr_raw, cfg, energy_ratio=0.95)
    logger.info(f"[Info] Recovered Crop Indices for Testing: {crop_indices}")

    X_test_clean = te_raw
    Y_test = te_y.astype(int).ravel()

    # 5. Define Ablation Configurations
    ablation_configs = [
        {
            "name": "Ours (Entropy Gate)", 
            "params": {}, # 默认参数
            "model_type": "moe"
        },
        {
            "name": "Ablation: Uniform (Average All)",
            "params": {"gate_alpha": 0.0, "top_k": cfg.NUM_LOGICAL_EXPERTS}, 
            "model_type": "moe"
        },
        {
            "name": "Ablation: Hard Switch (Top-1)",
            "params": {"top_k": 1}, 
            "model_type": "moe"
        },
        {
            "name": "Baseline: Single BLS",
            "params": {},
            "model_type": "bls"
        }
    ]

    snr_range = range(0, 26, 5)
    results = []

    logger.info("\n[Start] Running Ablation Study (Runtime Surgery)...")
    
    for snr in tqdm(snr_range, desc="Evaluating SNR"):
        X_moe, X_bls = prepare_data_batch(X_test_clean, snr, cfg, transformers, crop_indices=crop_indices)

        for config in ablation_configs:
            name = config["name"]
            
            if config["model_type"] == "bls":
                preds = model_bls.predict(X_bls)
            
            elif config["model_type"] == "moe":
                with temporary_model_config(model_moe, **config["params"]):
                    preds = model_moe.predict(X_moe)
            
            acc = accuracy_score(Y_test, preds)
            results.append({"SNR": snr, "Method": name, "Accuracy": acc})

    # 6. Save & Plot
    df = pd.DataFrame(results)
    save_path = os.path.join(output_root, "ablation_final.csv")
    df.to_csv(save_path, index=False)
    
    plot_results(df, output_root, logger)
    logger.info("[DONE] Ablation Experiment completed successfully.")

if __name__ == "__main__":
    run_ablation_experiment()