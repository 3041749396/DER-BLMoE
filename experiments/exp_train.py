# experiments/exp_train_and_save_models.py
import os
import sys
import time
import pickle
import shutil
import logging
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
from sklearn.decomposition import PCA

# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from configs.config_moe_bls import get_moe_config, MoeBLSConfig
from utils.normal_data_utils import (
    load_normal_data,
    compute_bandwidth_importance,
)
from models.bls import BLSClassifier
from models.sfebln import SFEBLNClassifier
from models.moe_entropy_gate import MoEBLSEntropyResidualGate
from models.dl_trainer import PyTorchTrainer


def setup_logger(log_file):
    logger = logging.getLogger("TrainLogger")
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
# PCA Helpers
# ---------------------------------------------------------
def perform_pca_fit(X_bands_train, n_components, seed):
    X_train_pca = []
    pca_list = []
    for i, tr in enumerate(X_bands_train):
        if isinstance(n_components, int) and tr.shape[1] <= n_components:
            X_train_pca.append(tr)
            pca_list.append(None)
            continue

        if isinstance(n_components, float) and 0 < n_components < 1:
            solver = "full"
        else:
            solver = "randomized"

        pca = PCA(n_components=n_components, svd_solver=solver, random_state=seed + i)
        X_train_pca.append(pca.fit_transform(tr))
        pca_list.append(pca)
    return X_train_pca, pca_list

def apply_pca_transform(X_bands, pca_list):
    X_out = []
    for i, band in enumerate(X_bands):
        if pca_list[i] is not None:
            X_out.append(pca_list[i].transform(band))
        else:
            X_out.append(band)
    return X_out


# ---------------------------------------------------------
# IO Helpers
# ---------------------------------------------------------
def save_model_pickle(obj, filepath, logger):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(obj, f)
    logger.info(f"    -> Saved: {os.path.basename(filepath)}")

def save_model_torch(model, filepath, logger):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    logger.info(f"    -> Saved: {os.path.basename(filepath)}")

def split_into_bands_stft(X_stft: np.ndarray, num_bands: int):
    if num_bands <= 1:
        return [X_stft]
    return np.array_split(X_stft, num_bands, axis=1)


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------
def run_training_pipeline(
    cfg: MoeBLSConfig,
    run_idx: int,
    base_seed: int,
    data_train: Tuple[np.ndarray, np.ndarray, np.ndarray],
    data_test: Tuple[np.ndarray, np.ndarray, np.ndarray],
    save_root: str,
    logger: logging.Logger
):
    seed = base_seed + run_idx
    run_dir = os.path.join(save_root, f"run{run_idx + 1:02d}")
    os.makedirs(run_dir, exist_ok=True)
    
    logger.info(f"\n[{run_idx + 1:02d}] Starting Training Run (Seed={seed}) .")
    training_times = {}

    X_stft_all, X_raw_all, Y_all = data_train
    Y_all = Y_all.astype(int).ravel()

    num_samples = X_stft_all.shape[0]
    rng = np.random.RandomState(seed)
    idx = rng.permutation(num_samples)

    split = int(0.8 * num_samples)
    tr_idx, val_idx = idx[:split], idx[split:]

    X_stft_tr, X_stft_val = X_stft_all[tr_idx], X_stft_all[val_idx]
    X_raw_tr, X_raw_val = X_raw_all[tr_idx], X_raw_all[val_idx]
    Y_tr, Y_val = Y_all[tr_idx], Y_all[val_idx]

    logger.info(f"  [Split] Train={X_stft_tr.shape[0]}  Val={X_stft_val.shape[0]}")

    num_bands = cfg.NUM_LOGICAL_EXPERTS
    X_stft_tr_bands = split_into_bands_stft(X_stft_tr, num_bands)
    X_stft_val_bands = split_into_bands_stft(X_stft_val, num_bands)

    band_weights = compute_bandwidth_importance(X_stft_tr_bands, method="variance")
    logger.info(f"  [Gate] Band importance weights: {band_weights}")

    if cfg.USE_PCA:
        logger.info("  [Data-STFT] Fitting PCA transformers.")
        X_tr_bands_bl, pca_list_bl = perform_pca_fit(X_stft_tr_bands, cfg.PCA_DIM_BASELINE, seed + 100)
        X_tr_bands_moe, pca_list_moe = perform_pca_fit(X_stft_tr_bands, cfg.PCA_DIM_MOE, seed + 200)
        X_val_bands_moe = apply_pca_transform(X_stft_val_bands, pca_list_moe)
        X_tr_bl_input = np.concatenate(X_tr_bands_bl, axis=1)
    else:
        pca_list_bl = [None] * num_bands
        pca_list_moe = [None] * num_bands
        X_tr_bands_moe = X_stft_tr_bands
        X_val_bands_moe = X_stft_val_bands
        X_tr_bl_input = np.concatenate(X_stft_tr_bands, axis=1)

    save_model_pickle(pca_list_bl, os.path.join(run_dir, "pca_transformers_baseline.pkl"), logger)
    save_model_pickle(pca_list_moe, os.path.join(run_dir, "pca_transformers_moe.pkl"), logger)

    # BLS
    logger.info("  [Train] BLS (Standard - STFT).")
    bls = BLSClassifier(
        input_dim=X_tr_bl_input.shape[1], num_classes=cfg.NUM_CLASSES,
        feature_win_num=cfg.BLS_FEATURE_WIN_NUM, feature_nodes_per_win=cfg.BLS_FEATURE_NODES_PER_WIN,
        enhance_nodes=cfg.BLS_ENHANCE_NODES, reg_lambda=cfg.BLS_REG_LAMBDA, random_state=seed,
    )
    t0 = time.time()
    bls.fit(X_tr_bl_input, Y_tr)
    t_bls = time.time() - t0
    training_times["BLS"] = t_bls
    logger.info(f"    -> Training Time: {t_bls:.4f}s")
    save_model_pickle(bls, os.path.join(run_dir, "run_bls_model.pkl"), logger)

    # SFEBLN
    logger.info("  [Train] SFEBLN (Unified - RAW).")
    sfebln = SFEBLNClassifier.from_config(cfg, random_state=seed)
    t0 = time.time()
    sfebln.fit(X_raw_tr, Y_tr)
    t_sfebln = time.time() - t0
    training_times["SFEBLN"] = t_sfebln
    logger.info(f"    -> Training Time: {t_sfebln:.4f}s")
    save_model_pickle(sfebln, os.path.join(run_dir, "run_sfebln_model.pkl"), logger)

    # MoE-BLS
    logger.info("  [Train] Flagship MoE-BLS (Entropy Gate - Simplified).")
    moe_flagship = MoEBLSEntropyResidualGate(
        input_dims_per_band=[band.shape[1] for band in X_tr_bands_moe],
        num_classes=cfg.NUM_CLASSES, total_expert_feature_win_num=cfg.TOTAL_WIN_NUM,
        expert_feature_nodes_per_win=cfg.NODES_PER_WIN, total_expert_enhance_nodes=cfg.TOTAL_ENHANCE_NUM,
        expert_reg_lambda=cfg.REG_LAMBDA, gate_alpha=2.0, top_k=cfg.MOE_TOP_K,
        importance_scores=band_weights, random_state=seed,
    )
    t0 = time.time()
    moe_flagship.fit(X_tr_bands_moe, Y_tr)
    t_moe = time.time() - t0
    training_times["DER-BLMoE (Ours)"] = t_moe
    logger.info(f"    -> Training Time: {t_moe:.4f}s")
    save_model_pickle(moe_flagship, os.path.join(run_dir, "run_flagship_moe_model.pkl"), logger)

    # DL Models
    dl_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"  [Train] Deep Learning Models (RAW) on {dl_device}.")

    trainer_resnet = PyTorchTrainer(model_name="resnet18", input_len=X_raw_tr.shape[1], num_classes=cfg.NUM_CLASSES, device=dl_device)
    t0 = time.time()
    trainer_resnet.fit(X_raw_tr, Y_tr, X_raw_val, Y_val)
    t_resnet = time.time() - t0
    training_times["ResNet18"] = t_resnet
    logger.info(f"    -> ResNet18 Training Time: {t_resnet:.4f}s")
    save_model_torch(trainer_resnet.model, os.path.join(run_dir, "run_resnet18_best.pth"), logger)

    trainer_mobile = PyTorchTrainer(model_name="mobilenetv2", input_len=X_raw_tr.shape[1], num_classes=cfg.NUM_CLASSES, device=dl_device)
    t0 = time.time()
    trainer_mobile.fit(X_raw_tr, Y_tr, X_raw_val, Y_val)
    t_mobile = time.time() - t0
    training_times["MobileNetV2"] = t_mobile
    logger.info(f"    -> MobileNetV2 Training Time: {t_mobile:.4f}s")
    save_model_torch(trainer_mobile.model, os.path.join(run_dir, "run_mobilenetv2_best.pth"), logger)

    save_model_pickle(training_times, os.path.join(run_dir, "training_times.pkl"), logger)
    logger.info(f"[{run_idx + 1:02d}] Run Completed.")

def main():
    cfg = get_moe_config()
    
    now = datetime.now()
    exp_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{cfg.NUM_CLASSES}Class"
    save_root = os.path.join(PROJECT_ROOT, "checkpoints", exp_name)
    os.makedirs(save_root, exist_ok=True)

    logger = setup_logger(os.path.join(save_root, "train.log"))
    logger.info("============================================================")
    logger.info(f"  Experiment Start: {exp_name}")
    logger.info("============================================================")

    train_main, test_main, _, _ = load_normal_data(cfg)
    mc_times = getattr(cfg, "INF_BENCH_MC", 1)
    base_seed = getattr(cfg, "GLOBAL_SEED", 2025)

    logger.info(f"[INFO] Total Runs: {mc_times}")

    for run_idx in range(mc_times):
        run_training_pipeline(cfg, run_idx, base_seed, train_main, test_main, save_root, logger)

    logger.info("\n[DONE] All training runs completed.")

if __name__ == "__main__":
    main()