# experiments/exp_test.py
import os
import sys
import pickle
import logging
import time
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

import torch
import torch.nn as nn

# ---------------------------------------------------------
# Formatting Settings
# ---------------------------------------------------------
# Do not force Times New Roman globally.
# Only the confusion-matrix plotting function uses Times New Roman.
# Suppress repeated matplotlib font lookup messages when Times New Roman is not installed.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

plt.rcParams.update(plt.rcParamsDefault)
plt.rcParams["axes.unicode_minus"] = False

CM_FONT = "Times New Roman"

# ---------------------------------------------------------
# Path Setup
# ---------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from configs.config_moe_bls import get_moe_config
except ModuleNotFoundError:
    from config.config_moe_bls import get_moe_config

from models.bls import BLSClassifier
from models.resnet_1d import ResNet18_1D
from models.sfebln import SFEBLNClassifier
from models.ajcn_official_adapter import (
    build_official_ajcn_model_from_info,
    raw_iq_to_ajcn_tensor,
    count_ajcn_params,
    estimate_ajcn_flops,
)
from utils.normal_data_utils import load_normal_data


# ---------------------------------------------------------
# Basic Utilities
# ---------------------------------------------------------
def setup_logger(log_file):
    logger = logging.getLogger("TestLogger")
    logger.setLevel(logging.INFO)
    if logger.hasHandlers():
        logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def safe_name(name: str) -> str:
    return (
        name.replace(" ", "_")
            .replace("/", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
    )


def t_critical_95(n: int) -> float:
    """Two-sided 95% t critical value."""
    if n <= 1:
        return 0.0

    df = n - 1
    if student_t is not None:
        return float(student_t.ppf(0.975, df=df))

    # Fallback table for common Monte Carlo trial counts.
    t_table = {
        1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
        6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
        16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
        21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
        26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        40: 2.021, 60: 2.000, 120: 1.980,
    }

    if df in t_table:
        return t_table[df]
    if df < 40:
        return t_table[30]
    if df < 60:
        return t_table[40]
    if df < 120:
        return t_table[60]
    return 1.960


def summarize_series_ci(values):
    """
    Summarize one metric over repeated runs.
    CI95 is the two-sided 95% confidence interval half-width:
        t_{0.975, n-1} * std / sqrt(n).
    """
    values = pd.Series(values).dropna().astype(float)
    n = int(len(values))

    if n == 0:
        return {
            "N": 0,
            "Mean": np.nan,
            "Std": np.nan,
            "CI95_HalfWidth": np.nan,
            "CI95_Lower": np.nan,
            "CI95_Upper": np.nan,
            "Mean±Std": "nan ± nan",
            "Mean±95CI": "nan ± nan",
        }

    mean = float(values.mean())
    std = float(values.std(ddof=1)) if n > 1 else 0.0
    ci95 = float(t_critical_95(n) * std / np.sqrt(n)) if n > 1 else 0.0

    return {
        "N": n,
        "Mean": mean,
        "Std": std,
        "CI95_HalfWidth": ci95,
        "CI95_Lower": mean - ci95,
        "CI95_Upper": mean + ci95,
        "Mean±Std": f"{mean:.6g} ± {std:.6g}",
        "Mean±95CI": f"{mean:.6g} ± {ci95:.6g}",
    }


def mean_std_ci_table(df, group_col, value_cols):
    rows = []
    for group_name, sub in df.groupby(group_col):
        row = {group_col: group_name}
        for col in value_cols:
            if col not in sub.columns:
                continue
            stats = summarize_series_ci(sub[col])
            row[f"{col}_N"] = stats["N"]
            row[f"{col}_Mean"] = stats["Mean"]
            row[f"{col}_Std"] = stats["Std"]
            row[f"{col}_CI95_HalfWidth"] = stats["CI95_HalfWidth"]
            row[f"{col}_CI95_Lower"] = stats["CI95_Lower"]
            row[f"{col}_CI95_Upper"] = stats["CI95_Upper"]
            row[f"{col}_Mean±Std"] = stats["Mean±Std"]
            row[f"{col}_Mean±95CI"] = stats["Mean±95CI"]
        rows.append(row)
    return pd.DataFrame(rows)


# Keep the old function name for compatibility.
def mean_std_table(df, group_col, value_cols):
    return mean_std_ci_table(df, group_col, value_cols)


# ---------------------------------------------------------
# Params and FLOPs
# ---------------------------------------------------------
def _calc_pca_metrics(pca_list):
    params, flops = 0, 0
    if not pca_list:
        return 0, 0
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
        for w, b in zip(W_feat, b_feat):
            total += w.size + b.size
    W_enh, b_enh = getattr(model, "W_enhance", None), getattr(model, "b_enhance", None)
    if W_enh is not None and b_enh is not None:
        total += W_enh.size + b_enh.size
    if getattr(model, "beta", None) is not None:
        total += model.beta.size
    return int(total)


def count_params(model, model_type: str) -> int:
    if "pytorch" in model_type:
        return int(sum(p.numel() for p in model.parameters()))

    if model_type == "sfebln":
        cnt = sum(beta.size for beta in getattr(model, "Beta1OfEachWindow", []))
        if getattr(model, "weightOfEnhanceLayer", None) is not None:
            cnt += model.weightOfEnhanceLayer.size
        if getattr(model, "weightOfSPLayer", None) is not None:
            cnt += model.weightOfSPLayer.size
        if getattr(model, "OutputWeight", None) is not None:
            cnt += model.OutputWeight.size
        return int(cnt)

    if "moe" in model_type:
        return int(sum(_count_bls_params(exp) for exp in getattr(model, "experts", [])))

    if hasattr(model, "beta"):
        return _count_bls_params(model)

    return 0


def _calc_linear_flops(in_features, out_features):
    return 2 * in_features * out_features + out_features


def _calc_conv1d_flops(layer, input_len):
    out_len = (
        input_len
        + 2 * layer.padding[0]
        - layer.dilation[0] * (layer.kernel_size[0] - 1)
        - 1
    ) // layer.stride[0] + 1
    weight_ops = layer.in_channels * layer.kernel_size[0] * layer.out_channels // layer.groups
    flops = 2 * weight_ops * out_len + (layer.out_channels * out_len if layer.bias is not None else 0)
    return flops, out_len


def _count_bls_flops(model: BLSClassifier, input_sample: np.ndarray) -> int:
    flops = 0
    if getattr(model, "W_feature", None):
        for W in model.W_feature:
            flops += _calc_linear_flops(input_sample.size, W.shape[1]) + 4 * W.shape[1]

    n_feature = model.feature_win_num * model.feature_nodes_per_win
    if getattr(model, "W_enhance", None) is not None:
        flops += _calc_linear_flops(n_feature, model.W_enhance.shape[1]) + 4 * model.W_enhance.shape[1]

    if getattr(model, "beta", None) is not None:
        flops += _calc_linear_flops(n_feature + model.enhance_nodes, model.num_classes)

    return int(flops)


def count_flops(model, model_type: str, input_sample) -> int:
    if "pytorch" in model_type:
        flops, current_len = 0, input_sample.shape[-1] // 2
        for m in model.modules():
            if isinstance(m, nn.Conv1d):
                f, current_len = _calc_conv1d_flops(m, current_len)
                flops += f
            elif isinstance(m, nn.Linear):
                flops += _calc_linear_flops(m.in_features, m.out_features)
            elif isinstance(m, nn.BatchNorm1d):
                flops += 4 * m.num_features * current_len
        return int(flops)

    if model_type == "sfebln":
        flops = sum(_calc_linear_flops(b.shape[0], b.shape[1]) for b in getattr(model, "Beta1OfEachWindow", []))
        for w_name in ["weightOfEnhanceLayer", "weightOfSPLayer"]:
            w = getattr(model, w_name, None)
            if w is not None:
                flops += _calc_linear_flops(w.shape[0], w.shape[1]) + 4 * w.shape[1]
        if getattr(model, "OutputWeight", None) is not None:
            flops += _calc_linear_flops(model.OutputWeight.shape[0], model.OutputWeight.shape[1])
        return int(flops)

    if "moe" in model_type:
        total_flops = sum(
            _count_bls_flops(exp, input_sample[i] if isinstance(input_sample, list) else input_sample)
            for i, exp in enumerate(getattr(model, "experts", []))
        )
        if hasattr(model, "num_classes"):
            total_flops += model.num_experts * (model.num_classes * 4 + 5 + model.num_classes * 2)
        return int(total_flops)

    if hasattr(model, "beta"):
        return _count_bls_flops(model, input_sample)

    return 0


def load_state_dict_safe(model, path, device):
    try:
        state = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(path, map_location=device)
    model.load_state_dict(state)


def load_training_times(run_path: str) -> dict:
    """Load model construction/training times saved during training."""
    path = os.path.join(run_path, "training_times.pkl")
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def get_model_construction_time(training_times: dict, model_name: str) -> float:
    """Robustly match model names in training_times.pkl."""
    candidates = [model_name]
    alias = {
        "BLS": ["BLS", "Baseline BLS", "run_bls_model"],
        "SFEBLN": ["SFEBLN", "SFE-BLN", "run_sfebln_model"],
        "DER-BLMoE (Ours)": ["DER-BLMoE (Ours)", "DER-BLMoE", "Ours", "run_flagship_moe_model"],
        "ResNet18": ["ResNet18", "ResNet-18", "ResNet-18-1D", "FineZero", "run_resnet18_best"],
        "AJCN": ["AJCN", "AJCN-1D", "run_ajcn_best"],
    }
    candidates.extend(alias.get(model_name, []))

    for key in candidates:
        if key in training_times:
            try:
                return float(training_times[key])
            except Exception:
                pass

    # Case-insensitive fallback.
    lower_map = {str(k).lower(): v for k, v in training_times.items()}
    for key in candidates:
        if str(key).lower() in lower_map:
            try:
                return float(lower_map[str(key).lower()])
            except Exception:
                pass

    return np.nan


def sync_if_cuda(device):
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------
def normalized_confusion_matrix(y_true, y_pred, num_classes):
    labels = np.arange(num_classes)
    cm = confusion_matrix(y_true, y_pred, labels=labels).astype(np.float64)
    denom = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(denom, 1.0), where=denom != 0)
    return cm_norm


def plot_confusion_matrix_mean(cm_mean, save_path, title):
    plt.figure(figsize=(7.5, 6.2))
    ax = sns.heatmap(
        cm_mean,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=np.arange(cm_mean.shape[1]),
        yticklabels=np.arange(cm_mean.shape[0]),
        cbar_kws={"label": "Mean Row-normalized Value"},
        annot_kws={
            "fontsize": 7,
            "fontname": CM_FONT,
        },
    )

    ax.set_xlabel("Predicted Label", fontweight="bold", fontname=CM_FONT)
    ax.set_ylabel("True Label", fontweight="bold", fontname=CM_FONT)
    ax.set_title(title, fontweight="bold", fontname=CM_FONT)

    for tick in ax.get_xticklabels():
        tick.set_fontname(CM_FONT)
        tick.set_fontweight("bold")
    for tick in ax.get_yticklabels():
        tick.set_fontname(CM_FONT)
        tick.set_fontweight("bold")

    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_fontname(CM_FONT)
    cbar.ax.yaxis.label.set_fontweight("bold")
    for tick in cbar.ax.get_yticklabels():
        tick.set_fontname(CM_FONT)

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# Test Data Preparation
# ---------------------------------------------------------
def prepare_test_inputs(run_path, cfg, X_test_stft):
    try:
        with open(os.path.join(run_path, "pca_transformers_baseline.pkl"), "rb") as f:
            pca_list_bl = pickle.load(f)
        with open(os.path.join(run_path, "pca_transformers_moe.pkl"), "rb") as f:
            pca_list_moe = pickle.load(f)
    except FileNotFoundError:
        pca_list_bl = [None] * cfg.NUM_LOGICAL_EXPERTS
        pca_list_moe = [None] * cfg.NUM_LOGICAL_EXPERTS

    x_bands_stft = np.array_split(X_test_stft, cfg.NUM_LOGICAL_EXPERTS, axis=1)

    x_bands_moe = [pca.transform(b) if pca is not None else b for pca, b in zip(pca_list_moe, x_bands_stft)]
    x_bands_bl = [pca.transform(b) if pca is not None else b for pca, b in zip(pca_list_bl, x_bands_stft)]
    x_bl_concat = np.concatenate(x_bands_bl, axis=1)

    moe_pca_params, moe_pca_flops = _calc_pca_metrics(pca_list_moe)
    bl_pca_params, bl_pca_flops = _calc_pca_metrics(pca_list_bl)

    return {
        "X_bands_moe": x_bands_moe,
        "X_bl_concat": x_bl_concat,
        "moe_overhead": (moe_pca_params, moe_pca_flops),
        "bl_overhead": (bl_pca_params, bl_pca_flops),
    }


# ---------------------------------------------------------
# Expert-level Diagnostics
# ---------------------------------------------------------
def build_expert_sample_records(run_name, y_true, details):
    """
    Each row corresponds to one test sample under one expert.
    Entropy is the Shannon entropy computed from that expert's own softmax output.
    It is NOT residual entropy and NOT normalized by the mean entropy.
    """
    entropies = details["entropies"]
    expert_preds = details["expert_preds"]
    topk_mask = details["topk_mask"].astype(int)

    rows = []
    n_samples, n_experts = entropies.shape
    for i in range(n_samples):
        true_label = int(y_true[i])
        for e in range(n_experts):
            pred_label = int(expert_preds[i, e])
            rows.append({
                "Run": run_name,
                "Sample_Index": i,
                "True_Label": true_label,
                "Expert_Index": e,
                "Expert_Entropy": float(entropies[i, e]),
                "Expert_Pred_Label": pred_label,
                "Selected_TopK": int(topk_mask[i, e]),
                "Expert_Correct": int(pred_label == true_label),
            })
    return pd.DataFrame(rows)


def compute_expert_stats_for_run(run_name, y_true, final_pred, details):
    entropies = details["entropies"]
    expert_preds = details["expert_preds"]
    topk_mask = details["topk_mask"].astype(bool)
    expert_correct = (expert_preds == y_true[:, None])

    selected_correct_ratio = float(expert_correct[topk_mask].mean()) if np.any(topk_mask) else np.nan
    selected_wrong_ratio = float((~expert_correct[topk_mask]).mean()) if np.any(topk_mask) else np.nan

    unselected_mask = ~topk_mask
    unselected_correct_ratio = float(expert_correct[unselected_mask].mean()) if np.any(unselected_mask) else np.nan
    unselected_wrong_ratio = float((~expert_correct[unselected_mask]).mean()) if np.any(unselected_mask) else np.nan

    selection_summary = {
        "Run": run_name,
        "Selected_Expert_Correct_Ratio": selected_correct_ratio,
        "Selected_Expert_Wrong_Ratio": selected_wrong_ratio,
        "Unselected_Expert_Correct_Ratio": unselected_correct_ratio,
        "Unselected_Expert_Wrong_Ratio": unselected_wrong_ratio,
        "Final_DER_Accuracy": float(accuracy_score(y_true, final_pred)),
    }

    expert_rows = []
    n_experts = expert_preds.shape[1]
    for e in range(n_experts):
        expert_rows.append({
            "Run": run_name,
            "Expert_Index": e,
            "Expert_Accuracy": float(accuracy_score(y_true, expert_preds[:, e])),
            "Expert_Mean_Entropy": float(entropies[:, e].mean()),
            "Expert_Selected_Rate": float(topk_mask[:, e].mean()),
        })

    return selection_summary, pd.DataFrame(expert_rows)


# ---------------------------------------------------------
# One-run Evaluation
# ---------------------------------------------------------
def evaluate_run(run_path, cfg, X_test_stft, X_test_raw, y_test, device, logger):
    run_name = os.path.basename(run_path)
    logger.info(f"\n[Eval] Directory: {run_path}")

    prepared = prepare_test_inputs(run_path, cfg, X_test_stft)
    x_bands_moe = prepared["X_bands_moe"]
    x_bl_concat = prepared["X_bl_concat"]
    moe_pca_params, moe_pca_flops = prepared["moe_overhead"]
    bl_pca_params, bl_pca_flops = prepared["bl_overhead"]
    training_times = load_training_times(run_path)

    models_to_eval = [
        ("BLS", "run_bls_model.pkl", "bls", x_bl_concat, x_bl_concat[0]),
        ("SFEBLN", "run_sfebln_model.pkl", "sfebln", X_test_raw, X_test_raw[0]),
        ("DER-BLMoE (Ours)", "run_flagship_moe_model.pkl", "moe_f", x_bands_moe, [b[0] for b in x_bands_moe]),
        ("ResNet18", "run_resnet18_best.pth", "pytorch_res", X_test_raw, X_test_raw[0]),
        ("AJCN", "run_ajcn_best.pth", "pytorch_ajcn", X_test_raw, X_test_raw[0]),
    ]

    metric_rows = []
    cm_norm_dict = {}
    expert_records_df = None
    expert_selection_summary = None
    expert_accuracy_df = None

    for model_name, model_file, model_type, x_input, x_sample_flops in models_to_eval:
        model_path = os.path.join(run_path, model_file)
        if not os.path.exists(model_path):
            logger.warning(f"  [Skip] Missing model file: {model_file}")
            continue

        if "pytorch" in model_type:
            if model_type == "pytorch_res":
                model = ResNet18_1D(num_classes=cfg.NUM_CLASSES).to(device)
            elif model_type == "pytorch_ajcn":
                info_path = os.path.join(run_path, "run_ajcn_info.pkl")
                if not os.path.exists(info_path):
                    logger.warning("  [Skip] Missing AJCN info file: run_ajcn_info.pkl")
                    continue
                with open(info_path, "rb") as f:
                    ajcn_info = pickle.load(f)
                model = build_official_ajcn_model_from_info(ajcn_info, num_classes=cfg.NUM_CLASSES).to(device)
            else:
                logger.warning(f"  [Skip] Unknown PyTorch model type: {model_type}")
                continue
            load_state_dict_safe(model, model_path, device)
            model.eval()
        else:
            with open(model_path, "rb") as f:
                model = pickle.load(f)

        moe_details = None

        if "pytorch" in model_type:
            if model_type == "pytorch_ajcn":
                x_torch = torch.tensor(raw_iq_to_ajcn_tensor(x_input), dtype=torch.float32).to(device)
            else:
                L = x_input.shape[1] // 2
                x_torch = torch.tensor(np.stack([x_input[:, :L], x_input[:, L:]], axis=1), dtype=torch.float32).to(device)

            sync_if_cuda(device)
            t0 = time.perf_counter()
            with torch.no_grad():
                out = model(x_torch)
                if isinstance(out, tuple):
                    out = out[0]
                pred = out.argmax(dim=1).cpu().numpy()
            sync_if_cuda(device)
            inference_time_s = time.perf_counter() - t0
        else:
            t0 = time.perf_counter()
            if model_type == "moe_f" and hasattr(model, "predict_proba_with_details"):
                probs, moe_details = model.predict_proba_with_details(x_input)
            else:
                probs = model.predict_proba(x_input)
            pred = probs.argmax(axis=1)
            inference_time_s = time.perf_counter() - t0

        latency_ms_per_sample = inference_time_s * 1000.0 / max(1, len(y_test))
        construction_time_s = get_model_construction_time(training_times, model_name)

        accuracy = float(accuracy_score(y_test, pred))
        if model_type == "pytorch_ajcn":
            core_params = count_ajcn_params(model)
            core_flops = estimate_ajcn_flops(model, input_shape=(3, 32, 32))
        else:
            core_params = count_params(model, model_type)
            core_flops = count_flops(model, model_type, x_sample_flops)

        overhead_params = moe_pca_params if model_type == "moe_f" else (bl_pca_params if model_type == "bls" else 0)
        overhead_flops = moe_pca_flops if model_type == "moe_f" else (bl_pca_flops if model_type == "bls" else 0)

        params_total = int(core_params + overhead_params)
        flops_total = int(core_flops + overhead_flops)

        metric_rows.append({
            "Run": run_name,
            "Model": model_name,
            "Accuracy": accuracy,
            "Params_Total": params_total,
            "FLOPs_Total": flops_total,
            "Params_Core": int(core_params),
            "Params_Overhead": int(overhead_params),
            "FLOPs_Core": int(core_flops),
            "FLOPs_Overhead": int(overhead_flops),
            "Model_Construction_Time_s": float(construction_time_s),
            "Inference_Time_s": float(inference_time_s),
            "Latency_ms_per_sample": float(latency_ms_per_sample),
        })

        if model_name in ["BLS", "DER-BLMoE (Ours)"]:
            cm_norm_dict[model_name] = normalized_confusion_matrix(y_test, pred, cfg.NUM_CLASSES)

        if model_type == "moe_f" and moe_details is not None:
            expert_records_df = build_expert_sample_records(run_name, y_test, moe_details)
            expert_selection_summary, expert_accuracy_df = compute_expert_stats_for_run(
                run_name, y_test, pred, moe_details
            )

        logger.info(
            f"   - {model_name:18s} | Acc={accuracy:.4f} | "
            f"Params={params_total / 1e3:.2f}K | FLOPs={flops_total / 1e6:.2f}M | "
            f"Construct={construction_time_s:.4f}s | "
            f"Infer={inference_time_s:.4f}s | Latency={latency_ms_per_sample:.4f}ms/sample"
        )

    return {
        "metrics": pd.DataFrame(metric_rows),
        "cm_norm": cm_norm_dict,
        "expert_records": expert_records_df,
        "expert_selection_summary": expert_selection_summary,
        "expert_accuracy": expert_accuracy_df,
    }


# ---------------------------------------------------------
# Aggregation and Saving
# ---------------------------------------------------------
def aggregate_and_save(run_outputs, output_root, cfg, logger):
    # 1) Model metrics: every run and final mean/std/95% CI.
    metrics_all = pd.concat([out["metrics"] for out in run_outputs if out["metrics"] is not None], ignore_index=True)
    model_order = [m for m in ["BLS", "SFEBLN", "ResNet18", "AJCN", "DER-BLMoE (Ours)"] if m in metrics_all["Model"].unique()]
    metrics_all["Model"] = pd.Categorical(metrics_all["Model"], categories=model_order, ordered=True)
    metrics_all = metrics_all.sort_values(["Run", "Model"])
    metrics_all.to_csv(os.path.join(output_root, "model_metrics_per_run.csv"), index=False)

    metrics_summary = mean_std_ci_table(
        metrics_all,
        group_col="Model",
        value_cols=[
            "Accuracy",
            "Params_Total",
            "FLOPs_Total",
            "Model_Construction_Time_s",
            "Inference_Time_s",
            "Latency_ms_per_sample",
        ],
    )
    metrics_summary["Model"] = pd.Categorical(metrics_summary["Model"], categories=model_order, ordered=True)
    metrics_summary = metrics_summary.sort_values("Model")
    metrics_summary.to_csv(os.path.join(output_root, "model_metrics_mean_std.csv"), index=False)
    metrics_summary.to_csv(os.path.join(output_root, "model_metrics_mean_std_ci.csv"), index=False)

    logger.info("\n[Final Model Performance: Mean ± 95% CI]")
    for _, row in metrics_summary.iterrows():
        logger.info(
            f"  {row['Model']:18s} | "
            f"Acc={row['Accuracy_Mean']:.4f} ± {row['Accuracy_CI95_HalfWidth']:.4f} "
            f"(95% CI=[{row['Accuracy_CI95_Lower']:.4f}, {row['Accuracy_CI95_Upper']:.4f}], "
            f"Std={row['Accuracy_Std']:.4f}, N={int(row['Accuracy_N'])}) | "
            f"Params={row['Params_Total_Mean'] / 1e3:.2f}K | "
            f"FLOPs={row['FLOPs_Total_Mean'] / 1e6:.2f}M | "
            f"Construct={row['Model_Construction_Time_s_Mean']:.4f} ± "
            f"{row['Model_Construction_Time_s_CI95_HalfWidth']:.4f}s | "
            f"Infer={row['Inference_Time_s_Mean']:.4f} ± "
            f"{row['Inference_Time_s_CI95_HalfWidth']:.4f}s | "
            f"Latency={row['Latency_ms_per_sample_Mean']:.4f} ± "
            f"{row['Latency_ms_per_sample_CI95_HalfWidth']:.4f}ms/sample"
        )

    # 2) Only save final averaged normalized confusion matrices for BLS and DER-BLMoE.
    cm_dir = os.path.join(output_root, "mean_confusion_matrices")
    os.makedirs(cm_dir, exist_ok=True)

    for model_name in ["BLS", "DER-BLMoE (Ours)"]:
        mats = [out["cm_norm"][model_name] for out in run_outputs if model_name in out["cm_norm"]]
        if not mats:
            continue
        cm_mean = np.mean(np.stack(mats, axis=0), axis=0)
        cm_std = np.std(np.stack(mats, axis=0), axis=0, ddof=1) if len(mats) > 1 else np.zeros_like(cm_mean)

        sname = safe_name(model_name)
        pd.DataFrame(cm_mean).to_csv(os.path.join(cm_dir, f"{sname}_mean_confusion_matrix.csv"), index=False)
        pd.DataFrame(cm_std).to_csv(os.path.join(cm_dir, f"{sname}_std_confusion_matrix.csv"), index=False)
        plot_confusion_matrix_mean(
            cm_mean,
            os.path.join(cm_dir, f"{sname}_mean_confusion_matrix.png"),
            f"{model_name} Mean Normalized Confusion Matrix",
        )

    # 3) Expert sample-level records.
    expert_record_dfs = [out["expert_records"] for out in run_outputs if out["expert_records"] is not None]
    if expert_record_dfs:
        expert_records_all = pd.concat(expert_record_dfs, ignore_index=True)
        expert_records_all.to_csv(os.path.join(output_root, "expert_sample_records_all_runs.csv"), index=False)
    else:
        expert_records_all = pd.DataFrame()

    # 4) Selection correctness ratios.
    selection_rows = [out["expert_selection_summary"] for out in run_outputs if out["expert_selection_summary"] is not None]
    if selection_rows:
        selection_per_run = pd.DataFrame(selection_rows)
        selection_per_run.to_csv(os.path.join(output_root, "expert_selection_correctness_per_run.csv"), index=False)

        selection_metric_rows = []
        for col in [
            "Selected_Expert_Correct_Ratio",
            "Unselected_Expert_Correct_Ratio",
            "Selected_Expert_Wrong_Ratio",
            "Unselected_Expert_Wrong_Ratio",
            "Final_DER_Accuracy",
        ]:
            stats = summarize_series_ci(selection_per_run[col])
            selection_metric_rows.append({
                "Metric": col,
                "N": stats["N"],
                "Mean": stats["Mean"],
                "Std": stats["Std"],
                "CI95_HalfWidth": stats["CI95_HalfWidth"],
                "CI95_Lower": stats["CI95_Lower"],
                "CI95_Upper": stats["CI95_Upper"],
                "Mean±Std": stats["Mean±Std"],
                "Mean±95CI": stats["Mean±95CI"],
            })
        selection_summary = pd.DataFrame(selection_metric_rows)
        selection_summary.to_csv(os.path.join(output_root, "expert_selection_correctness_mean_std.csv"), index=False)
        selection_summary.to_csv(os.path.join(output_root, "expert_selection_correctness_mean_std_ci.csv"), index=False)

        logger.info("\n[Expert Selection Correctness Ratios: Mean ± 95% CI]")
        for _, row in selection_summary.iterrows():
            logger.info(
                f"  {row['Metric']}: {row['Mean']:.4f} ± {row['CI95_HalfWidth']:.4f} | "
                f"95% CI=[{row['CI95_Lower']:.4f}, {row['CI95_Upper']:.4f}] | "
                f"Std={row['Std']:.4f} | N={int(row['N'])}"
            )

    # 5) Individual expert accuracy.
    expert_acc_dfs = [out["expert_accuracy"] for out in run_outputs if out["expert_accuracy"] is not None]
    if expert_acc_dfs:
        expert_acc_all = pd.concat(expert_acc_dfs, ignore_index=True)
        expert_acc_all.to_csv(os.path.join(output_root, "expert_accuracy_per_run.csv"), index=False)

        expert_acc_summary = mean_std_table(
            expert_acc_all,
            group_col="Expert_Index",
            value_cols=["Expert_Accuracy", "Expert_Mean_Entropy", "Expert_Selected_Rate"],
        )
        expert_acc_summary = expert_acc_summary.sort_values("Expert_Index")
        expert_acc_summary.to_csv(os.path.join(output_root, "expert_accuracy_mean_std.csv"), index=False)
        expert_acc_summary.to_csv(os.path.join(output_root, "expert_accuracy_mean_std_ci.csv"), index=False)

        logger.info("\n[Individual Expert Accuracy: Mean ± 95% CI]")
        for _, row in expert_acc_summary.iterrows():
            logger.info(
                f"  Expert {int(row['Expert_Index'])}: "
                f"Acc={row['Expert_Accuracy_Mean']:.4f} ± {row['Expert_Accuracy_CI95_HalfWidth']:.4f} | "
                f"95% CI=[{row['Expert_Accuracy_CI95_Lower']:.4f}, "
                f"{row['Expert_Accuracy_CI95_Upper']:.4f}] | "
                f"MeanEntropy={row['Expert_Mean_Entropy_Mean']:.4f} ± "
                f"{row['Expert_Mean_Entropy_CI95_HalfWidth']:.4f} | "
                f"SelectedRate={row['Expert_Selected_Rate_Mean']:.4f} ± "
                f"{row['Expert_Selected_Rate_CI95_HalfWidth']:.4f}"
            )

    logger.info(f"\n[Output] All results saved to: {output_root}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    cfg = get_moe_config()

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

    now = datetime.now()
    eval_folder_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{cfg.NUM_CLASSES}Class_Test"
    output_root = os.path.join(
        cfg.RESULTS_ROOT if cfg.RESULTS_ROOT else os.path.join(PROJECT_ROOT, "compare_results"),
        eval_folder_name,
    )
    os.makedirs(output_root, exist_ok=True)

    logger = setup_logger(os.path.join(output_root, "test.log"))
    logger.info("============================================================")
    logger.info(f"  Evaluating Checkpoints from: {latest_exp}")
    logger.info(f"  Results saving to: {output_root}")
    logger.info("============================================================")

    _, (te_stft, te_raw, te_y), _, _ = load_normal_data(cfg)
    y_test = te_y.astype(int).ravel()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_dirs = sorted([d for d in os.listdir(exp_path) if d.startswith("run")])
    run_outputs = []

    for d in run_dirs:
        run_path = os.path.join(exp_path, d)
        if not os.path.isdir(run_path):
            continue
        out = evaluate_run(run_path, cfg, te_stft, te_raw, y_test, device, logger)
        if len(out["metrics"]) > 0:
            run_outputs.append(out)

    if not run_outputs:
        logger.warning("[Warn] No valid run outputs.")
        return

    aggregate_and_save(run_outputs, output_root, cfg, logger)


if __name__ == "__main__":
    main()