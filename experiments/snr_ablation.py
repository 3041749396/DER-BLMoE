# experiments/snr_ablation.py
"""
SNR + Top-k ablation for DER-BLMoE, integrated with Baseline BLS.

This script compares, under each SNR:
    1) Baseline BLS
    2) DER Top-k with k = 1, 2, 4
    3) Uniform average of all expert outputs
    4) DER-BLMoE with the default k in the config, e.g., k=3

Important notes
---------------
- Changing k is inference-only. No retraining is performed.
- The DER gate remains parameter-free. Gate_Trainable_Params = 0.
- Uniform average is implemented by setting gate_alpha=0 and top_k=num_experts,
  so all experts receive equal weights.
- The same noisy STFT feature batch is shared by BLS, all k settings, uniform
  average, and default DER under the same run and SNR.
- The per-run CSV is updated immediately after each result is produced.
- The final summary reports mean accuracy with 95% confidence intervals.
- Default SNR range is from -15 dB to 20 dB.
"""

import os
import sys
import pickle
import argparse
import logging
from contextlib import contextmanager
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import seaborn as sns
except ImportError:
    sns = None

try:
    from scipy.stats import t as student_t
except Exception:
    student_t = None

from tqdm import tqdm
from sklearn.metrics import accuracy_score


# ---------------------------------------------------------
# Formatting Settings
# ---------------------------------------------------------
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False


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

from utils.normal_data_utils import (
    load_normal_data,
    stft_features_from_raw,
    calculate_global_crop_indices,
)
from utils.noise_data_utils import add_awgn_iq_batch

# Explicit imports for pickle loading.
from models.bls import BLSClassifier
from models.moe_entropy_gate import MoEBLSEntropyResidualGate


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def setup_logger(log_file):
    logger = logging.getLogger("SNRTopKAblationLogger")
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


@contextmanager
def temporary_model_config(model, **kwargs):
    """
    Temporarily modify model attributes during inference.
    The original values are restored after exiting the context.
    """
    original_state = {}

    for key in kwargs:
        if hasattr(model, key):
            original_state[key] = getattr(model, key)

    for key, value in kwargs.items():
        if hasattr(model, key):
            setattr(model, key, value)

    try:
        yield model
    finally:
        for key, value in original_state.items():
            setattr(model, key, value)


def find_experiment_dir(exp_arg=None):
    base_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    if exp_arg is not None:
        exp_path = exp_arg if os.path.isabs(exp_arg) else os.path.join(base_dir, exp_arg)

        if not os.path.isdir(exp_path):
            raise FileNotFoundError(f"Specified experiment directory not found: {exp_path}")

        return os.path.basename(exp_path), exp_path

    if not os.path.isdir(base_dir):
        raise FileNotFoundError(f"Checkpoints dir not found: {base_dir}")

    exp_dirs = [
        d for d in os.listdir(base_dir)
        if "Class" in d and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not exp_dirs:
        raise FileNotFoundError("No experiment directories found in checkpoints/.")

    latest_exp = max(
        exp_dirs,
        key=lambda d: os.path.getmtime(os.path.join(base_dir, d)),
    )

    return latest_exp, os.path.join(base_dir, latest_exp)


def parse_int_list(arg_value, default_values):
    if arg_value is None or str(arg_value).strip() == "":
        return list(default_values)

    vals = []

    for item in str(arg_value).split(","):
        item = item.strip()
        if item:
            vals.append(int(item))

    return sorted(set(vals))


def append_result_to_csv(row, csv_path):
    """
    Save each result immediately to avoid losing completed results
    if the script is interrupted later.
    """
    row_df = pd.DataFrame([row])
    file_exists = os.path.exists(csv_path)

    row_df.to_csv(
        csv_path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding="utf-8-sig",
    )


def load_run_resources(run_path, logger):
    """
    Load the saved baseline BLS, DER-BLMoE, and their PCA transformers.
    """
    # DER-BLMoE
    moe_path = os.path.join(run_path, "run_flagship_moe_model.pkl")

    if not os.path.exists(moe_path):
        logger.warning(f"[Skip] Missing DER model in {run_path}")
        return None

    with open(moe_path, "rb") as f:
        model_moe = pickle.load(f)

    pca_moe_path = os.path.join(run_path, "pca_transformers_moe.pkl")

    if os.path.exists(pca_moe_path):
        with open(pca_moe_path, "rb") as f:
            pca_moe = pickle.load(f)
    else:
        logger.warning(
            f"[Warn] Missing pca_transformers_moe.pkl in {run_path}; "
            f"using identity transforms."
        )
        pca_moe = None

    # Baseline BLS
    bls_path = os.path.join(run_path, "run_bls_model.pkl")

    if os.path.exists(bls_path):
        with open(bls_path, "rb") as f:
            model_bls = pickle.load(f)
    else:
        logger.warning(
            f"[Warn] Missing run_bls_model.pkl in {run_path}; "
            f"Baseline BLS will be skipped."
        )
        model_bls = None

    pca_bls_path = os.path.join(run_path, "pca_transformers_baseline.pkl")

    if os.path.exists(pca_bls_path):
        with open(pca_bls_path, "rb") as f:
            pca_bls = pickle.load(f)
    else:
        logger.warning(
            f"[Warn] Missing pca_transformers_baseline.pkl in {run_path}; "
            f"using identity transforms for BLS."
        )
        pca_bls = None

    return {
        "model_moe": model_moe,
        "pca_moe": pca_moe,
        "model_bls": model_bls,
        "pca_bls": pca_bls,
    }


def prepare_noisy_test_features(
    X_raw_clean,
    snr,
    cfg,
    pca_moe,
    pca_bls,
    crop_indices,
):
    """
    Generate one noisy test batch for a given SNR and convert it into:
      - per-expert PCA features for DER-BLMoE
      - concatenated PCA features for baseline BLS

    The same noisy STFT feature batch is used by all methods, ensuring fairness.
    """
    X_raw_noisy = add_awgn_iq_batch(
        X_raw_clean.copy(),
        snr_db=snr,
        per_sample=True,
    )

    X_stft = stft_features_from_raw(
        X_raw_noisy,
        cfg,
        crop_indices=crop_indices,
    )

    X_bands_raw = np.array_split(
        X_stft,
        cfg.NUM_LOGICAL_EXPERTS,
        axis=1,
    )

    X_bands_moe = []

    for i, band in enumerate(X_bands_raw):
        pca = pca_moe[i] if pca_moe is not None else None
        X_bands_moe.append(pca.transform(band) if pca is not None else band)

    X_bands_bls = []

    for i, band in enumerate(X_bands_raw):
        pca = pca_bls[i] if pca_bls is not None else None
        X_bands_bls.append(pca.transform(band) if pca is not None else band)

    X_bls_concat = np.concatenate(X_bands_bls, axis=1)

    return X_bands_moe, X_bls_concat


def predict_der_with_setting(model, X_bands_moe, setting):
    """
    setting:
        {"name": ..., "params": {...}, "type": ...}
    """
    with temporary_model_config(model, **setting["params"]):
        if hasattr(model, "predict_proba_with_details"):
            probs, _ = model.predict_proba_with_details(X_bands_moe)
        else:
            probs = model.predict_proba(X_bands_moe)

    return probs.argmax(axis=1)


def build_method_configs(num_experts, default_k, k_values):
    method_configs = []

    # Baseline BLS is not controlled by k.
    method_configs.append({
        "Method": "Baseline BLS",
        "Method_Type": "BLS_Baseline",
        "K": 0,
        "params": {},
        "Gate_Trainable_Params": 0,
        "Retrained": 0,
    })

    # Requested DER k values.
    for k in k_values:
        if k < 1 or k > num_experts:
            raise ValueError(
                f"k={k} is invalid. It must satisfy 1 <= k <= {num_experts}."
            )

        if k == default_k:
            name = f"DER-BLMoE (default k={k})"
        elif k == 1:
            name = "DER Top-k (k=1, hard switch)"
        else:
            name = f"DER Top-k (k={k})"

        method_configs.append({
            "Method": name,
            "Method_Type": "DER_TopK",
            "K": int(k),
            "params": {"top_k": int(k)},
            "Gate_Trainable_Params": 0,
            "Retrained": 0,
        })

    # Add default DER-BLMoE if default k is not in requested k list.
    if default_k not in k_values:
        method_configs.append({
            "Method": f"DER-BLMoE (default k={default_k})",
            "Method_Type": "DER_Default",
            "K": int(default_k),
            "params": {"top_k": int(default_k)},
            "Gate_Trainable_Params": 0,
            "Retrained": 0,
        })

    # Uniform average of all expert outputs.
    method_configs.append({
        "Method": "Uniform Average Experts",
        "Method_Type": "Uniform_Average",
        "K": int(num_experts),
        "params": {
            "gate_alpha": 0.0,
            "top_k": int(num_experts),
        },
        "Gate_Trainable_Params": 0,
        "Retrained": 0,
    })

    return method_configs


def calc_ci95_half_width(std, n):
    """
    Compute the half-width of the 95% confidence interval:

        CI_half_width = t_{0.975, n-1} * std / sqrt(n)

    If scipy is unavailable, use 1.96 as a normal-approximation fallback.
    """
    if n <= 1 or std is None or np.isnan(std):
        return 0.0

    if student_t is not None:
        t_value = float(student_t.ppf(0.975, df=n - 1))
    else:
        t_value = 1.96

    return float(t_value * std / np.sqrt(n))


def summarize_results(df):
    """
    Summarize per-run results using mean accuracy and 95% confidence interval.

    The final CSV will contain:
        Accuracy_Mean
        Accuracy_Std
        Accuracy_CI95_HalfWidth
        Accuracy_CI95_Low
        Accuracy_CI95_High
        Accuracy_Mean±95CI
        Accuracy_95CI_Range
    """
    summary = (
        df.groupby(["SNR", "Method"], as_index=False)
          .agg(
              Accuracy_Mean=("Accuracy", "mean"),
              Accuracy_Std=("Accuracy", "std"),
              Accuracy_Min=("Accuracy", "min"),
              Accuracy_Max=("Accuracy", "max"),
              Num_Runs=("Accuracy", "count"),
              K=("K", "first"),
              Method_Type=("Method_Type", "first"),
              Gate_Trainable_Params=("Gate_Trainable_Params", "first"),
              Retrained=("Retrained", "first"),
          )
    )

    summary["Accuracy_Std"] = summary["Accuracy_Std"].fillna(0.0)

    summary["Accuracy_CI95_HalfWidth"] = summary.apply(
        lambda r: calc_ci95_half_width(
            std=float(r["Accuracy_Std"]),
            n=int(r["Num_Runs"]),
        ),
        axis=1,
    )

    summary["Accuracy_CI95_Low"] = (
        summary["Accuracy_Mean"] - summary["Accuracy_CI95_HalfWidth"]
    ).clip(lower=0.0)

    summary["Accuracy_CI95_High"] = (
        summary["Accuracy_Mean"] + summary["Accuracy_CI95_HalfWidth"]
    ).clip(upper=1.0)

    summary["Accuracy_Mean±95CI"] = summary.apply(
        lambda r: (
            f"{r['Accuracy_Mean']:.6g} ± "
            f"{r['Accuracy_CI95_HalfWidth']:.6g}"
        ),
        axis=1,
    )

    summary["Accuracy_95CI_Range"] = summary.apply(
        lambda r: (
            f"[{r['Accuracy_CI95_Low']:.6g}, "
            f"{r['Accuracy_CI95_High']:.6g}]"
        ),
        axis=1,
    )

    return summary


def plot_snr_topk(summary_df, save_path, method_order):
    """
    Plot SNR robustness curves.
    Error bars denote 95% confidence intervals.
    """
    plt.figure(figsize=(10.2, 7.0))

    if sns is not None:
        sns.set_style("whitegrid")
        sns.set_context("paper", font_scale=1.35)

    markers = ["o", "s", "^", "D", "X", "v", "P", "*"]

    if sns is not None:
        palette = sns.color_palette("tab10", n_colors=len(method_order))
    else:
        palette = [None] * len(method_order)

    for idx, method in enumerate(method_order):
        sub = summary_df[summary_df["Method"] == method].sort_values("SNR")

        if sub.empty:
            continue

        x = sub["SNR"].values
        y = sub["Accuracy_Mean"].values
        yerr = sub["Accuracy_CI95_HalfWidth"].values

        plot_kwargs = {
            "x": x,
            "y": y,
            "yerr": yerr,
            "fmt": markers[idx % len(markers)] + "-",
            "linewidth": 2.2,
            "markersize": 7.5,
            "capsize": 3.5,
            "elinewidth": 1.1,
            "label": method,
        }

        if palette[idx] is not None:
            plot_kwargs["color"] = palette[idx]

        plt.errorbar(**plot_kwargs)

    plt.xlabel("SNR (dB)", fontweight="bold", fontname="Times New Roman")
    plt.ylabel("Accuracy", fontweight="bold", fontname="Times New Roman")
    plt.title(
        "SNR Robustness with 95% Confidence Intervals",
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.ylim(0.0, 1.02)

    plt.legend(
        loc="lower right",
        prop={
            "family": "Times New Roman",
            "size": 9,
        },
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_grouped_bar(summary_df, save_path, method_order):
    """
    Plot grouped bar chart.
    Error bars denote 95% confidence intervals.
    This version does not require seaborn.
    """
    snr_values = sorted(summary_df["SNR"].unique())
    x = np.arange(len(snr_values))

    num_methods = len(method_order)
    width = 0.8 / max(num_methods, 1)

    plt.figure(figsize=(11.5, 6.8))

    for idx, method in enumerate(method_order):
        sub = (
            summary_df[summary_df["Method"] == method]
            .set_index("SNR")
            .reindex(snr_values)
        )

        y = sub["Accuracy_Mean"].values
        yerr = sub["Accuracy_CI95_HalfWidth"].values

        offset = (idx - (num_methods - 1) / 2.0) * width

        plt.bar(
            x + offset,
            y,
            width=width,
            yerr=yerr,
            capsize=3,
            label=method,
            edgecolor="black",
            linewidth=0.8,
        )

    plt.xticks(x, [str(v) for v in snr_values])
    plt.ylabel("Accuracy", fontweight="bold", fontname="Times New Roman")
    plt.xlabel("SNR (dB)", fontweight="bold", fontname="Times New Roman")
    plt.title(
        "Top-k Ablation under Different SNR Conditions",
        fontweight="bold",
        fontname="Times New Roman",
    )
    plt.ylim(0.0, 1.02)
    plt.grid(True, axis="y", linestyle="--", alpha=0.35)

    plt.legend(
        loc="lower right",
        prop={
            "family": "Times New Roman",
            "size": 8,
        },
        frameon=True,
        edgecolor="black",
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Integrated SNR + Top-k ablation for DER-BLMoE with Baseline BLS."
    )

    parser.add_argument(
        "--exp_dir",
        type=str,
        default=None,
        help=(
            "Checkpoint experiment folder name under checkpoints/, "
            "or an absolute path. Default: latest experiment."
        ),
    )

    parser.add_argument(
        "--k_values",
        type=str,
        default="1,2,4",
        help="Comma-separated k values for DER Top-k ablation. Default: 1,2,4.",
    )

    parser.add_argument(
        "--snr_values",
        type=str,
        default="-15,-10,-5,0,5,10,15,20",
        help="Comma-separated SNR values in dB. Default: -15,-10,-5,0,5,10,15,20.",
    )

    args = parser.parse_args()

    cfg = get_moe_config()
    latest_exp, exp_path = find_experiment_dir(args.exp_dir)

    now = datetime.now()
    output_name = (
        f"{now.year}_{now.month}_{now.day}_"
        f"{now.hour}_{now.minute:02d}_"
        f"{cfg.NUM_CLASSES}Class_SNR_TopK_Ablation"
    )

    output_root = os.path.join(
        cfg.RESULTS_ROOT if cfg.RESULTS_ROOT else os.path.join(PROJECT_ROOT, "compare_results"),
        output_name,
    )

    os.makedirs(output_root, exist_ok=True)

    log_path = os.path.join(output_root, "snr_topk_ablation.log")
    per_run_path = os.path.join(output_root, "snr_topk_ablation_per_run.csv")
    summary_path = os.path.join(output_root, "snr_topk_ablation_summary.csv")
    plot_path = os.path.join(output_root, "snr_topk_ablation_accuracy.png")
    bar_path = os.path.join(output_root, "snr_topk_ablation_bar.png")

    logger = setup_logger(log_path)

    logger.info("============================================================")
    logger.info("  Integrated SNR + Top-k Ablation with Baseline BLS")
    logger.info(f"  Checkpoint experiment: {latest_exp}")
    logger.info(f"  Results saving to: {output_root}")
    logger.info("  Summary format: mean accuracy ± 95% confidence interval")
    logger.info("  Default SNR range: -15,-10,-5,0,5,10,15,20 dB")
    logger.info("============================================================")

    if student_t is None:
        logger.info(
            "[Notice] scipy.stats is unavailable. "
            "The 95% CI will use the normal approximation value 1.96."
        )
    else:
        logger.info("[Notice] scipy.stats is available. The 95% CI will use Student's t distribution.")

    if sns is None:
        logger.info("[Notice] seaborn is unavailable. Plots will use matplotlib only.")

    # Load data.
    (tr_stft, tr_raw, tr_y), (te_stft, te_raw, te_y), _, _ = load_normal_data(cfg)

    y_test = te_y.astype(int).ravel()
    X_test_clean = te_raw

    # Recover crop indices to keep STFT feature dimension consistent.
    energy_ratio = getattr(cfg, "STFT_ENERGY_RATIO", 0.95)
    crop_indices = calculate_global_crop_indices(
        tr_raw,
        cfg,
        energy_ratio=energy_ratio,
    )

    logger.info(f"Crop indices: {crop_indices}")
    logger.info(f"Energy ratio for cropping: {energy_ratio}")

    # Locate runs.
    run_dirs = sorted([
        d for d in os.listdir(exp_path)
        if d.startswith("run") and os.path.isdir(os.path.join(exp_path, d))
    ])

    if not run_dirs:
        raise FileNotFoundError(f"No runXX directories found in {exp_path}")

    # Determine num_experts/default_k from first available DER model.
    first_model = None

    for d in run_dirs:
        p = os.path.join(exp_path, d, "run_flagship_moe_model.pkl")

        if os.path.exists(p):
            with open(p, "rb") as f:
                first_model = pickle.load(f)
            break

    if first_model is None:
        raise FileNotFoundError("No run_flagship_moe_model.pkl found in any runXX directory.")

    num_experts = int(getattr(first_model, "num_experts", cfg.NUM_LOGICAL_EXPERTS))
    default_k = int(getattr(cfg, "MOE_TOP_K", getattr(first_model, "top_k", min(3, num_experts))))

    k_values = parse_int_list(
        args.k_values,
        default_values=[1, 2, 4],
    )

    snr_values = parse_int_list(
        args.snr_values,
        default_values=[-15, -10, -5, 0, 5, 10, 15, 20],
    )

    method_configs = build_method_configs(
        num_experts=num_experts,
        default_k=default_k,
        k_values=k_values,
    )

    method_order = [m["Method"] for m in method_configs]

    logger.info(f"Num experts: {num_experts}")
    logger.info(f"Default DER k: {default_k}")
    logger.info(f"Evaluated k values: {k_values}")
    logger.info(f"SNR values: {snr_values}")
    logger.info("Methods:")

    for m in method_configs:
        logger.info(
            f"  - {m['Method']} | "
            f"type={m['Method_Type']} | "
            f"K={m['K']} | "
            f"GateParams=0 | "
            f"Retrained=0"
        )

    all_rows = []

    # If a previous file with the same name exists, remove it.
    # This avoids appending new results to stale results.
    if os.path.exists(per_run_path):
        os.remove(per_run_path)

    for run_name in tqdm(run_dirs, desc="Runs"):
        run_path = os.path.join(exp_path, run_name)
        resources = load_run_resources(run_path, logger)

        if resources is None:
            continue

        model_moe = resources["model_moe"]
        pca_moe = resources["pca_moe"]
        model_bls = resources["model_bls"]
        pca_bls = resources["pca_bls"]

        logger.info(f"\n[Run] {run_name}")

        for snr in tqdm(snr_values, desc=f"{run_name} SNR", leave=False):
            # Same noisy batch is shared by all methods for this run and SNR.
            X_bands_moe, X_bls_concat = prepare_noisy_test_features(
                X_raw_clean=X_test_clean,
                snr=snr,
                cfg=cfg,
                pca_moe=pca_moe,
                pca_bls=pca_bls,
                crop_indices=crop_indices,
            )

            for method in method_configs:
                if method["Method_Type"] == "BLS_Baseline":
                    if model_bls is None:
                        continue

                    pred = model_bls.predict(X_bls_concat)

                else:
                    pred = predict_der_with_setting(
                        model_moe,
                        X_bands_moe,
                        method,
                    )

                acc = float(accuracy_score(y_test, pred))

                row = {
                    "Run": run_name,
                    "SNR": int(snr),
                    "Method": method["Method"],
                    "Method_Type": method["Method_Type"],
                    "K": int(method["K"]),
                    "Accuracy": acc,
                    "Gate_Trainable_Params": int(method["Gate_Trainable_Params"]),
                    "Retrained": int(method["Retrained"]),
                }

                all_rows.append(row)

                # Immediately save each result.
                append_result_to_csv(row, per_run_path)

                logger.info(
                    f"  SNR={snr:>3} | "
                    f"{method['Method']:<32s} | "
                    f"Acc={acc:.4f}"
                )

    if not all_rows:
        raise RuntimeError("No valid results were produced.")

    # Rebuild per-run dataframe from the saved CSV to make sure the final summary
    # is exactly based on the recorded per-run results.
    per_run_df = pd.read_csv(per_run_path)

    summary_df = summarize_results(per_run_df)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    plot_snr_topk(
        summary_df,
        plot_path,
        method_order=method_order,
    )

    plot_grouped_bar(
        summary_df,
        bar_path,
        method_order=method_order,
    )

    logger.info("\n[Summary: Accuracy Mean ± 95% CI]")

    for snr in snr_values:
        logger.info(f"\nSNR = {snr} dB")

        sub = summary_df[summary_df["SNR"] == snr].set_index("Method")

        for method in method_order:
            if method in sub.index:
                row = sub.loc[method]

                logger.info(
                    f"  {method:<32s} | "
                    f"Acc={row['Accuracy_Mean']:.4f} ± "
                    f"{row['Accuracy_CI95_HalfWidth']:.4f} "
                    f"(95% CI: "
                    f"[{row['Accuracy_CI95_Low']:.4f}, "
                    f"{row['Accuracy_CI95_High']:.4f}]) | "
                    f"Std={row['Accuracy_Std']:.4f} | "
                    f"Runs={int(row['Num_Runs'])} | "
                    f"K={int(row['K'])} | "
                    f"GateParams={int(row['Gate_Trainable_Params'])}"
                )

    logger.info(f"\n[Output] Log file: {log_path}")
    logger.info(f"[Output] Per-run CSV: {per_run_path}")
    logger.info(f"[Output] Summary CSV with 95% CI: {summary_path}")
    logger.info(f"[Output] Line plot with 95% CI: {plot_path}")
    logger.info(f"[Output] Bar plot with 95% CI: {bar_path}")
    logger.info("[DONE] Integrated SNR + Top-k ablation completed.")


if __name__ == "__main__":
    main()