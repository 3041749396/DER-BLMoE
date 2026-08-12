# experiments/exp_pca_analysis.py
"""
Lightweight PCA analysis for DER-BLMoE.

This script is designed to address the reviewer request:
"The manuscript mentions PCA projection but never visualizes it. Showing the
sub-band feature distributions before and after dimensionality reduction would
help justify the choice of D_pca and demonstrate information retention."

Given the page limit, this lightweight version keeps only:
1) Part A: cumulative explained variance curves for each sub-band PCA.
2) Part B: before/after PCA feature visualization for representative experts.

Outputs
-------
compare_results/<timestamp>_<Class>Class_PCA_Analysis/
    pca_explained_variance_curves.csv
    pca_explained_variance_curves.png
    pca_info_retention_summary.csv
    pca_feature_vis_expert*_before_after.png
    pca_analysis.log
"""

import os
import sys
import logging
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

try:
    from configs.config_moe_bls import get_moe_config
except ModuleNotFoundError:
    from config.config_moe_bls import get_moe_config

from utils.normal_data_utils import load_normal_data


def setup_logger(log_file):
    logger = logging.getLogger("PCAAnalysisLogger")
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


def parse_int_list(arg_value, default_values):
    if arg_value is None or str(arg_value).strip() == "":
        return list(default_values)
    vals = []
    for item in str(arg_value).split(","):
        item = item.strip()
        if item:
            vals.append(int(item))
    return sorted(set(vals))


def split_bands(X_stft, num_bands):
    return np.array_split(X_stft, num_bands, axis=1)


def fit_band_pcas(X_train_bands, pca_dim, random_state=2025):
    pca_list = []
    X_train_pca_bands = []
    for band in X_train_bands:
        n_comp = int(min(pca_dim, band.shape[0], band.shape[1]))
        pca = PCA(n_components=n_comp, random_state=random_state)
        X_pca = pca.fit_transform(band)
        pca_list.append(pca)
        X_train_pca_bands.append(X_pca)
    return pca_list, X_train_pca_bands


# ---------------------------------------------------------
# Part A: cumulative explained variance curves
# ---------------------------------------------------------
def compute_explained_variance_curves(X_train_bands, cfg, output_root, logger, max_components=None):
    logger.info("[Part A] Computing cumulative explained variance curves...")
    rows = []
    default_dim = int(getattr(cfg, "PCA_DIM_MOE", 16))

    for expert_idx, X_band in enumerate(X_train_bands):
        max_comp_possible = min(X_band.shape[0], X_band.shape[1])
        if max_components is None:
            max_comp = min(max_comp_possible, max(128, default_dim * 4))
        else:
            max_comp = min(max_comp_possible, int(max_components))

        pca = PCA(n_components=max_comp, random_state=int(getattr(cfg, "GLOBAL_SEED", 2025)))
        pca.fit(X_band)
        cum = np.cumsum(pca.explained_variance_ratio_)

        for d, val in enumerate(cum, start=1):
            rows.append({
                "Expert": expert_idx,
                "PCA_Dim": d,
                "Cumulative_Explained_Variance": float(val),
                "Explained_Variance_Ratio": float(pca.explained_variance_ratio_[d - 1]),
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_root, "pca_explained_variance_curves.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(8.2, 5.8))
    for expert_idx in sorted(df["Expert"].unique()):
        sub = df[df["Expert"] == expert_idx]
        plt.plot(sub["PCA_Dim"], sub["Cumulative_Explained_Variance"], linewidth=2.0, label=f"Expert {expert_idx}")

    plt.axvline(default_dim, linestyle="--", linewidth=1.6, color="black", label=f"Default $D_{{pca}}$={default_dim}")
    plt.xlabel("PCA Dimension", fontweight="bold", fontname="Times New Roman")
    plt.ylabel("Cumulative Explained Variance Ratio", fontweight="bold", fontname="Times New Roman")
    plt.title("Sub-band PCA Cumulative Explained Variance", fontweight="bold", fontname="Times New Roman")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.ylim(0.0, 1.02)
    plt.legend(frameon=True, edgecolor="black", prop={"family": "Times New Roman", "size": 9})
    plt.tight_layout()
    fig_path = os.path.join(output_root, "pca_explained_variance_curves.png")
    plt.savefig(fig_path, dpi=600, bbox_inches="tight")
    plt.close()

    logger.info(f"    -> Saved: {csv_path}")
    logger.info(f"    -> Saved: {fig_path}")
    return df


def compute_info_retention_summary(explained_df, pca_dims, output_root, logger):
    rows = []
    for expert in sorted(explained_df["Expert"].unique()):
        sub = explained_df[explained_df["Expert"] == expert].set_index("PCA_Dim")
        for d in pca_dims:
            if d in sub.index:
                val = float(sub.loc[d, "Cumulative_Explained_Variance"])
            else:
                val = float(sub["Cumulative_Explained_Variance"].iloc[-1])
            rows.append({
                "Expert": int(expert),
                "PCA_Dim": int(d),
                "Cumulative_Explained_Variance": val,
            })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(output_root, "pca_info_retention_summary.csv")
    df.to_csv(csv_path, index=False)
    logger.info(f"    -> Saved: {csv_path}")
    return df


# ---------------------------------------------------------
# Part B: before/after PCA visualization
# ---------------------------------------------------------
def balanced_sample_indices(y, per_class=50, seed=2025, max_total=800):
    rng = np.random.RandomState(seed)
    selected = []
    classes = np.unique(y.astype(int))

    for c in classes:
        idx = np.where(y.astype(int) == c)[0]
        if len(idx) == 0:
            continue
        take = min(per_class, len(idx))
        selected.extend(rng.choice(idx, size=take, replace=False).tolist())

    selected = np.array(selected, dtype=int)
    if len(selected) > max_total:
        selected = rng.choice(selected, size=max_total, replace=False)

    selected = selected[np.argsort(y[selected])]
    return selected


def make_tsne_embedding(X, seed=2025):
    X = np.asarray(X, dtype=np.float64)
    X = StandardScaler(with_mean=True, with_std=True).fit_transform(X)
    perplexity = min(30, max(5, (X.shape[0] - 1) // 3))
    tsne = TSNE(
        n_components=2,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
    )
    return tsne.fit_transform(X)


def plot_before_after_tsne(emb_before, emb_after, y, expert_idx, save_path):
    classes = np.unique(y.astype(int))
    palette = sns.color_palette("tab10", n_colors=max(len(classes), 10))

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.8))
    for ax, emb, title in zip(axes, [emb_before, emb_after], ["Before PCA", "After PCA Projection"]):
        for c in classes:
            mask = (y.astype(int) == c)
            ax.scatter(
                emb[mask, 0],
                emb[mask, 1],
                s=14,
                alpha=0.75,
                color=palette[int(c) % len(palette)],
                label=f"C{int(c)}" if title == "Before PCA" else None,
                edgecolors="none",
            )
        ax.set_title(title, fontweight="bold", fontname="Times New Roman")
        ax.set_xlabel("t-SNE Dim 1", fontname="Times New Roman")
        ax.set_ylabel("t-SNE Dim 2", fontname="Times New Roman")
        ax.grid(True, linestyle="--", alpha=0.25)

    axes[0].legend(
        loc="best",
        frameon=True,
        edgecolor="black",
        prop={"family": "Times New Roman", "size": 7},
        markerscale=1.2,
    )

    fig.suptitle(f"Sub-band Expert {expert_idx}: Feature Distribution Before/After PCA", fontweight="bold", fontname="Times New Roman")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def visualize_before_after_pca(X_train_bands, X_train_pca_bands, y_train, output_root, experts_to_plot, per_class, seed, logger):
    logger.info("[Part B] Visualizing before/after PCA feature distributions...")
    idx = balanced_sample_indices(y_train, per_class=per_class, seed=seed, max_total=900)
    y_sub = y_train[idx].astype(int)

    for expert_idx in experts_to_plot:
        if expert_idx < 0 or expert_idx >= len(X_train_bands):
            logger.warning(f"    [Skip] Expert {expert_idx} is out of range.")
            continue

        X_before = X_train_bands[expert_idx][idx]
        X_after = X_train_pca_bands[expert_idx][idx]

        emb_before = make_tsne_embedding(X_before, seed=seed + expert_idx * 17)
        emb_after = make_tsne_embedding(X_after, seed=seed + expert_idx * 17)

        save_path = os.path.join(output_root, f"pca_feature_vis_expert{expert_idx}_before_after.png")
        plot_before_after_tsne(emb_before, emb_after, y_sub, expert_idx, save_path)
        logger.info(f"    -> Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Lightweight PCA visualization for DER-BLMoE.")
    parser.add_argument("--experts_to_plot", type=str, default="0,2,3", help="Representative experts to visualize.")
    parser.add_argument("--vis_per_class", type=int, default=50, help="Samples per class for t-SNE visualization.")
    parser.add_argument("--max_curve_components", type=int, default=None, help="Maximum PCA components for explained-variance curves.")
    args = parser.parse_args()

    cfg = get_moe_config()
    seed = int(getattr(cfg, "GLOBAL_SEED", 2025))
    now = datetime.now()
    output_name = f"{now.year}_{now.month}_{now.day}_{now.hour}_{now.minute:02d}_{cfg.NUM_CLASSES}Class_PCA_Analysis"
    output_root = os.path.join(
        cfg.RESULTS_ROOT if cfg.RESULTS_ROOT else os.path.join(PROJECT_ROOT, "compare_results"),
        output_name,
    )
    os.makedirs(output_root, exist_ok=True)

    logger = setup_logger(os.path.join(output_root, "pca_analysis.log"))
    logger.info("============================================================")
    logger.info("  Lightweight PCA Visualization")
    logger.info(f"  Results saving to: {output_root}")
    logger.info("============================================================")

    (tr_stft, tr_raw, tr_y), (_, _, _), _, _ = load_normal_data(cfg)
    y_train = tr_y.astype(int).ravel()

    num_experts = int(getattr(cfg, "NUM_LOGICAL_EXPERTS", 5))
    default_dim = int(getattr(cfg, "PCA_DIM_MOE", 16))
    experts_to_plot = parse_int_list(args.experts_to_plot, default_values=[0, 2, 3])
    experts_to_plot = [e for e in experts_to_plot if 0 <= e < num_experts]

    logger.info(f"Num experts: {num_experts}")
    logger.info(f"Default PCA_DIM_MOE: {default_dim}")
    logger.info(f"Experts to visualize: {experts_to_plot}")

    X_train_bands = split_bands(tr_stft, num_experts)

    explained_df = compute_explained_variance_curves(
        X_train_bands=X_train_bands,
        cfg=cfg,
        output_root=output_root,
        logger=logger,
        max_components=args.max_curve_components,
    )
    compute_info_retention_summary(
        explained_df=explained_df,
        pca_dims=[default_dim],
        output_root=output_root,
        logger=logger,
    )

    _, X_train_default_pca_bands = fit_band_pcas(
        X_train_bands,
        pca_dim=default_dim,
        random_state=seed,
    )

    visualize_before_after_pca(
        X_train_bands=X_train_bands,
        X_train_pca_bands=X_train_default_pca_bands,
        y_train=y_train,
        output_root=output_root,
        experts_to_plot=experts_to_plot,
        per_class=args.vis_per_class,
        seed=seed,
        logger=logger,
    )

    logger.info("[DONE] Lightweight PCA analysis completed.")


if __name__ == "__main__":
    main()
