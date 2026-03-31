# utils/noise_data_utils.py
# -------------------------------------------------------------
# 噪声鲁棒性实验工具：
#   1) 在原始 IQ (Real+Imag 拼接) 上添加复高斯白噪声 (AWGN)
#   2) （可选）从配置直接读取原始数据 -> 加噪 -> 重算 STFT
# -------------------------------------------------------------
from typing import Optional, Tuple

import numpy as np

from .normal_data_utils import (
    load_normal_data,
    stft_features_from_raw,
)


# -------------------------------------------------------------
# IQ <-> Complex 辅助函数（自包含实现）
# -------------------------------------------------------------
def _iq_concat_to_complex(x: np.ndarray) -> np.ndarray:
    """
    将 Real+Imag 拼接格式的 IQ 转为复数数组。

    约定：
    - 输入 x: shape = (N, 2L)，实数
      前 L 为实部，后 L 为虚部
    - 如果本身就是复数 (N, L)，则直接返回
    """
    x = np.asarray(x)

    if np.iscomplexobj(x):
        # 已经是复数格式
        if x.ndim != 2:
            raise ValueError(f"Expected 2D complex array, got shape {x.shape}")
        return x

    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for IQ concat, got shape {x.shape}")

    N, D = x.shape
    if D % 2 != 0:
        raise ValueError(f"Last dim must be even for IQ concat, got D={D}")

    L = D // 2
    real = x[:, :L]
    imag = x[:, L:]
    return real + 1j * imag


def _complex_to_iq_concat(x: np.ndarray) -> np.ndarray:
    """
    将复数 IQ 数组转换为 Real+Imag 拼接格式。

    输入：
    - x: shape = (N, L)，complex

    输出：
    - iq_concat: shape = (N, 2L)，float32
      前 L 为实部，后 L 为虚部
    """
    x = np.asarray(x)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D complex array, got shape {x.shape}")

    real = np.real(x)
    imag = np.imag(x)
    return np.concatenate([real, imag], axis=1).astype(np.float32)


# -------------------------------------------------------------
# 1) 纯加噪接口：在给定 Raw IQ 上加 AWGN
# -------------------------------------------------------------
def add_awgn_iq_batch(
    X_raw: np.ndarray,
    snr_db: float,
    per_sample: bool = True,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    在 IQ 时域序列上添加复高斯白噪声 (AWGN)，得到指定 SNR 的含噪数据。

    设计原则：
    - 输入 / 输出格式与现有工程一致：Real+Imag 拼接 (N, 2L) 的 float32。
    - 噪声在“信号输入端”注入，模拟物理信道中的 AWGN。
    - 默认按 sample 级别控制 SNR，使每个样本的信噪比大致等于 snr_db。

    参数
    ----
    X_raw : np.ndarray
        shape = (N, 2L) 或复数 (N, L)。
        常规情况：Real+Imag 拼接格式 (N, 2L)，前 L 为实部，后 L 为虚部。
    snr_db : float
        目标信噪比 (dB)，例如 -5, 0, 5, 10, 20 等。
    per_sample : bool, default=True
        True  -> 每个样本单独估计信号功率并控制 SNR；
        False -> 用整个 batch 的平均功率控制统一 SNR。
    seed : Optional[int]
        随机种子，用于噪声生成的可复现性；为 None 时使用默认随机源。

    返回
    ----
    X_raw_noisy : np.ndarray
        与 X_raw 相同 shape 的含噪 IQ（Real+Imag 拼接），dtype=float32。
    """
    # 0. 统一转成复数形式 (N, L_seq)
    X_complex = _iq_concat_to_complex(X_raw)  # (N, L)
    N, L_seq = X_complex.shape

    rng = np.random.default_rng(seed)

    # 1. 估计信号功率 E[|x|^2]
    if per_sample:
        # 每个样本独立功率 -> 每个样本 SNR 尽量接近 snr_db
        power_sig = np.mean(np.abs(X_complex) ** 2, axis=1, keepdims=True)  # (N,1)
    else:
        # 整个 batch 一个平均功率
        avg_power = np.mean(np.abs(X_complex) ** 2)
        power_sig = np.full((N, 1), avg_power, dtype=np.float64)

    # 避免零功率样本导致数值问题
    eps = 1e-12
    power_sig = np.maximum(power_sig, eps)

    snr_linear = 10.0 ** (snr_db / 10.0)
    power_noise = power_sig / snr_linear  # (N, 1)

    # 复高斯噪声：实/虚各占一半功率
    sigma = np.sqrt(power_noise / 2.0)  # (N, 1)

    # 2. 生成复高斯白噪声并叠加
    noise_real = rng.standard_normal(size=(N, L_seq))
    noise_imag = rng.standard_normal(size=(N, L_seq))
    noise = (noise_real + 1j * noise_imag) * sigma  # 广播到 (N, L_seq)

    X_noisy_complex = X_complex + noise

    # 3. 转回 Real+Imag 拼接格式，保持与原始数据格式一致
    X_raw_noisy = _complex_to_iq_concat(X_noisy_complex)
    return X_raw_noisy


# -------------------------------------------------------------
# 2) “按你想法”封装的一站式接口（可选使用）
# -------------------------------------------------------------
def load_noise_robustness_data(
    cfg,
    snr_db: float,
    per_sample_snr: bool = True,
    seed: Optional[int] = None,
) -> Tuple[
    Tuple[np.ndarray, np.ndarray, np.ndarray],
    Tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """
    噪声鲁棒性实验专用数据加载入口（基于 MoeBLSConfig）。

    核心流程：
      1) 使用 normal_data_utils.load_normal_data(cfg) 读取“干净”的训练 / 测试数据：
           (X_train_stft, X_train_raw, Y_train),
           (X_test_stft_clean, X_test_raw_clean, Y_test)
      2) 仅在测试集 Raw IQ 上注入 AWGN，得到 X_test_raw_noisy；
      3) 在 X_test_raw_noisy 上重新计算 STFT 特征，得到 X_test_stft_noisy；
      4) 训练集保持完全不变（干净）。

    返回：
      (X_train_stft, X_train_raw, Y_train),
      (X_test_stft_noisy, X_test_raw_noisy, Y_test)
    """
    # 1. 加载“干净”的训练 / 测试数据
    (X_train_stft, X_train_raw, Y_train), (
        X_test_stft_clean,
        X_test_raw_clean,
        Y_test,
    ) = load_normal_data(cfg)

    # 2. 测试集加噪
    X_test_raw_noisy = add_awgn_iq_batch(
        X_test_raw_clean,
        snr_db=snr_db,
        per_sample=per_sample_snr,
        seed=seed,
    )

    print(
        f"[NoiseData] Applied AWGN to test set: "
        f"SNR={snr_db:.2f} dB, per_sample_snr={per_sample_snr}"
    )

    # 3. 在含噪 Raw 上重算 STFT 特征
    X_test_stft_noisy = stft_features_from_raw(X_test_raw_noisy, cfg)

    print(f"[NoiseData] X_train_stft shape      = {X_train_stft.shape}")
    print(f"[NoiseData] X_test_stft_noisy shape = {X_test_stft_noisy.shape}")
    print(f"[NoiseData] X_test_raw_noisy shape  = {X_test_raw_noisy.shape}")

    return (
        (X_train_stft, X_train_raw, Y_train),
        (X_test_stft_noisy, X_test_raw_noisy, Y_test),
    )
