import os
from dataclasses import dataclass

# 自动根据当前项目位置推断工程根目录，兼容 Windows / Linux
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) #


@dataclass
class MoeBLSConfig:
    # ================= 1. 基础路径与数据文件 =================
    # 数据根目录：统一放所有 .npy 数据
    DATA_ROOT: str = os.path.join(PROJECT_ROOT, "Datasets") #
    # 结果根目录：模型、对比结果等
    RESULTS_ROOT: str = os.path.join(PROJECT_ROOT, "compare_results") #

    # --- 传统 STFT 特征文件（现在主流程不用它们，只保留兼容） ---
    X_TRAIN_FILE: str = "X_train_10Class.npy" #
    Y_TRAIN_FILE: str = "Y_train_10Class.npy" #
    X_TEST_FILE: str = "X_test_10Class.npy" #
    Y_TEST_FILE: str = "Y_test_10Class.npy" #

    # --- RAW IQ 文件（Matlab 预处理导出的 Real+Imag 拼接） ---
    X_TRAIN_RAW_FILE: str = "X_train_10Class.npy" #
    X_TEST_RAW_FILE: str = "X_test_10Class.npy" #

    # ================= 2. 任务与模型规模 =================
    # 注意：如果后面切换 20Class / 30Class / 90Class，需要手动改这里
    NUM_CLASSES: int = 10 #
    # MoE 的逻辑专家数（= STFT 频段数）
    NUM_LOGICAL_EXPERTS: int = 5 #

    # 是否在 STFT 特征上做 PCA
    USE_PCA: bool = True #
    # MoE 用的 PCA 维度
    PCA_DIM_MOE: int = 16
    # BLS / baseline 用的 PCA 维度
    PCA_DIM_BASELINE: int = NUM_LOGICAL_EXPERTS * PCA_DIM_MOE #

    # 全局随机种子（训练脚本会在此基础上 + run_idx）
    GLOBAL_SEED: int = 2025 #

    # ================= 3. BLS / MoE-BLS 超参数 =================
    # 每个专家使用的 BLS 结构（频段内特征 BLS）
    TOTAL_WIN_NUM: int = 12 #
    NODES_PER_WIN: int = 10 #
    TOTAL_ENHANCE_NUM: int = 1200 #
    REG_LAMBDA: float = 1e-4 #

    # [DEPRECATED] 全局专家 (Global Expert) 配置
    # -------------------------------------------------------
    # 仅为兼容旧脚本保留，不再被任何训练 / 推理代码使用。
    USE_GLOBAL_EXPERT: bool = False #
    GLOBAL_EXPERT_WEIGHT: float = 0.0 #
    # -------------------------------------------------------

    # （旧版本用的）全局 BLS 占比，当前代码不使用，只保留字段避免报错
    GLOBAL_RATIO: float = 0.2 #

    # 旧版掩码增强比例——当前训练流程完全不再使用，只保留兼容
    MASK_AUGMENT_RATIO: float = 0.2 #

    # Gate（门控网络）的 BLS 结构（当前熵门控不显式用到这些结构参数，保留接口）
    GATE_WIN_NUM: int = 5 #
    GATE_NODES_PER_WIN: int = 10 #
    GATE_ENHANCE_NODES: int = 50 #
    GATE_REG_LAMBDA: float = 1e-4 #

    # MoE Top-k 与早停阈值
    MOE_TOP_K: int = 3 #
    EARLY_EXIT_THRESHOLD: float = 0.5 #

    # ================= 4. SFEBLN 参数 =================
    SFEBLN_FEATURE_WIN_NUM: int = 10 # 对应 N2
    SFEBLN_NODES_PER_WIN: int = 20  # 对应 N1
    SFEBLN_ENHANCE_NODES: int = 2000 # 对应 N3
    SFEBLN_SP_NODES: int = 200      # 对应 N4
    SFEBLN_SHRINK: float = 0.5      # 对应收缩系数 s
    SFEBLN_REG: float = 2 ** -24    # 对应正则化系数 C
    SFEBLN_FFT_N: int = 32          # 信号处理节点支路 FFT 点数
    SFEBLN_EX_FFT_N: int = 1024     # 外部特征提取支路 FFT 点数

    # ================= 5. BLS Baseline 参数 =================
    BLS_FEATURE_WIN_NUM: int = TOTAL_WIN_NUM #
    BLS_FEATURE_NODES_PER_WIN: int = NODES_PER_WIN #
    BLS_ENHANCE_NODES: int = TOTAL_ENHANCE_NUM #
    BLS_REG_LAMBDA: float = 0.0001 #

    # ================= 6. 其它实验相关参数 =================
    # 训练多次（MC 次数）用来做平均性能 / 统计
    INF_BENCH_MC: int = 1 #
    # 老的频段缺失实验里用的 mask 数量，后续新的 band-missing 会单独写脚本
    BAND_MISSING_NUM_MASKS: int = 30 #

    # ================= 7. 深度学习基线参数 =================
    DL_BATCH_SIZE: int = 64 #
    DL_EPOCHS: int = 200 #
    DL_LR: float = 1e-3 #

    # ================= 8. Python 端 STFT 参数（对齐 Matlab） =================
    STFT_NPERSEG: int = 128 #
    STFT_NO_OVERLAP: int = 64 #
    STFT_NFFT: int = 256 #

    def __post_init__(self): #
        # 防御性逻辑：如果外部修改了 RESULTS_ROOT 为 None，这里回退到默认
        if self.RESULTS_ROOT is None: #
            self.RESULTS_ROOT = os.path.join(PROJECT_ROOT, "compare_results") #
        if self.DATA_ROOT is None: #
            self.DATA_ROOT = os.path.join(PROJECT_ROOT, "datasets") #


def get_moe_config() -> MoeBLSConfig:
    return MoeBLSConfig() #