import numpy as np
from sklearn import preprocessing
from numpy import random
from scipy import linalg as LA
from scipy import stats, fft
from typing import Optional

# 引入项目配置
from configs.config_moe_bls import MoeBLSConfig


# =============================================================================
# 核心算法辅助函数 (严格对齐 SFEBLN.py 源码逻辑)
# =============================================================================

def tansig(x):
    return (2 / (1 + np.exp(-2 * x))) - 1


def pinv(A, reg):
    """源码公式: np.mat(reg*I + A.T@A).I @ A.T"""
    return np.linalg.solve(reg * np.eye(A.shape[1]) + A.T @ A, A.T)


def shrinkage(a, b):
    return np.maximum(a - b, 0) - np.maximum(-a - b, 0)


def sparse_bls(A, b):
    """基于 ADMM 的稀疏求解器 (对齐源码变量名与逻辑)"""
    lam = 0.001
    itrs = 50
    AA = A.T @ A
    m = A.shape[1]
    n = b.shape[1]
    wk = np.zeros([m, n])
    ok = np.zeros([m, n])
    uk = np.zeros([m, n])

    try:
        L1 = np.linalg.inv(AA + np.eye(m))
    except np.linalg.LinAlgError:
        L1 = np.linalg.pinv(AA + np.eye(m))

    L2 = L1 @ A.T @ b
    for i in range(itrs):
        ck = L2 + L1 @ (ok - uk)
        ok = shrinkage(ck + uk, lam)
        uk = uk + ck - ok
        wk = ok
    return wk


def real2complex(x):
    cutoff = int(x.shape[-1] / 2)
    return x[:, :cutoff] + 1j * x[:, cutoff:]


def sp_fft(x, n):
    x_fft = fft.fft(x, n, axis=1)
    return np.concatenate((x_fft.real, x_fft.imag), axis=1)


def sp_dct(x, n):
    x_dct = fft.dct(x, n=n, axis=1)
    return np.concatenate((x_dct.real, x_dct.imag), axis=1)


# =============================================================================
# SFEBLN 分类器类 (重构版)
# =============================================================================

class SFEBLNClassifier:
    def __init__(self, num_classes=10, feature_win_num=10, feature_nodes_per_win=20,
                 enhance_nodes=3000, sp_nodes=100, shrink_coeff=0.5, reg_lambda=2 ** -24,
                 fftn=32, ex_fftn=2048, random_state=None):
        self.num_classes = num_classes
        self.N2 = feature_win_num
        self.N1 = feature_nodes_per_win
        self.N3 = enhance_nodes
        self.N4 = sp_nodes
        self.s = shrink_coeff
        self.C = reg_lambda
        self.fftn = fftn
        self.ex_fftn = ex_fftn
        self.rng_seed = random_state

        self.Beta1OfEachWindow = []
        self.distOfMaxAndMin = []
        self.minOfEachWindow = []
        self.weightOfEnhanceLayer = None
        self.weightOfSPLayer = None
        self.parameterOfShrink_Enhance = None  # 即源码中的 parameterOfShrink
        self.parameterOfShrink_SP = None
        self.OutputWeight = None
        # 归一化范围定义
        self.ymin, self.ymax = 0, 1

    @classmethod
    def from_config(cls, cfg: MoeBLSConfig, random_state=None):
        return cls(num_classes=cfg.NUM_CLASSES, feature_win_num=cfg.SFEBLN_FEATURE_WIN_NUM,
                   feature_nodes_per_win=cfg.SFEBLN_NODES_PER_WIN, enhance_nodes=cfg.SFEBLN_ENHANCE_NODES,
                   sp_nodes=cfg.SFEBLN_SP_NODES, shrink_coeff=cfg.SFEBLN_SHRINK, reg_lambda=cfg.SFEBLN_REG,
                   fftn=cfg.SFEBLN_FFT_N, ex_fftn=cfg.SFEBLN_EX_FFT_N, random_state=random_state)

    def _get_external_features(self, X):
        """对齐源码：先 complex 再 fft 再 scale"""
        X_complex = real2complex(X)
        X_ex = sp_fft(X_complex, n=self.ex_fftn)
        X_ex = preprocessing.scale(X_ex, axis=1)
        # 增加偏置项 0.1
        X_ex_bias = np.hstack([X_ex, 0.1 * np.ones((X_ex.shape[0], 1))])
        return X_complex, X_ex_bias

    def fit(self, X, y):
        X_complex, FeatureOfInputDataWithBias = self._get_external_features(X)

        # 转换标签为 One-hot (对齐源码逻辑)
        if y.ndim == 1 or y.shape[1] == 1:
            y_onehot = np.zeros((y.size, self.num_classes))
            y_onehot[np.arange(y.size), y.astype(int).ravel()] = 1
        else:
            y_onehot = y

        # 1. Feature Mapping Layer
        OutputOfFeatureMappingLayer = np.zeros([X.shape[0], self.N2 * self.N1])
        for i in range(self.N2):
            random.seed(i + (self.rng_seed or 0))  # 严格对齐源码 seed(i)
            weight = 2 * random.randn(FeatureOfInputDataWithBias.shape[1], self.N1) - 1
            feat_win = FeatureOfInputDataWithBias @ weight
            scaler = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit(feat_win)
            feat_win_norm = scaler.transform(feat_win)

            beta = sparse_bls(feat_win_norm, FeatureOfInputDataWithBias).T
            self.Beta1OfEachWindow.append(beta)

            out_win = FeatureOfInputDataWithBias @ beta
            d_max_min = np.max(out_win, axis=0) - np.min(out_win, axis=0)
            d_max_min[d_max_min == 0] = 1.0  # 防止除零
            self.distOfMaxAndMin.append(d_max_min)
            self.minOfEachWindow.append(np.min(out_win, axis=0))

            OutputOfFeatureMappingLayer[:, self.N1 * i: self.N1 * (i + 1)] = (out_win - self.minOfEachWindow[
                i]) / d_max_min

        # 2. Enhancement Layer
        Enh_Input = np.hstack([OutputOfFeatureMappingLayer, 0.1 * np.ones((X.shape[0], 1))])
        random.seed(self.rng_seed)
        if self.N2 * self.N1 >= self.N3:
            self.weightOfEnhanceLayer = LA.orth(2 * random.randn(self.N2 * self.N1 + 1, self.N3)) - 1
        else:
            self.weightOfEnhanceLayer = LA.orth(2 * random.randn(self.N2 * self.N1 + 1, self.N3).T - 1).T

        Enh_Temp = Enh_Input @ self.weightOfEnhanceLayer
        self.parameterOfShrink_Enhance = self.s / np.max(Enh_Temp)
        OutputOfEnhanceLayer = tansig(Enh_Temp * self.parameterOfShrink_Enhance)

        # 3. SP Layer
        x_fft = preprocessing.scale(sp_fft(X_complex, n=self.fftn), axis=1)
        x_dct = preprocessing.scale(sp_dct(X_complex, n=self.fftn), axis=1)
        SP_in = np.concatenate((x_fft, x_dct), axis=1)
        SP_Input = np.hstack([SP_in, 0.1 * np.ones((SP_in.shape[0], 1))])

        random.seed(self.rng_seed)
        if SP_in.shape[1] >= self.N4:
            self.weightOfSPLayer = LA.orth(2 * random.randn(SP_Input.shape[1], self.N4)) - 1
        else:
            self.weightOfSPLayer = LA.orth(2 * random.randn(SP_Input.shape[1], self.N4).T - 1).T

        SP_Temp = SP_Input @ self.weightOfSPLayer
        self.parameterOfShrink_SP = self.s / np.max(SP_Temp)

        # --- 关键对齐点：源码训练时使用了增强层的参数 ---
        OutputOfSPLayer = tansig(SP_Temp * self.parameterOfShrink_Enhance)

        # 4. Output
        Final_Input = np.hstack([OutputOfFeatureMappingLayer, OutputOfEnhanceLayer, OutputOfSPLayer])
        self.OutputWeight = pinv(Final_Input, self.C) @ y_onehot
        return self

    def predict_proba(self, X):
        X_complex, FeatureOfBias = self._get_external_features(X)

        # Mapping Test
        Mapping_Test = np.zeros([X.shape[0], self.N2 * self.N1])
        for i in range(self.N2):
            out_win_test = FeatureOfBias @ self.Beta1OfEachWindow[i]
            # 严格遵循源码的测试归一化公式
            Mapping_Test[:, self.N1 * i: self.N1 * (i + 1)] = (self.ymax - self.ymin) * \
                                                              (out_win_test - self.minOfEachWindow[i]) / \
                                                              self.distOfMaxAndMin[i] - self.ymin

        # Enhance Test
        Enh_Out = tansig((np.hstack([Mapping_Test, 0.1 * np.ones((X.shape[0], 1))]) @ \
                          self.weightOfEnhanceLayer) * self.parameterOfShrink_Enhance)

        # SP Test
        x_fft_test = preprocessing.scale(sp_fft(X_complex, n=self.fftn), axis=1)
        x_dct_test = preprocessing.scale(sp_dct(X_complex, n=self.fftn), axis=1)
        SP_in_test = np.concatenate((x_fft_test, x_dct_test), axis=1)
        SP_Out = tansig((np.hstack([SP_in_test, 0.1 * np.ones((X.shape[0], 1))]) @ \
                         self.weightOfSPLayer) * self.parameterOfShrink_SP)

        # --- 源码这里返回的是线性输出，为了兼容 sklearn 我们手动转概率 ---
        Output = np.hstack([Mapping_Test, Enh_Out, SP_Out]) @ self.OutputWeight
        exp_score = np.exp(Output - np.max(Output, axis=1, keepdims=True))
        return exp_score / np.sum(exp_score, axis=1, keepdims=True)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)