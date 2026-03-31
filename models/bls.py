# models/bls.py
import numpy as np
from typing import Optional


def _one_hot(y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    将标签 y 转为 one-hot 矩阵 (N, C)
    """
    y = y.astype(int).ravel()
    N = y.shape[0]
    Y_onehot = np.zeros((N, num_classes), dtype=np.float64)
    Y_onehot[np.arange(N), y] = 1.0
    return Y_onehot


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


class BLSClassifier:
    """
    宽度学习系统（BLS）分类器，简化实现版本：

      H = [Z_f, Z_e]
      beta = (H^T H + λI)^(-1) H^T Y

      其中：
        Z_f: 多个特征节点窗口拼接
        Z_e: 增强节点

    可作为：
      - 单模型基线
      - 4 个专家中的每一个
      - 门控网络（输入为专家输出）
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        feature_win_num: int = 10,
        feature_nodes_per_win: int = 20,
        enhance_nodes: int = 200,
        reg_lambda: float = 2**-30,
        random_state: Optional[int] = None,
    ):
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.feature_win_num = feature_win_num
        self.feature_nodes_per_win = feature_nodes_per_win
        self.enhance_nodes = enhance_nodes
        self.reg_lambda = reg_lambda
        self.rng = np.random.RandomState(random_state)

        self.W_feature = None
        self.b_feature = None
        self.W_enhance = None
        self.b_enhance = None
        self.beta = None

    def _gen_feature_nodes(self, X: np.ndarray) -> np.ndarray:
        """
        生成特征节点输出 Z_f: (N, feature_win_num * feature_nodes_per_win)
        """
        X = X.astype(np.float64)
        N = X.shape[0]
        Z_list = []

        if self.W_feature is None:
            self.W_feature = []
            self.b_feature = []
            for _ in range(self.feature_win_num):
                W = self.rng.uniform(-1, 1,
                                     size=(self.input_dim, self.feature_nodes_per_win))
                b = self.rng.uniform(-1, 1,
                                     size=(self.feature_nodes_per_win,))
                self.W_feature.append(W)
                self.b_feature.append(b)

        for i in range(self.feature_win_num):
            W = self.W_feature[i]
            b = self.b_feature[i]
            Z_i = _sigmoid(X @ W + b)  # (N, feature_nodes_per_win)
            Z_list.append(Z_i)

        Z_f = np.concatenate(Z_list, axis=1)  # (N, F)
        return Z_f

    def _gen_enhance_nodes(self, Z_f: np.ndarray) -> np.ndarray:
        """
        生成增强节点输出 Z_e: (N, enhance_nodes)
        """
        N, D_f = Z_f.shape
        if self.enhance_nodes <= 0:
            return np.zeros((N, 0), dtype=np.float64)

        if self.W_enhance is None:
            self.W_enhance = self.rng.uniform(-1, 1,
                                              size=(D_f, self.enhance_nodes))
            self.b_enhance = self.rng.uniform(-1, 1,
                                              size=(self.enhance_nodes,))
        Z_e = _sigmoid(Z_f @ self.W_enhance + self.b_enhance)
        return Z_e

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        拟合 BLS 模型：
          1. 生成特征节点 Z_f
          2. 生成增强节点 Z_e
          3. 拼接 H = [Z_f, Z_e]
          4. 求解 beta = (H^T H + λI)^(-1) H^T Y
        """
        X = X.astype(np.float64)
        Y = _one_hot(y, self.num_classes)  # (N, C)

        Z_f = self._gen_feature_nodes(X)   # (N, F)
        Z_e = self._gen_enhance_nodes(Z_f) # (N, E)
        H = np.concatenate([Z_f, Z_e], axis=1)  # (N, F+E)

        Ht = H.T
        regI = self.reg_lambda * np.eye(H.shape[1])
        self.beta = np.linalg.solve(Ht @ H + regI, Ht @ Y)  # (F+E, C)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别概率 (N, C)
        """
        if self.beta is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")

        X = X.astype(np.float64)
        Z_f = self._gen_feature_nodes(X)
        Z_e = self._gen_enhance_nodes(Z_f)
        H = np.concatenate([Z_f, Z_e], axis=1)
        logits = H @ self.beta  # (N, C)

        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测类别标签 (N,)
        """
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)
