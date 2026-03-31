# models/moe_entropy_gate.py
import numpy as np
from typing import List, Optional
from .bls import BLSClassifier
from .moe_bls import distribute_budget


def stable_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    """
    数值稳定的 Softmax 函数。
    将任意范围的 Logits 映射为和为 1 的概率分布。
    """
    # 减去最大值防止 exp 溢出
    shift_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shift_x)
    return exp_x / (np.sum(exp_x, axis=axis, keepdims=True) + 1e-12)


class MoEBLSEntropyResidualGate:
    """
    DER-BLMoE (Dynamic Entropy Residual Broad Learning Mixture of Experts)

    修改说明：
    1. [Math Fix] 强制对专家输出使用 Softmax，确保计算熵之前的概率分布合法 (Sum=1, >0)。
    2. [Logic Upgrade] 引入动态基准计算，将绝对熵升级为动态熵残差 (Dynamic Entropy Residual)。
    3. [Logic Clean] 移除物理 L2 能量否决机制，完全依赖熵的“软剔除”能力，提高推理速度并消除 PCA 降维带来的物理能量歧义。
    """

    def __init__(
            self,
            input_dims_per_band: List[int],
            num_classes: int,
            total_expert_feature_win_num: int,
            expert_feature_nodes_per_win: int,
            total_expert_enhance_nodes: int,
            expert_reg_lambda: float,
            # 门控参数
            gate_alpha: float = 6.0,  # 控制对熵残差的敏感度
            top_k: int = 3,  # 仅保留 Top-K 专家
            importance_scores: Optional[List[float]] = None,
            random_state: Optional[int] = None,
            **kwargs
    ):
        self.input_dims = input_dims_per_band
        self.num_classes = int(num_classes)
        self.num_experts = len(input_dims_per_band)
        self.base_seed = int(random_state or 2025)

        self.expert_feature_nodes_per_win = int(expert_feature_nodes_per_win)
        self.expert_reg_lambda = float(expert_reg_lambda)

        # 预算分配
        weights = importance_scores if importance_scores is not None else [1.0] * self.num_experts
        self.wins_per_expert = distribute_budget(int(total_expert_feature_win_num), weights, min_val=2)
        self.nodes_per_expert = distribute_budget(int(total_expert_enhance_nodes), weights, min_val=50)

        self.max_entropy = float(np.log(self.num_classes) + 1e-9)
        self.gate_alpha = gate_alpha
        self.top_k = top_k

        self.experts: List[BLSClassifier] = []

        # 可视化记录
        self.last_entropy_norm = None
        self.last_entropy_residual = None  # 记录残差供分析
        self.last_gate_weights = None

    def _compute_gate_weights(self, entropies: np.ndarray):
        """
        基于【动态熵残差】(Dynamic Entropy Residual) 计算门控权重得分。
        公式: Score = exp( alpha * ((Mean_Entropy - Entropy) / Max_Entropy) )
        """
        # 1. 动态基准计算 (计算当前样本在所有专家中的平均熵)
        mean_entropy = entropies.mean(axis=1, keepdims=True)

        # 2. 计算动态熵残差 (Dynamic Entropy Residual)
        # 相对优势 = 环境平均混乱度 - 该专家的实际混乱度
        # 差值为正: 表现优于环境平均; 差值为负: 表现劣于环境平均
        entropy_residual = mean_entropy - entropies

        # 记录归一化前的真实残差供后续可视化或消融实验分析
        self.last_entropy_residual = entropy_residual

        # 将残差相对于最大可能熵进行归一化 (-1 ~ 1 之间)
        residual_norm = entropy_residual / self.max_entropy

        # 3. 基于残差映射得分
        # 注意：残差越大（置信度优势越明显），得分呈指数级升高
        scores = np.exp(self.gate_alpha * residual_norm)

        # 4. Top-K 稀疏化
        if 0 < self.top_k < self.num_experts:
            mask = np.zeros_like(scores)
            # 找到得分最高（残差最大）的 K 个索引
            # argsort 取负号是从大到小排
            topk_idx = np.argsort(-scores, axis=1)[:, :self.top_k]

            # 创建行索引矩阵，用于高级索引赋值
            rows = np.arange(scores.shape[0])[:, None]
            mask[rows, topk_idx] = 1.0

            # 应用 Mask，未选中的专家得分强制置零
            scores *= mask

        return scores

    def fit(
            self,
            X_tr_bands: List[np.ndarray],
            y_tr: np.ndarray,
            **kwargs
    ):
        """
        训练模型：仅训练各个频段的 BLS 专家。由于门控是启发式前向映射，无需反向传播。
        """
        self.experts = []
        for i in range(self.num_experts):
            bls = BLSClassifier(
                input_dim=self.input_dims[i],
                num_classes=self.num_classes,
                feature_win_num=self.wins_per_expert[i],
                feature_nodes_per_win=self.expert_feature_nodes_per_win,
                enhance_nodes=self.nodes_per_expert[i],
                reg_lambda=self.expert_reg_lambda,
                random_state=self.base_seed + i
            )
            bls.fit(X_tr_bands[i], y_tr)
            self.experts.append(bls)

        return self

    def predict_proba(self, X_bands: List[np.ndarray]) -> np.ndarray:
        probs_list = []

        # 1. 获取所有专家的预测概率
        for exp, xb in zip(self.experts, X_bands):
            xb_float = xb.astype(np.float64)

            # 宽学习输出的概率化映射
            raw_output = exp.predict_proba(xb_float)
            p = stable_softmax(raw_output, axis=1)
            probs_list.append(p)

        p_stack = np.stack(probs_list, axis=1)  # (N, Experts, Classes)

        # 2. 计算预测香农熵 (Uncertainty Quantification)
        probs_safe = p_stack + 1e-12
        entropies = -np.sum(probs_safe * np.log(probs_safe), axis=2)  # (N, Experts)
        self.last_entropy_norm = entropies / self.max_entropy

        # 3. 计算基于【动态熵残差】的门控得分
        raw_weights = self._compute_gate_weights(entropies)  # (N, Experts)

        # 4. 全局归一化
        sum_weights = raw_weights.sum(axis=1, keepdims=True) + 1e-12
        final_weights = raw_weights / sum_weights

        self.last_gate_weights = final_weights

        # 5. 加权求和输出最终聚合概率 (Final Aggregation)
        final_probs = (final_weights[:, :, None] * p_stack).sum(axis=1)
        return final_probs

    def predict(self, X_bands: List[np.ndarray]) -> np.ndarray:
        return self.predict_proba(X_bands).argmax(axis=1)