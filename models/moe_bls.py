# models/moe_bls.py
import numpy as np
from typing import List, Optional
from .bls import BLSClassifier


# =========================================================================
#  辅助工具：整数预算分配器 (用于自适应分配)
# =========================================================================
def distribute_budget(total_budget: int, weights: List[float], min_val: int = 1) -> List[int]:
    """
    根据权重将 total_budget 分配给 N 个专家，保证总和不变且均为整数。
    """
    weights = np.array(weights)
    if weights.sum() == 0: weights = np.ones_like(weights)
    weights = weights / (weights.sum() + 1e-9)
    n_experts = len(weights)

    allocations = np.floor(total_budget * weights).astype(int)
    allocations = np.maximum(allocations, min_val)

    current_sum = allocations.sum()
    diff = total_budget - current_sum

    if diff > 0:
        indices = np.argsort(weights)[::-1]
        for i in range(diff): allocations[indices[i % n_experts]] += 1
    elif diff < 0:
        diff = abs(diff)
        indices = np.argsort(weights)
        for i in range(diff * 2):
            idx = indices[i % n_experts]
            if diff == 0: break
            if allocations[idx] > min_val:
                allocations[idx] -= 1
                diff -= 1
    return allocations.tolist()


# =========================================================================
#  MoE 对照组：均匀分配 (Uniform)
# =========================================================================
class MoEBLSUniform:
    """
    均匀 MoE：所有专家参数一致，最终结果取平均。
    用于证明"仅仅分频段"是不够的，必须要有自适应和协同。
    """

    def __init__(
            self,
            input_dims_per_band: List[int],
            num_classes: int,
            total_feature_win_num: int,
            feature_nodes_per_win: int,
            total_enhance_nodes: int,
            reg_lambda: float,
            random_state: Optional[int] = None,
    ):
        self.num_classes = num_classes
        self.num_experts = len(input_dims_per_band)

        # 均匀瓜分预算
        self.expert_feature_win_num = max(1, total_feature_win_num // self.num_experts)
        self.expert_enhance_nodes = max(1, total_enhance_nodes // self.num_experts)
        self.feature_nodes_per_win = feature_nodes_per_win
        self.reg_lambda = reg_lambda
        self.base_seed = random_state or 0
        self.experts: List[BLSClassifier] = []

    def fit(self, X_bands: List[np.ndarray], y: np.ndarray):
        self.experts = []
        for i in range(self.num_experts):
            bls = BLSClassifier(
                input_dim=X_bands[i].shape[1],
                num_classes=self.num_classes,
                feature_win_num=self.expert_feature_win_num,
                feature_nodes_per_win=self.feature_nodes_per_win,
                enhance_nodes=self.expert_enhance_nodes,
                reg_lambda=self.reg_lambda,
                random_state=self.base_seed + i,
            )
            bls.fit(X_bands[i], y)
            self.experts.append(bls)
        return self

    def predict_proba(self, X_bands: List[np.ndarray]) -> np.ndarray:
        probs = [exp.predict_proba(x) for exp, x in zip(self.experts, X_bands)]
        return sum(probs) / len(probs)

    def predict(self, X_bands: List[np.ndarray]) -> np.ndarray:
        return self.predict_proba(X_bands).argmax(axis=1)