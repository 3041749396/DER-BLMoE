# models/moe_node_attention_gate.py
# -*- coding: utf-8 -*-
"""
Medium learnable node-attention gate for DER-BLMoE ablation.

This version is intentionally between the original tiny gate and the previous
heavy gate. It keeps the original STFT, sub-band PCA, variance-aware BLS experts
unchanged, and only replaces DER fusion with a moderately sized trainable fusion
module.

Default target for the current 5-expert DER-BLMoE setting:
    gate_num_heads = 4
    gate_hidden_dim = 1024
    gate_mlp_depth = 2
    gate_shared_scorer = True

With typical node dimensions [70, 212, 610, 346, 92], this adds roughly:
    gate params  ≈ 23.7K
    gate FLOPs   ≈ 0.185M/sample
    total FLOPs  ≈ original DER FLOPs + 0.185M/sample

If you want total FLOPs closer to 0.5M, use --gate-hidden-dim 1408, which raises
extra gate params to roughly 32K and total FLOPs to about 0.49M/sample.
"""

from __future__ import annotations

import copy
from typing import List, Optional, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError as e:  # pragma: no cover
    raise ImportError("PyTorch is required for the learnable node-attention gate.") from e

from .bls import BLSClassifier
from .moe_bls import distribute_budget


_EPS = 1e-12


def _stable_softmax_np(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / (np.sum(ex, axis=axis, keepdims=True) + _EPS)


class _TorchNodeAttentionGate(nn.Module):
    """
    Medium learnable gate over frozen expert outputs.

    For expert i:
        A_i = softmax(Theta_i, dim=-1)               # h attention heads over D_i nodes
        c_i,h = sum_j A_i,hj * H_ij                  # h node-attended contexts
        q_i = [c_i, p_i, max(p_i), entropy(p_i)]     # gate feature vector
        s_i = MLP(q_i)                               # expert score
        w_i = softmax_i(s_i)                         # expert fusion weight
        P_final = sum_i w_i * p_i

    Compared with the original tiny gate, this module has multi-head node
    attention and a hidden-layer scorer. Compared with a heavy per-expert MLP
    gate, the scorer is shared across experts by default, so the parameter count
    stays around tens of thousands rather than hundreds of thousands.
    """

    def __init__(
        self,
        node_dims: List[int],
        num_classes: int,
        top_k: int = 0,
        num_heads: int = 4,
        hidden_dim: int = 1024,
        mlp_depth: int = 2,
        dropout: float = 0.0,
        use_entropy_features: bool = True,
        shared_scorer: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()
        self.node_dims = [int(d) for d in node_dims]
        self.num_experts = len(self.node_dims)
        self.num_classes = int(num_classes)
        self.top_k = int(top_k or 0)
        self.num_heads = int(max(1, num_heads))
        self.hidden_dim = int(max(1, hidden_dim))
        self.mlp_depth = int(max(1, mlp_depth))
        self.dropout = float(max(0.0, dropout))
        self.use_entropy_features = bool(use_entropy_features)
        self.shared_scorer = bool(shared_scorer)

        # Multi-head node attention, one matrix per expert because D_i may differ.
        self.node_attn_logits = nn.ParameterList([
            nn.Parameter(torch.zeros(self.num_heads, d, dtype=torch.float32))
            for d in self.node_dims
        ])

        extra_dim = 2 if self.use_entropy_features else 0  # max_prob and entropy
        self.scorer_input_dim = self.num_heads + self.num_classes + extra_dim

        if self.shared_scorer:
            self.score_mlp = self._build_score_mlp(self.scorer_input_dim, self.hidden_dim, self.mlp_depth, self.dropout)
            self.score_mlps = None
            self._init_mlp(self.score_mlp, init_scale)
        else:
            self.score_mlp = None
            self.score_mlps = nn.ModuleList([
                self._build_score_mlp(self.scorer_input_dim, self.hidden_dim, self.mlp_depth, self.dropout)
                for _ in range(self.num_experts)
            ])
            for mlp in self.score_mlps:
                self._init_mlp(mlp, init_scale)

    @staticmethod
    def _build_score_mlp(in_dim: int, hidden_dim: int, mlp_depth: int, dropout: float) -> nn.Sequential:
        layers: List[nn.Module] = []
        if mlp_depth <= 1:
            layers.append(nn.Linear(in_dim, 1))
            return nn.Sequential(*layers)

        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))

        for _ in range(mlp_depth - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout(p=dropout))

        layers.append(nn.Linear(hidden_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_mlp(mlp: nn.Sequential, init_scale: float = 1.0):
        for module in mlp.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=float(init_scale))
                nn.init.zeros_(module.bias)

    def _apply_topk(self, scores: torch.Tensor) -> torch.Tensor:
        if self.top_k <= 0 or self.top_k >= self.num_experts:
            return scores
        idx = torch.topk(scores, k=self.top_k, dim=1).indices
        mask = torch.zeros_like(scores)
        mask.scatter_(1, idx, 1.0)
        return scores.masked_fill(mask <= 0, -1e9)

    def _score_feature_for_expert(self, H: torch.Tensor, p_i: torch.Tensor, i: int) -> torch.Tensor:
        # H: [B, D_i], attn: [num_heads, D_i]
        attn = torch.softmax(self.node_attn_logits[i], dim=1)
        contexts = H @ attn.transpose(0, 1)  # [B, num_heads]

        if self.use_entropy_features:
            p_safe = torch.clamp(p_i, min=1e-8, max=1.0)
            max_prob = torch.max(p_safe, dim=1, keepdim=True).values
            entropy = -torch.sum(p_safe * torch.log(p_safe), dim=1, keepdim=True)
            feat = torch.cat([contexts, p_i, max_prob, entropy], dim=1)
        else:
            feat = torch.cat([contexts, p_i], dim=1)
        return feat

    def forward(self, H_list: List[torch.Tensor], p_stack: torch.Tensor, return_weights: bool = False):
        scores = []
        for i, H in enumerate(H_list):
            p_i = p_stack[:, i, :]
            feat = self._score_feature_for_expert(H, p_i, i)
            if self.shared_scorer:
                score_i = self.score_mlp(feat).squeeze(1)
            else:
                score_i = self.score_mlps[i](feat).squeeze(1)
            scores.append(score_i)

        score_mat = torch.stack(scores, dim=1)
        score_mat = self._apply_topk(score_mat)
        weights = torch.softmax(score_mat, dim=1)
        final_probs = torch.sum(weights.unsqueeze(-1) * p_stack, dim=1)
        final_probs = torch.clamp(final_probs, min=1e-8, max=1.0)
        if return_weights:
            return final_probs, weights
        return final_probs

    def num_trainable_params(self) -> int:
        return int(sum(p.numel() for p in self.parameters() if p.requires_grad))

    def _mlp_flops_once(self) -> int:
        last = self.scorer_input_dim
        total = 0
        if self.mlp_depth <= 1:
            total += 2 * last * 1
        else:
            total += 2 * last * self.hidden_dim
            last = self.hidden_dim
            for _ in range(self.mlp_depth - 2):
                total += 2 * last * self.hidden_dim
                last = self.hidden_dim
            total += 2 * last * 1
        return int(total)

    def estimate_flops_per_sample(self) -> int:
        """Approximate extra gate FLOPs per sample; excludes frozen BLS experts."""
        total = 0
        for D in self.node_dims:
            total += 2 * int(D) * self.num_heads  # multi-head node contexts
            if self.use_entropy_features:
                total += 4 * self.num_classes     # max-prob/entropy rough cost
            total += self._mlp_flops_once()       # scorer applied once per expert
        total += 4 * self.num_experts
        total += 2 * self.num_experts * self.num_classes
        return int(total)


class MoEBLSLearnedNodeAttentionGate:
    """
    BLS-MoE with a medium learnable node-attention gate.

    For a fusion-only ablation, call `copy_experts_from(der_model)` and then
    `fit_gate_only(...)`; this freezes the DER-BLMoE experts and trains only
    the attention fusion gate.
    """

    def __init__(
        self,
        input_dims_per_band: List[int],
        num_classes: int,
        total_expert_feature_win_num: int,
        expert_feature_nodes_per_win: int,
        total_expert_enhance_nodes: int,
        expert_reg_lambda: float,
        top_k: int = 0,
        importance_scores: Optional[List[float]] = None,
        random_state: Optional[int] = None,
        gate_epochs: int = 300,
        gate_lr: float = 1e-3,
        gate_weight_decay: float = 1e-4,
        gate_batch_size: int = 256,
        gate_patience: int = 40,
        gate_device: str = "cpu",
        gate_num_heads: int = 4,
        gate_hidden_dim: int = 1024,
        gate_mlp_depth: int = 2,
        gate_dropout: float = 0.0,
        gate_use_entropy_features: bool = True,
        gate_shared_scorer: bool = True,
        **kwargs,
    ):
        self.input_dims = list(input_dims_per_band)
        self.num_classes = int(num_classes)
        self.num_experts = len(self.input_dims)
        self.base_seed = int(random_state or 2025)

        self.expert_feature_nodes_per_win = int(expert_feature_nodes_per_win)
        self.expert_reg_lambda = float(expert_reg_lambda)

        weights = importance_scores if importance_scores is not None else [1.0] * self.num_experts
        self.wins_per_expert = distribute_budget(int(total_expert_feature_win_num), weights, min_val=2)
        self.nodes_per_expert = distribute_budget(int(total_expert_enhance_nodes), weights, min_val=50)

        self.top_k = int(top_k or 0)
        self.gate_epochs = int(gate_epochs)
        self.gate_lr = float(gate_lr)
        self.gate_weight_decay = float(gate_weight_decay)
        self.gate_batch_size = int(gate_batch_size)
        self.gate_patience = int(gate_patience)
        self.gate_device = gate_device if gate_device in ("cpu", "cuda") else "cpu"
        if self.gate_device == "cuda" and not torch.cuda.is_available():
            self.gate_device = "cpu"

        self.gate_num_heads = int(gate_num_heads)
        self.gate_hidden_dim = int(gate_hidden_dim)
        self.gate_mlp_depth = int(gate_mlp_depth)
        self.gate_dropout = float(gate_dropout)
        self.gate_use_entropy_features = bool(gate_use_entropy_features)
        self.gate_shared_scorer = bool(gate_shared_scorer)

        self.experts: List[BLSClassifier] = []
        self.gate: Optional[_TorchNodeAttentionGate] = None
        self.node_dims: Optional[List[int]] = None

        self.last_gate_weights = None
        self.last_train_history = []

    @staticmethod
    def _expert_nodes(expert: BLSClassifier, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float64)
        Z_f = expert._gen_feature_nodes(X)
        Z_e = expert._gen_enhance_nodes(Z_f)
        H = np.concatenate([Z_f, Z_e], axis=1)
        return H.astype(np.float32)

    @staticmethod
    def _expert_probs(expert: BLSClassifier, X: np.ndarray) -> np.ndarray:
        p = expert.predict_proba(X.astype(np.float64))
        p = np.clip(p, _EPS, 1.0)
        p = p / (p.sum(axis=1, keepdims=True) + _EPS)
        return p.astype(np.float32)

    def _collect_gate_inputs(self, X_bands: List[np.ndarray]) -> Tuple[List[np.ndarray], np.ndarray]:
        if not self.experts:
            raise RuntimeError("Experts are not available. Fit the model or copy experts first.")
        H_list = []
        P_list = []
        for expert, xb in zip(self.experts, X_bands):
            H_list.append(self._expert_nodes(expert, xb))
            P_list.append(self._expert_probs(expert, xb))
        p_stack = np.stack(P_list, axis=1).astype(np.float32)
        return H_list, p_stack

    def fit(self, X_tr_bands: List[np.ndarray], y_tr: np.ndarray, X_val_bands=None, y_val=None, **kwargs):
        self.experts = []
        for i in range(self.num_experts):
            bls = BLSClassifier(
                input_dim=self.input_dims[i],
                num_classes=self.num_classes,
                feature_win_num=self.wins_per_expert[i],
                feature_nodes_per_win=self.expert_feature_nodes_per_win,
                enhance_nodes=self.nodes_per_expert[i],
                reg_lambda=self.expert_reg_lambda,
                random_state=self.base_seed + i,
            )
            bls.fit(X_tr_bands[i], y_tr)
            self.experts.append(bls)
        return self.fit_gate_only(X_tr_bands, y_tr, X_val_bands, y_val)

    def copy_experts_from(self, other_model) -> "MoEBLSLearnedNodeAttentionGate":
        self.experts = copy.deepcopy(other_model.experts)
        if hasattr(other_model, "wins_per_expert"):
            self.wins_per_expert = list(other_model.wins_per_expert)
        if hasattr(other_model, "nodes_per_expert"):
            self.nodes_per_expert = list(other_model.nodes_per_expert)
        return self

    def fit_gate_only(self, X_tr_bands: List[np.ndarray], y_tr: np.ndarray, X_val_bands=None, y_val=None):
        y_tr = np.asarray(y_tr, dtype=np.int64).ravel()
        H_tr, P_tr = self._collect_gate_inputs(X_tr_bands)
        self.node_dims = [h.shape[1] for h in H_tr]

        H_val, P_val, y_val_arr = None, None, None
        if X_val_bands is not None and y_val is not None:
            y_val_arr = np.asarray(y_val, dtype=np.int64).ravel()
            H_val, P_val = self._collect_gate_inputs(X_val_bands)

        torch.manual_seed(self.base_seed)
        np.random.seed(self.base_seed)

        self.gate = _TorchNodeAttentionGate(
            node_dims=self.node_dims,
            num_classes=self.num_classes,
            top_k=self.top_k,
            num_heads=self.gate_num_heads,
            hidden_dim=self.gate_hidden_dim,
            mlp_depth=self.gate_mlp_depth,
            dropout=self.gate_dropout,
            use_entropy_features=self.gate_use_entropy_features,
            shared_scorer=self.gate_shared_scorer,
        ).to(self.gate_device)

        opt = torch.optim.AdamW(
            self.gate.parameters(),
            lr=self.gate_lr,
            weight_decay=self.gate_weight_decay,
        )

        N = len(y_tr)
        batch_size = max(1, min(self.gate_batch_size, N))
        best_state = copy.deepcopy(self.gate.state_dict())
        best_score = -np.inf
        patience_left = self.gate_patience
        self.last_train_history = []

        H_tr_t = [torch.from_numpy(h).to(self.gate_device) for h in H_tr]
        P_tr_t = torch.from_numpy(P_tr).to(self.gate_device)
        y_tr_t = torch.from_numpy(y_tr).to(self.gate_device)

        if H_val is not None:
            H_val_t = [torch.from_numpy(h).to(self.gate_device) for h in H_val]
            P_val_t = torch.from_numpy(P_val).to(self.gate_device)
            y_val_t = torch.from_numpy(y_val_arr).to(self.gate_device)
        else:
            H_val_t = P_val_t = y_val_t = None

        rng = np.random.RandomState(self.base_seed + 777)

        for epoch in range(1, self.gate_epochs + 1):
            self.gate.train()
            order = rng.permutation(N)
            epoch_loss = 0.0
            correct = 0

            for start in range(0, N, batch_size):
                idx_np = order[start:start + batch_size]
                idx_t = torch.from_numpy(idx_np).long().to(self.gate_device)
                H_b = [h.index_select(0, idx_t) for h in H_tr_t]
                P_b = P_tr_t.index_select(0, idx_t)
                y_b = y_tr_t.index_select(0, idx_t)

                opt.zero_grad(set_to_none=True)
                final_probs = self.gate(H_b, P_b, return_weights=False)
                loss = F.nll_loss(torch.log(final_probs), y_b)
                loss.backward()
                opt.step()

                epoch_loss += float(loss.item()) * len(idx_np)
                correct += int((final_probs.argmax(dim=1) == y_b).sum().item())

            train_loss = epoch_loss / max(1, N)
            train_acc = correct / max(1, N)

            if H_val_t is not None:
                val_acc = self._eval_torch_acc(H_val_t, P_val_t, y_val_t)
                monitor = val_acc
            else:
                val_acc = np.nan
                monitor = train_acc

            self.last_train_history.append({
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": float(train_acc),
                "val_acc": float(val_acc) if not np.isnan(val_acc) else np.nan,
            })

            if monitor > best_score + 1e-8:
                best_score = monitor
                best_state = copy.deepcopy(self.gate.state_dict())
                patience_left = self.gate_patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    break

        self.gate.load_state_dict(best_state)
        self.gate.to("cpu")
        self.gate_device = "cpu"
        return self

    @torch.no_grad()
    def _eval_torch_acc(self, H_list_t, P_t, y_t) -> float:
        self.gate.eval()
        final_probs = self.gate(H_list_t, P_t, return_weights=False)
        return float((final_probs.argmax(dim=1) == y_t).float().mean().item())

    def predict_proba(self, X_bands: List[np.ndarray], batch_size: int = 2048) -> np.ndarray:
        if self.gate is None:
            raise RuntimeError("Gate is not trained. Call fit() or fit_gate_only() first.")
        H_all, P_all = self._collect_gate_inputs(X_bands)
        N = P_all.shape[0]
        out_list = []
        w_list = []

        self.gate.eval()
        self.gate.to("cpu")
        with torch.no_grad():
            for start in range(0, N, batch_size):
                end = min(N, start + batch_size)
                H_b = [torch.from_numpy(h[start:end]).float() for h in H_all]
                P_b = torch.from_numpy(P_all[start:end]).float()
                probs_b, weights_b = self.gate(H_b, P_b, return_weights=True)
                out_list.append(probs_b.cpu().numpy())
                w_list.append(weights_b.cpu().numpy())

        final_probs = np.concatenate(out_list, axis=0)
        self.last_gate_weights = np.concatenate(w_list, axis=0)
        final_probs = np.clip(final_probs, _EPS, 1.0)
        final_probs = final_probs / (final_probs.sum(axis=1, keepdims=True) + _EPS)
        return final_probs

    def predict(self, X_bands: List[np.ndarray]) -> np.ndarray:
        return self.predict_proba(X_bands).argmax(axis=1)

    def count_gate_params(self) -> int:
        if self.gate is None:
            return 0
        return self.gate.num_trainable_params()

    def count_gate_flops(self) -> int:
        if self.gate is None:
            return 0
        return int(self.gate.estimate_flops_per_sample())

    def count_total_params_with_gate(self) -> int:
        total = self.count_gate_params()
        for exp in self.experts:
            if exp.W_feature is not None:
                for W, b in zip(exp.W_feature, exp.b_feature):
                    total += W.size + b.size
            if exp.W_enhance is not None:
                total += exp.W_enhance.size + exp.b_enhance.size
            if exp.beta is not None:
                total += exp.beta.size
        return int(total)
