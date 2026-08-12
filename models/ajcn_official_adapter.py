"""
Strict AJCN integration adapter for DER-BLMoE project.

This module intentionally keeps the AJCN paper source under third_party/AJCN-main
and imports a namespaced copy under third_party/ajcn_src. The algorithmic modules
used here are the AJCN source modules:
  - BaseModel / PrunedModel
  - convert_block_to_depthwise
  - PPOAgent / PolicyNetwork / ValueNetwork

Only the data adapter and experiment glue are newly written here, because the
original AJCN code expects 32x32x3 ADS-B tensors stored as ADS_train_data.npy,
whereas this project stores raw I/Q samples as [I, Q] concatenated vectors.
"""
from __future__ import annotations

import copy
import os
import random
import contextlib
import io
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# AJCN official / source-level modules, namespaced to avoid collision with this project's models package.
from third_party.ajcn_src.models.base_model import BaseModel
from third_party.ajcn_src.models.pruned_model import PrunedModel
from third_party.ajcn_src.compression.depthwise_conversion import convert_block_to_depthwise
from third_party.ajcn_src.rl.ppo_agent import PPOAgent

AJCN_LAYER_NAMES: List[str] = [
    "conv1",
    "res_block1.conv1",
    "res_block1.conv2",
    "res_block2.conv1",
    "res_block2.conv2",
    "res_block3.conv1",
    "res_block3.conv2",
]
AJCN_BLOCK_NAMES: List[str] = ["res_block1", "res_block2", "res_block3"]
AJCN_PRUNE_OPTIONS: List[float] = [round(i * 0.1, 1) for i in range(10)]


def set_ajcn_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resample_1d(x: np.ndarray, target_len: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32).ravel()
    if x.size == target_len:
        return x.astype(np.float32, copy=False)
    if x.size == 0:
        return np.zeros(target_len, dtype=np.float32)
    src = np.linspace(0.0, 1.0, num=x.size, dtype=np.float32)
    dst = np.linspace(0.0, 1.0, num=target_len, dtype=np.float32)
    return np.interp(dst, src, x).astype(np.float32)


def raw_iq_to_ajcn_tensor(X_raw: np.ndarray, image_size: int = 32, normalize: bool = True) -> np.ndarray:
    """
    Convert this project's raw I/Q representation to AJCN's official 32x32x3 tensor input.

    Input X_raw is assumed to be [I_part, Q_part] concatenated with shape (N, 2L).
    Output shape is (N, 3, 32, 32), where the three channels are resampled I, Q,
    and magnitude. This is the only task-specific adapter needed to evaluate the
    official AJCN architecture on this project's dataset.
    """
    X_raw = np.asarray(X_raw, dtype=np.float32)
    if X_raw.ndim != 2 or X_raw.shape[1] < 2:
        raise ValueError(f"X_raw must have shape (N, 2L), got {X_raw.shape}")

    n = X_raw.shape[0]
    L = X_raw.shape[1] // 2
    target = image_size * image_size
    out = np.empty((n, 3, image_size, image_size), dtype=np.float32)

    I = X_raw[:, :L]
    Q = X_raw[:, L:2 * L]
    mag = np.sqrt(I * I + Q * Q)

    for idx in range(n):
        channels = [I[idx], Q[idx], mag[idx]]
        for c, vec in enumerate(channels):
            v = _resample_1d(vec, target)
            if normalize:
                mu = float(v.mean())
                sigma = float(v.std())
                v = (v - mu) / (sigma + 1e-6)
            out[idx, c] = v.reshape(image_size, image_size)
    return out


def make_ajcn_loader(
    X_raw: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    image_size: int = 32,
    normalize: bool = True,
) -> DataLoader:
    X = raw_iq_to_ajcn_tensor(X_raw, image_size=image_size, normalize=normalize)
    y = np.asarray(y, dtype=np.int64).ravel()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def _model_logits(output):
    return output[0] if isinstance(output, tuple) else output


def evaluate_ajcn_accuracy(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model = model.to(device)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = _model_logits(model(xb))
            pred = logits.argmax(dim=1)
            total += int(yb.numel())
            correct += int((pred == yb).sum().item())
    return 100.0 * correct / max(total, 1)


def predict_ajcn(model: nn.Module, X_raw: np.ndarray, device: torch.device, batch_size: int = 256) -> np.ndarray:
    X = raw_iq_to_ajcn_tensor(X_raw)
    ds = TensorDataset(torch.from_numpy(X), torch.zeros(X.shape[0], dtype=torch.long))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    model = model.to(device)
    model.eval()
    preds: List[np.ndarray] = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = _model_logits(model(xb))
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds, axis=0) if preds else np.empty((0,), dtype=np.int64)


def train_ajcn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int,
    lr: float,
    optimizer_name: str = "sgd",
    weight_decay: float = 0.0,
    verbose: bool = False,
) -> Tuple[nn.Module, float]:
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    if optimizer_name.lower() == "adam":
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    best_acc = -1.0
    best_state = copy.deepcopy(model.state_dict())
    for epoch in range(max(int(epochs), 0)):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = _model_logits(model(xb))
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
        eval_loader = val_loader if val_loader is not None else train_loader
        acc = evaluate_ajcn_accuracy(model, eval_loader, device)
        if acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"[AJCN] epoch {epoch + 1}/{epochs}, acc={acc:.2f}%")
    model.load_state_dict(best_state)
    return model, float(best_acc)


def count_ajcn_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _conv2d_flops(layer: nn.Conv2d, h: int, w: int) -> Tuple[int, int, int]:
    kh, kw = layer.kernel_size
    sh, sw = layer.stride
    ph, pw = layer.padding
    dh, dw = layer.dilation
    out_h = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    out_w = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    weight_ops = layer.in_channels * kh * kw * layer.out_channels // layer.groups
    flops = 2 * weight_ops * out_h * out_w
    if layer.bias is not None:
        flops += layer.out_channels * out_h * out_w
    return int(flops), int(out_h), int(out_w)


def estimate_ajcn_flops(model: nn.Module, input_shape: Tuple[int, int, int] = (3, 32, 32)) -> int:
    """Estimate AJCN FLOPs with forward hooks so residual/shortcut branches use actual tensor shapes."""
    device = next(model.parameters()).device if any(True for _ in model.parameters()) else torch.device("cpu")
    was_training = model.training
    model.eval()
    flops = 0
    hooks = []

    def conv_hook(module, inputs, output):
        nonlocal flops
        x = inputs[0]
        y = output
        if isinstance(y, tuple):
            y = y[0]
        batch, out_c, out_h, out_w = y.shape
        kh, kw = module.kernel_size
        ops_per_pos = module.in_channels * kh * kw * module.out_channels // module.groups
        flops += int(batch * out_h * out_w * (2 * ops_per_pos + (module.out_channels if module.bias is not None else 0)))

    def bn_hook(module, inputs, output):
        nonlocal flops
        y = output[0] if isinstance(output, tuple) else output
        flops += int(4 * y.numel())

    def linear_hook(module, inputs, output):
        nonlocal flops
        x = inputs[0]
        batch = x.shape[0] if x.ndim > 1 else 1
        flops += int(batch * (2 * module.in_features * module.out_features + module.out_features))

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.BatchNorm2d):
            hooks.append(m.register_forward_hook(bn_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(linear_hook))

    with torch.no_grad():
        dummy = torch.zeros((1,) + tuple(input_shape), dtype=torch.float32, device=device)
        model(dummy)

    for h in hooks:
        h.remove()
    if was_training:
        model.train()
    return int(flops)


def _default_prune_ratios() -> Dict[str, float]:
    return {name: 0.0 for name in AJCN_LAYER_NAMES}


def _ratios_to_dict(ratios: Sequence[float]) -> Dict[str, float]:
    d = _default_prune_ratios()
    for name, value in zip(AJCN_LAYER_NAMES, ratios):
        d[name] = float(value)
    return d


def _convert_actions_to_blocks(actions: Sequence[int]) -> List[str]:
    actions = list(actions)
    blocks = []
    groups = {
        "res_block1": [1, 2],
        "res_block2": [3, 4],
        "res_block3": [5, 6],
    }
    for block, idxs in groups.items():
        if any(i < len(actions) and int(actions[i]) == 1 for i in idxs):
            blocks.append(block)
    return blocks


def build_official_ajcn_model(
    num_classes: int,
    prune_ratios: Optional[Sequence[float] | Dict[str, float]] = None,
    convert_actions: Optional[Sequence[int]] = None,
) -> nn.Module:
    """Build an AJCN model using official BaseModel/PrunedModel and official DSC conversion code."""
    if prune_ratios is None:
        model = BaseModel(num_classes=num_classes)
    else:
        if isinstance(prune_ratios, dict):
            ratio_dict = {**_default_prune_ratios(), **{k: float(v) for k, v in prune_ratios.items()}}
        else:
            ratio_dict = _ratios_to_dict(prune_ratios)
        model = PrunedModel(num_classes=num_classes, prune_ratios=ratio_dict)

    if convert_actions is not None:
        for block in _convert_actions_to_blocks(convert_actions):
            # The official conversion function prints layer details; silence it inside batch experiments.
            with contextlib.redirect_stdout(io.StringIO()):
                model = convert_block_to_depthwise(model, block, prune_ratio=0.0)
    return model


def build_official_ajcn_model_from_info(info: Dict, num_classes: int) -> nn.Module:
    return build_official_ajcn_model(
        num_classes=num_classes,
        prune_ratios=info.get("prune_ratios"),
        convert_actions=info.get("convert_actions"),
    )


def _layer_l2_vector(model: nn.Module) -> np.ndarray:
    values: List[float] = []
    named = dict(model.named_modules())
    for name in AJCN_LAYER_NAMES:
        mod = named.get(name)
        if mod is not None and hasattr(mod, "weight"):
            values.append(float(torch.norm(mod.weight.detach()).cpu().item()))
        else:
            values.append(0.0)
    arr = np.asarray(values, dtype=np.float32)
    return (arr - arr.mean()) / (arr.std() + 1e-6)


def _state_for_layer(base_model: nn.Module, layer_idx: int, params_ratio: float, flops_ratio: float) -> np.ndarray:
    l2 = _layer_l2_vector(base_model)
    state = np.zeros(10, dtype=np.float32)
    state[:7] = l2
    state[7] = float(layer_idx) / max(len(AJCN_LAYER_NAMES) - 1, 1)
    state[8] = float(params_ratio)
    state[9] = float(flops_ratio)
    return state


@dataclass
class OfficialAJCNFitResult:
    model: nn.Module
    info: Dict
    training_time: float


class OfficialAJCNTrainer:
    """Experiment-level AJCN trainer using official AJCN modules plus DER-BLMoE data adapter."""

    def __init__(self, cfg, device: Optional[torch.device] = None, seed: int = 2025, logger=None):
        self.cfg = cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = int(seed)
        self.logger = logger
        self.batch_size = int(getattr(cfg, "AJCN_BATCH_SIZE", getattr(cfg, "DL_BATCH_SIZE", 64)))
        self.base_epochs = int(getattr(cfg, "AJCN_BASE_EPOCHS", 30))
        self.search_epochs = int(getattr(cfg, "AJCN_RL_CANDIDATE_EPOCHS", 1))
        self.final_epochs = int(getattr(cfg, "AJCN_FINAL_EPOCHS", 40))
        self.rl_episodes = int(getattr(cfg, "AJCN_RL_EPISODES", 50))
        self.lr_base = float(getattr(cfg, "AJCN_BASE_LR", 5e-4))
        self.lr_final = float(getattr(cfg, "AJCN_FINAL_LR", 1e-2))
        self.lr_rl = float(getattr(cfg, "AJCN_RL_LR", 5e-4))
        self.reward_param_weight = float(getattr(cfg, "AJCN_REWARD_PARAM_WEIGHT", 0.5))
        self.reward_flops_weight = float(getattr(cfg, "AJCN_REWARD_FLOPS_WEIGHT", 0.5))
        self.image_size = int(getattr(cfg, "AJCN_IMAGE_SIZE", 32))
        self.prune_options = list(getattr(cfg, "AJCN_PRUNE_RATIOS", AJCN_PRUNE_OPTIONS))
        if len(self.prune_options) != 10:
            self.prune_options = AJCN_PRUNE_OPTIONS
        num_threads = int(getattr(cfg, "AJCN_NUM_THREADS", 1))
        if num_threads > 0:
            torch.set_num_threads(num_threads)

    def _log(self, msg: str) -> None:
        if self.logger is not None:
            self.logger.info(msg)
        else:
            print(msg)

    def fit(self, X_train_raw, y_train, X_val_raw, y_val) -> OfficialAJCNFitResult:
        set_ajcn_seed(self.seed)
        t_start = time.time()
        train_loader = make_ajcn_loader(X_train_raw, y_train, self.batch_size, True, self.image_size)
        val_loader = make_ajcn_loader(X_val_raw, y_val, self.batch_size, False, self.image_size)

        self._log("  [AJCN] Training official BaseModel before RL compression.")
        base_model = BaseModel(num_classes=int(self.cfg.NUM_CLASSES))
        base_model, base_acc = train_ajcn_model(
            base_model, train_loader, val_loader, self.device,
            epochs=self.base_epochs, lr=self.lr_base, optimizer_name="sgd", verbose=False,
        )
        orig_params = count_ajcn_params(base_model)
        orig_flops = estimate_ajcn_flops(base_model)
        self._log(f"    -> Base Acc={base_acc:.2f}% | Params={orig_params/1e6:.4f}M | FLOPs={orig_flops/1e6:.4f}M")

        agent = PPOAgent(
            state_dim=10,
            action_dims=[len(self.prune_options), 2],
            device=self.device,
            lr=self.lr_rl,
            gamma=float(getattr(self.cfg, "AJCN_RL_GAMMA", 0.99)),
            clip_ratio=float(getattr(self.cfg, "AJCN_RL_CLIP_RATIO", 0.2)),
            value_coef=float(getattr(self.cfg, "AJCN_RL_VALUE_COEF", 0.5)),
            entropy_coef=float(getattr(self.cfg, "AJCN_RL_ENTROPY_COEF", 0.01)),
        )

        best = {
            "reward": -1e9,
            "acc": 0.0,
            "prune_ratios": [0.0] * len(AJCN_LAYER_NAMES),
            "convert_actions": [0] * len(AJCN_LAYER_NAMES),
            "params": orig_params,
            "flops": orig_flops,
        }

        self._log(f"  [AJCN] PPO compression search episodes={self.rl_episodes}.")
        for ep in range(self.rl_episodes):
            ratios: List[float] = []
            converts: List[int] = []
            states = []
            actions = []
            log_probs = []
            params_ratio = 1.0
            flops_ratio = 1.0

            for li in range(len(AJCN_LAYER_NAMES)):
                state = _state_for_layer(base_model, li, params_ratio, flops_ratio)
                action, log_prob = agent.select_action(state)
                prune_idx, convert_action = int(action[0]), int(action[1])
                ratios.append(float(self.prune_options[prune_idx]))
                converts.append(convert_action)
                states.append(state)
                actions.append([prune_idx, convert_action])
                log_probs.append(log_prob)

            candidate = build_official_ajcn_model(self.cfg.NUM_CLASSES, ratios, converts)
            candidate, cand_acc = train_ajcn_model(
                candidate, train_loader, val_loader, self.device,
                epochs=self.search_epochs, lr=self.lr_final, optimizer_name="sgd", verbose=False,
            )
            cand_params = count_ajcn_params(candidate)
            cand_flops = estimate_ajcn_flops(candidate)
            acc_ratio = cand_acc / max(base_acc, 1e-6)
            params_ratio = cand_params / max(orig_params, 1)
            flops_ratio = cand_flops / max(orig_flops, 1)
            reward = acc_ratio - self.reward_param_weight * params_ratio - self.reward_flops_weight * flops_ratio

            # Store the same terminal reward for each layer decision, matching layerwise PPO decision process.
            for i, state in enumerate(states):
                next_state = states[i + 1] if i + 1 < len(states) else _state_for_layer(base_model, len(AJCN_LAYER_NAMES)-1, params_ratio, flops_ratio)
                done = (i == len(states) - 1)
                agent.store_transition(state, actions[i], log_probs[i], reward, next_state, done)
            agent.update(
                batch_size=int(getattr(self.cfg, "AJCN_RL_BATCH_SIZE", 64)),
                epochs=int(getattr(self.cfg, "AJCN_RL_UPDATE_EPOCHS", 10)),
            )

            if reward > best["reward"]:
                best.update({
                    "reward": float(reward),
                    "acc": float(cand_acc),
                    "prune_ratios": [float(x) for x in ratios],
                    "convert_actions": [int(x) for x in converts],
                    "params": int(cand_params),
                    "flops": int(cand_flops),
                })
            if (ep + 1) % max(int(getattr(self.cfg, "AJCN_LOG_INTERVAL", 10)), 1) == 0 or ep == 0:
                self._log(
                    f"    -> Episode {ep+1:03d}/{self.rl_episodes}: "
                    f"reward={reward:.4f}, acc={cand_acc:.2f}%, params={cand_params/1e6:.4f}M, flops={cand_flops/1e6:.4f}M"
                )

        self._log("  [AJCN] Final fine-tuning of best official AJCN compressed model.")
        final_model = build_official_ajcn_model(self.cfg.NUM_CLASSES, best["prune_ratios"], best["convert_actions"])
        final_model, final_acc = train_ajcn_model(
            final_model, train_loader, val_loader, self.device,
            epochs=self.final_epochs, lr=self.lr_final, optimizer_name="sgd", verbose=False,
        )
        final_params = count_ajcn_params(final_model)
        final_flops = estimate_ajcn_flops(final_model)
        info = {
            "method_name": "AJCN",
            "strict_integration": True,
            "source_code_preserved_at": "third_party/AJCN-main",
            "namespaced_source_used_at": "third_party/ajcn_src",
            "note": "Official AJCN BaseModel/PrunedModel/DSC conversion/PPOAgent are used. Only raw-IQ-to-32x32x3 data adapter and experiment glue are project-specific.",
            "input_adapter": "raw_iq_to_ajcn_tensor: channels=[I,Q,magnitude], resampled to 32x32",
            "num_classes": int(self.cfg.NUM_CLASSES),
            "image_size": self.image_size,
            "base_accuracy_val_percent": float(base_acc),
            "final_accuracy_val_percent": float(final_acc),
            "original_params": int(orig_params),
            "original_flops": int(orig_flops),
            "final_params": int(final_params),
            "final_flops": int(final_flops),
            "prune_ratios": [float(x) for x in best["prune_ratios"]],
            "convert_actions": [int(x) for x in best["convert_actions"]],
            "layer_names": list(AJCN_LAYER_NAMES),
            "prune_ratio_dict": _ratios_to_dict(best["prune_ratios"]),
            "converted_blocks": _convert_actions_to_blocks(best["convert_actions"]),
            "rl_episodes": self.rl_episodes,
            "base_epochs": self.base_epochs,
            "candidate_epochs": self.search_epochs,
            "final_epochs": self.final_epochs,
        }
        elapsed = time.time() - t_start
        self._log(f"    -> Final AJCN ValAcc={final_acc:.2f}% | Params={final_params/1e6:.4f}M | FLOPs={final_flops/1e6:.4f}M")
        return OfficialAJCNFitResult(model=final_model, info=info, training_time=elapsed)
