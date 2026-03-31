# models/dl_trainer.py
import time
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 引入学习率调度器
from torch.optim.lr_scheduler import StepLR 

from models.cnn_1d import MobileNetV2_1D
from models.resnet_1d import ResNet18_1D, ResNet34_1D
from configs.config_moe_bls import get_moe_config


class PyTorchTrainer:
    """
    通用 1D 深度网络训练器：
    - 支持 ResNet18 / ResNet34 / MobileNetV2_1D
    - 输入假定为实部+虚部拼接的一维向量 (N, 2*L)，内部自动 reshape 为 (N, 2, L)
    - 始终跑满 epochs，使用验证集选取最佳 epoch 的参数，不做早停。

    配置策略：
    - 如果 batch_size / epochs / lr 未显式指定，则从 MoeBLSConfig 中读取：
        DL_BATCH_SIZE, DL_EPOCHS, DL_LR
    - 如果在实验脚本里手动传入这些参数，则以手动参数为准。
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        input_len: int = 128,
        num_classes: int = 10,
        batch_size: Optional[int] = None,
        epochs: Optional[int] = None,
        lr: Optional[float] = None,
        device=None,
        patience: int = 5,
        min_delta: float = 1e-4,
    ):
        """
        patience: 为兼容旧接口保留，不触发提前停止。
        min_delta: 判定“metric 是否真正提升”的最小增量，用于更新 best checkpoint。
        """
        # 先读全局配置（如果需要）
        cfg = get_moe_config()

        # 如果没传，就用 config 里的 DL_*；传了就用传进来的
        if batch_size is None:
            batch_size = getattr(cfg, "DL_BATCH_SIZE", 64)
        if epochs is None:
            epochs = getattr(cfg, "DL_EPOCHS", 30)
        if lr is None:
            lr = getattr(cfg, "DL_LR", 1e-3)

        self.model_name = model_name
        # 原始数据为 2*L (Real + Imag)，实际序列长度是一半
        self.input_len = input_len // 2
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.min_delta = min_delta

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.model: Optional[nn.Module] = None

    # --------------------------------------------------
    # 内部工具
    # --------------------------------------------------
    def _build_model(self) -> nn.Module:
        """根据名称构建对应的 1D 模型。"""
        if self.model_name == "mobilenetv2":
            return MobileNetV2_1D(num_classes=self.num_classes, input_len=self.input_len)
        elif self.model_name == "resnet18":
            return ResNet18_1D(num_classes=self.num_classes)
        elif self.model_name == "resnet34":
            return ResNet34_1D(num_classes=self.num_classes)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")

    def _reshape_input(self, X: np.ndarray) -> torch.Tensor:
        """
        X shape: (N, 2*L) -> (N, 2, L) 供 1D Conv 使用。
        假定 X 为 [I_part, Q_part] 拼接。
        """
        X = X.astype(np.float32)
        N = X.shape[0]
        L = X.shape[1] // 2

        X_I = X[:, :L]
        X_Q = X[:, L:]
        X_reshaped = np.stack([X_I, X_Q], axis=1)  # (N, 2, L)
        return torch.tensor(X_reshaped, dtype=torch.float32)

    def _make_loader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        X_tensor = self._reshape_input(X)
        y_tensor = torch.tensor(y, dtype=torch.long)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)
        return loader

    # --------------------------------------------------
    # 训练：无早停，始终跑满 epochs，用验证集选最佳
    # --------------------------------------------------
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ):
        """
        训练模型：
        - 如提供验证集 (X_val, y_val)，则基于验证准确率选最佳 epoch 的模型参数；
        - 始终跑满 self.epochs 轮，不做提前停止。
        """
        self.model = self._build_model().to(self.device)
        criterion = nn.CrossEntropyLoss()

        # 优化器：
        # - MobileNetV2：Adam + weight_decay
        # - ResNet18/34：Adam
        if self.model_name == "mobilenetv2":
            optimizer = optim.Adam(
                self.model.parameters(), lr=self.lr, weight_decay=1e-4
            )
        else:
            optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

        # [新增] 学习率调度器：每 50 个 epoch，lr = lr * 0.1
        scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

        train_loader = self._make_loader(X_train, y_train, shuffle=True)
        val_loader = None
        use_val = (X_val is not None) and (y_val is not None)

        if use_val:
            val_loader = self._make_loader(X_val, y_val, shuffle=False)
            print(
                "[{}] Training with validation (FULL {} epochs) on {} ...".format(
                    self.model_name.upper(), self.epochs, self.device
                )
            )
        else:
            print(
                "[{}] Training (no validation, FULL {} epochs) on {} ...".format(
                    self.model_name.upper(), self.epochs, self.device
                )
            )

        best_metric = -np.inf
        best_state = None
        best_epoch = -1
        metric_name = "ValAcc" if use_val else "TrainAcc"

        start_t = time.time()
        for epoch in range(self.epochs):
            # ---------------------------
            # 1) Train
            # ---------------------------
            self.model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += batch_y.size(0)
                correct += predicted.eq(batch_y).sum().item()

            train_loss = total_loss / max(len(train_loader), 1)
            train_acc = 100.0 * correct / max(total, 1)

            # ---------------------------
            # 2) Val（如果有）
            # ---------------------------
            if use_val and val_loader is not None:
                self.model.eval()
                val_loss_sum = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = self.model(batch_X)
                        loss = criterion(outputs, batch_y)

                        val_loss_sum += loss.item()
                        _, predicted = outputs.max(1)
                        val_total += batch_y.size(0)
                        val_correct += predicted.eq(batch_y).sum().item()

                val_loss = val_loss_sum / max(len(val_loader), 1)
                val_acc = 100.0 * val_correct / max(val_total, 1)

                metric = val_acc  # 选最佳模型依据：验证准确率
            else:
                val_loss = train_loss
                val_acc = train_acc
                metric = train_acc

            # 日志 (可选：打印当前学习率)
            current_lr = optimizer.param_groups[0]['lr']
            if use_val:
                print(
                    "  Epoch {}/{} [LR={:.1e}] | TrainLoss: {:.4f} | TrainAcc: {:.2f}% | "
                    "ValLoss: {:.4f} | ValAcc: {:.2f}%".format(
                        epoch + 1, self.epochs, current_lr, train_loss, train_acc, val_loss, val_acc
                    )
                )
            else:
                print(
                    "  Epoch {}/{} [LR={:.1e}] | TrainLoss: {:.4f} | TrainAcc: {:.2f}%".format(
                        epoch + 1, self.epochs, current_lr, train_loss, train_acc
                    )
                )

            # ---------------------------
            # 3) 仅记录最佳 checkpoint，不做 early stop
            # ---------------------------
            if metric > best_metric + self.min_delta:
                best_metric = metric
                best_state = deepcopy(self.model.state_dict())
                best_epoch = epoch

            # [新增] 更新学习率
            scheduler.step()

        # 训练结束后，恢复到 best checkpoint（如果有）
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print(
                "[{}] Loaded best checkpoint (epoch {}, best {}={:.2f}%)".format(
                    self.model_name.upper(), best_epoch + 1, metric_name, best_metric
                )
            )

        print("  Training finished in {:.2f}s".format(time.time() - start_t))
        return self

    # --------------------------------------------------
    # 推理
    # --------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        对输入 X 做批量预测。
        """
        if self.model is None:
            raise RuntimeError("Model has not been trained. Call fit() first.")

        self.model.eval()
        X_tensor = self._reshape_input(X).to(self.device)

        preds = []
        with torch.no_grad():
            num_batches = int(np.ceil(X.shape[0] / float(self.batch_size)))
            for i in range(num_batches):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, X.shape[0])
                batch_X = X_tensor[start:end]

                outputs = self.model(batch_X)
                _, predicted = torch.max(outputs.data, 1)
                preds.append(predicted.cpu().numpy())

        return np.concatenate(preds)