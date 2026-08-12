import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import os

class DataProcessor:
    def __init__(self, batch_size, data_dir='./data/ADS'):
        self.batch_size = batch_size

        # self.data_dir = 'E:/work8/FedPrototypes/data/ADS'
        self.data_dir = './data/ADS'
        # 确保数据目录存在
        os.makedirs(self.data_dir, exist_ok=True)

    def load_data(self):
        try:
            # 检查文件是否存在
            train_data_path = os.path.join(self.data_dir, 'ADS_train_data.npy')
            train_label_path = os.path.join(self.data_dir, 'ADS_train_label.npy')
            test_data_path = os.path.join(self.data_dir, 'ADS_test_data.npy')
            test_label_path = os.path.join(self.data_dir, 'ADS_test_label.npy')
            
            for path in [train_data_path, train_label_path, test_data_path, test_label_path]:
                if not os.path.exists(path):
                    raise FileNotFoundError(f"数据文件不存在: {path}")
            
            # 加载训练和测试数据
            print(f"加载训练数据: {train_data_path}")
            train_data = np.load(train_data_path)
            train_labels = np.load(train_label_path)
            test_data = np.load(test_data_path)
            test_labels = np.load(test_label_path)
            
            print(f"训练数据形状: {train_data.shape}, 标签形状: {train_labels.shape}")
            print(f"测试数据形状: {test_data.shape}, 标签形状: {test_labels.shape}")

            # 将数据转换为 PyTorch 张量
            train_data = torch.tensor(train_data, dtype=torch.float32)
            train_labels = torch.tensor(train_labels, dtype=torch.long)
            test_data = torch.tensor(test_data, dtype=torch.float32)
            test_labels = torch.tensor(test_labels, dtype=torch.long)

            # 添加噪声并增加样本数量
            noise_factor = 0.1  # 噪声因子
            num_augmented_samples = 10  # 每个样本生成 10 个增强样本
            augmented_data = []

            for i in range(len(train_data)):
                original_sample = train_data[i].unsqueeze(0)  # 变为 1x3x32x32
                for _ in range(num_augmented_samples):
                    noisy_sample = original_sample + noise_factor * torch.randn_like(original_sample)  # 添加噪声
                    augmented_data.append(noisy_sample)  # 保留四维

            # 将增强后的样本与原始样本合并
            augmented_data = torch.cat(augmented_data, dim=0)  # 合并所有增强样本
            train_data_combined = torch.cat([train_data, augmented_data], dim=0)  # 合并原始数据与增强数据

            # 创建标签：将标签复制到增强样本中
            train_labels_combined = torch.cat(
                [train_labels] + [train_labels[i].unsqueeze(0).expand(num_augmented_samples) for i in
                                range(len(train_labels))], dim=0)

            # 划分验证集和测试集：每类按 5:5 比例划分
            test_data, val_data, test_labels, val_labels = self.split_validation_test(test_data, test_labels)

            # 创建数据加载器
            train_dataset = TensorDataset(train_data_combined, train_labels_combined)
            val_dataset = TensorDataset(val_data, val_labels)
            test_dataset = TensorDataset(test_data, test_labels)

            train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

            return train_loader, val_loader, test_loader
            
        except Exception as e:
            print(f"加载数据时出错: {e}")
            raise

    def split_validation_test(self, test_data, test_labels):
        """
        按照每类 5:5 的比例划分验证集和测试集。
        :param test_data: 测试集数据
        :param test_labels: 测试集标签
        :return: 划分后的测试集和验证集
        """
        unique_classes = torch.unique(test_labels)  # 获取唯一的类别标签
        val_indices = []
        test_indices = []

        for cls in unique_classes:
            cls_indices = torch.where(test_labels == cls)[0]  # 当前类别的所有索引
            cls_val_indices, cls_test_indices = train_test_split(
                cls_indices.numpy(), test_size=0.5, random_state=42, shuffle=True
            )
            val_indices.extend(cls_val_indices.tolist())
            test_indices.extend(cls_test_indices.tolist())

        # 构造验证集和测试集
        val_data = test_data[val_indices]
        val_labels = test_labels[val_indices]
        test_data = test_data[test_indices]
        test_labels = test_labels[test_indices]

        return test_data, val_data, test_labels, val_labels

if __name__ == "__main__":
    processor = DataProcessor(batch_size=128)
    train_loader, val_loader, test_loader = processor.load_data()

    # 验证数据加载器
    print("训练集批量数:", len(train_loader))
    print("验证集批量数:", len(val_loader))
    print("测试集批量数:", len(test_loader))

    # 示例：打印一个批次的形状
    for data, labels in train_loader:
        print("训练数据批次形状:", data.shape)
        print("训练标签批次形状:", labels.shape)
        break
