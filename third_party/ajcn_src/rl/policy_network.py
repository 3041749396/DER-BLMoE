import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    """
    策略网络，输出剪枝和转换动作的概率分布
    """
    def __init__(self, state_dim, action_dims, hidden_dim=128):
        """
        初始化策略网络
        
        参数:
            state_dim: 状态空间维度
            action_dims: 动作空间维度列表 [剪枝动作数, 转换动作数]
            hidden_dim: 隐藏层维度
        """
        super(PolicyNetwork, self).__init__()
        
        # 提取动作空间维度
        self.prune_action_dim = action_dims[0]  # 剪枝动作数
        self.convert_action_dim = action_dims[1]  # 转换动作数
        
        # 共享特征提取层
        # self.fc1 = nn.Linear(state_dim, hidden_dim)
        # self.fc2 = nn.Linear(hidden_dim, hidden_dim)

          # Actor网络（策略网络）
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),              # 更宽的隐藏层
            nn.ReLU(),                     # 激活函数
            nn.Dropout(0.5),                # Dropout层0.5
            # nn.Linear(64, action_space_per_layer),  # 输出每层剪枝比率的动作空间
        )
        
        # 剪枝动作头
        self.prune_head = nn.Linear(hidden_dim, self.prune_action_dim)
        
        # 转换动作头
        self.convert_head = nn.Linear(hidden_dim, self.convert_action_dim)
        self._initialize_weights()  # 初始化权重
    def forward(self, state):
        """
        前向传播
        
        参数:
            state: 输入状态
        
        返回:
            prune_probs: 剪枝动作概率分布
            convert_probs: 转换动作概率分布
        """
        # 特征提取
        x = self.fc1(state)
        # x = F.relu(self.fc2(x))
        
        # 剪枝动作概率
        prune_logits = self.prune_head(x)
        prune_probs = F.softmax(prune_logits, dim=-1)
        
        # 转换动作概率
        convert_logits = self.convert_head(x)
        convert_probs = F.softmax(convert_logits, dim=-1)
        
        return prune_probs, convert_probs
    
    def _initialize_weights(self):
    # 使用He初始化方法来初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming初始化（ReLU）
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class ValueNetwork(nn.Module):
    """
    价值网络，估计状态价值
    """
    def __init__(self, state_dim, hidden_dim=128):
        """
        初始化价值网络
        
        参数:
            state_dim: 状态空间维度
            hidden_dim: 隐藏层维度
        """
        super(ValueNetwork, self).__init__()
        
              # Actor网络（策略网络）
        self.fc1 = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),              # 更宽的隐藏层
            nn.ReLU(),                     # 激活函数
            nn.Dropout(0.5),                # Dropout层0.5
            # nn.Linear(64, action_space_per_layer),  # 输出每层剪枝比率的动作空间
            # nn.Linear(hidden_dim, hidden_dim),  # 更宽的隐藏层
            # nn.ReLU(),  # 激活函数
            # nn.Dropout(0.5),
        )

        self.fc_value=nn.Linear(128,1)# 价值估计
        self._initialize_weights()  # 初始化权重
    
    def forward(self, state):
        """
        前向传播
        
        参数:
            state: 输入状态
        
        返回:
            value: 状态价值估计
        """
        x = self.fc1(state)
     
        value =  self.fc_value(x)
        
        return value 

    def _initialize_weights(self):
        # 使用He初始化方法来初始化网络权重
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')  # Kaiming初始化（ReLU）
                if m.bias is not None:
                    nn.init.zeros_(m.bias)