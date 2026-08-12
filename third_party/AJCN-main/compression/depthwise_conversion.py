import torch
import torch.nn as nn
import copy
import time
from models.depthwise_model import DepthwiseSeparableConv, DepthwiseSeparableResidualBlock
import types

def convert_to_depthwise_separable(model, block_name=None):
    """
    将模型中的标准卷积残差块转换为深度可分离卷积残差块
    
    参数:
        model: 原始模型
        block_name: 要转换的残差块名称，如果为None则转换res_block1
    
    返回:
        converted_model: 转换后的模型
    """
    # 深拷贝模型，避免修改原始模型
    converted_model = copy.deepcopy(model)
    
    # 获取所有残差块
    residual_blocks = {
        'res_block1': converted_model.res_block1,
        'res_block2': converted_model.res_block2,
        'res_block3': converted_model.res_block3
    }
    
    # 确定要转换的残差块
    blocks_to_convert = []
    if block_name is None:
        # 默认只转换res_block1
        blocks_to_convert = ['res_block1']
    elif block_name == 'all':
        blocks_to_convert = list(residual_blocks.keys())
    else:
        blocks_to_convert = [block_name]
    
    # 转换指定的残差块
    for block_name in blocks_to_convert:
        if block_name in residual_blocks:
            block = residual_blocks[block_name]
            
            try:
                # 获取输入和输出通道数
                in_channels = block.conv1.in_channels
                out_channels = block.conv2.out_channels
                stride = block.conv1.stride[0]
                
                print(f"转换 {block_name}: in_channels={in_channels}, out_channels={out_channels}, stride={stride}")
                
                # 创建深度可分离卷积残差块
                new_block = DepthwiseSeparableResidualBlock(in_channels, out_channels, stride)
                
                # 替换原始残差块
                if block_name == 'res_block1':
                    converted_model.res_block1 = new_block
                elif block_name == 'res_block2':
                    converted_model.res_block2 = new_block
                elif block_name == 'res_block3':
                    converted_model.res_block3 = new_block
            except Exception as e:
                print(f"转换 {block_name} 时出错: {e}")
    
    return converted_model

def convert_layer_weights(conv_layer, depthwise_separable_layer):
    """
    将标准卷积层的权重转换为深度可分离卷积层的权重
    
    参数:
        conv_layer: 标准卷积层
        depthwise_separable_layer: 深度可分离卷积层
    
    返回:
        depthwise_separable_layer: 转换后的深度可分离卷积层
    """
    try:
        # 获取标准卷积的权重
        weight = conv_layer.weight.data
        
        # 获取通道数
        in_channels = weight.shape[1]
        out_channels = weight.shape[0]
        kernel_size = weight.shape[2]
        
        # 初始化深度卷积权重
        depthwise_weight = torch.zeros(in_channels, 1, kernel_size, kernel_size)
        
        # 计算深度卷积权重（每个输入通道的平均卷积核）
        for i in range(in_channels):
            depthwise_weight[i, 0] = torch.mean(weight[:, i], dim=0)
        
        # 初始化逐点卷积权重
        pointwise_weight = torch.zeros(out_channels, in_channels, 1, 1)
        
        # 计算逐点卷积权重（每个输出通道的通道权重）
        for i in range(out_channels):
            for j in range(in_channels):
                pointwise_weight[i, j, 0, 0] = torch.mean(weight[i, j])
        
        # 设置深度可分离卷积层的权重
        depthwise_separable_layer.depthwise.weight.data = depthwise_weight
        depthwise_separable_layer.pointwise.weight.data = pointwise_weight
        
        return depthwise_separable_layer
    except Exception as e:
        print(f"转换层权重时出错: {e}")
        return depthwise_separable_layer

def convert_block_to_depthwise(model, block_name, prune_ratio=0.0):
    """
    将残差块转换为深度可分离卷积，并应用剪枝
    
    参数:
        model: 当前模型
        block_name: 要转换的残差块名称
        prune_ratio: 应用于深度可分离卷积的剪枝率
    
    返回:
        converted_model: 转换后的模型
    """
    converted_model = copy.deepcopy(model)
    
    try:
        # 获取块的实际通道数和步长
        block = getattr(converted_model, block_name)
        
        # 获取前一个块的输出通道数
        if block_name == 'res_block1':
            prev_out_channels = converted_model.conv1.out_channels
        elif block_name == 'res_block2':
            if hasattr(converted_model.res_block1, 'conv2'):
                prev_out_channels = converted_model.res_block1.conv2.out_channels
            else:
                prev_out_channels = converted_model.res_block1.pointwise.out_channels
        elif block_name == 'res_block3':
            if hasattr(converted_model.res_block2, 'conv2'):
                prev_out_channels = converted_model.res_block2.conv2.out_channels
            else:
                prev_out_channels = converted_model.res_block2.pointwise.out_channels
        
        # 使用前一个块的实际输出通道数作为输入通道数
        in_channels = prev_out_channels
        
        # 获取当前块的输出通道数
        if hasattr(block, 'conv2'):
            original_out_channels = block.conv2.out_channels
        else:
            original_out_channels = block.pointwise.out_channels
            
        stride = block.conv1.stride[0] if hasattr(block, 'conv1') else block.depthwise.stride[0]
        
        # 应用输出通道剪枝
        if prune_ratio > 0:
            # 如果是最后一个块，确保输出通道数与全连接层匹配
            if block_name == 'res_block3':
                fc_in_features = converted_model.fc.in_features
                pruned_out_channels = fc_in_features  # 保持与全连接层匹配
                print(f"注意：最后一个块的输出通道数保持与全连接层匹配 ({fc_in_features})")
            else:
                # 计算剪枝后的输出通道数
                pruned_out_channels = max(1, int(original_out_channels * (1 - prune_ratio)))
            
            print(f"转换 {block_name}: 输入通道数={in_channels}, "
                  f"原始输出通道数={original_out_channels}, 剪枝后输出通道数={pruned_out_channels}, stride={stride}")
            
            out_channels = pruned_out_channels
        else:
            out_channels = original_out_channels
            print(f"转换 {block_name}: 输入通道数={in_channels}, 输出通道数={out_channels}, stride={stride}")
        
        # 创建深度可分离卷积模块
        # 注意：depthwise卷积的输入和输出通道数必须相同，等于输入通道数
        # pointwise卷积的输出通道数是剪枝后的输出通道数
        depthwise_module = nn.Module()
        
        # 深度卷积（每个通道单独卷积）
        depthwise_module.depthwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,  # 深度卷积的输出通道数等于输入通道数
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=in_channels,  # 分组卷积，每组一个通道
            bias=False
        )
        
        # 逐点卷积（1x1卷积，混合通道信息）
        depthwise_module.pointwise = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,  # 使用剪枝后的输出通道数
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        )
        
        # 批归一化和激活函数
        depthwise_module.bn1 = nn.BatchNorm2d(in_channels)  # 对应depthwise的输出
        depthwise_module.bn2 = nn.BatchNorm2d(out_channels)  # 对应pointwise的输出
        depthwise_module.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，添加shortcut连接
        depthwise_module.shortcut = None
        if stride != 1 or in_channels != out_channels:
            depthwise_module.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        # 定义前向传播函数
        def forward(self, x):
            identity = x
            
            # 深度卷积
            out = self.depthwise(x)
            out = self.bn1(out)
            out = self.relu(out)
            
            # 逐点卷积
            out = self.pointwise(out)
            out = self.bn2(out)
            
            # shortcut连接
            if self.shortcut is not None:
                identity = self.shortcut(x)
            
            out += identity
            out = self.relu(out)
            
            return out
        
        # 将前向传播函数绑定到模块
        depthwise_module.forward = types.MethodType(forward, depthwise_module)
        
        # 替换模型中的残差块
        setattr(converted_model, block_name, depthwise_module)
        
        # 如果是最后一个块，确保全连接层的输入特征数匹配
        if block_name == 'res_block3':
            if converted_model.fc.in_features != out_channels:
                print(f"更新全连接层：原始in_features={converted_model.fc.in_features}, 新in_features={out_channels}")
                # 创建新的全连接层
                new_fc = nn.Linear(
                    in_features=out_channels,
                    out_features=converted_model.fc.out_features,
                    bias=True if converted_model.fc.bias is not None else False
                )
                # 初始化权重
                nn.init.kaiming_normal_(new_fc.weight)
                if new_fc.bias is not None:
                    nn.init.zeros_(new_fc.bias)
                # 替换全连接层
                converted_model.fc = new_fc
        
        # 更新下一个块的输入通道数
        if block_name == 'res_block1':
            next_block = 'res_block2'
        elif block_name == 'res_block2':
            next_block = 'res_block3'
        else:
            next_block = None
            
        if next_block and hasattr(converted_model, next_block):
            next_block_obj = getattr(converted_model, next_block)
            
            # 更新下一个块的输入通道数
            if hasattr(next_block_obj, 'conv1'):
                # 标准残差块
                new_conv1 = nn.Conv2d(
                    in_channels=out_channels,  # 使用当前块的输出通道数
                    out_channels=next_block_obj.conv1.out_channels,
                    kernel_size=next_block_obj.conv1.kernel_size,
                    stride=next_block_obj.conv1.stride,
                    padding=next_block_obj.conv1.padding,
                    bias=False
                )
                nn.init.kaiming_normal_(new_conv1.weight)
                next_block_obj.conv1 = new_conv1
                
                # 更新shortcut
                if hasattr(next_block_obj, 'shortcut') and next_block_obj.shortcut is not None:
                    new_shortcut_conv = nn.Conv2d(
                        in_channels=out_channels,  # 使用当前块的输出通道数
                        out_channels=next_block_obj.shortcut[0].out_channels,
                        kernel_size=1,
                        stride=next_block_obj.shortcut[0].stride,
                        bias=False
                    )
                    nn.init.kaiming_normal_(new_shortcut_conv.weight)
                    next_block_obj.shortcut[0] = new_shortcut_conv
                    
            elif hasattr(next_block_obj, 'depthwise'):
                # 深度可分离卷积块
                # 更新depthwise层
                new_depthwise = nn.Conv2d(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=next_block_obj.depthwise.stride,
                    padding=1,
                    groups=out_channels,
                    bias=False
                )
                nn.init.kaiming_normal_(new_depthwise.weight)
                next_block_obj.depthwise = new_depthwise
                
                # 更新bn1
                next_block_obj.bn1 = nn.BatchNorm2d(out_channels)
                
                # 更新shortcut
                if hasattr(next_block_obj, 'shortcut') and next_block_obj.shortcut is not None:
                    new_shortcut_conv = nn.Conv2d(
                        in_channels=out_channels,
                        out_channels=next_block_obj.pointwise.out_channels,
                        kernel_size=1,
                        stride=next_block_obj.shortcut[0].stride,
                        bias=False
                    )
                    nn.init.kaiming_normal_(new_shortcut_conv.weight)
                    next_block_obj.shortcut[0] = new_shortcut_conv
                    next_block_obj.shortcut[1] = nn.BatchNorm2d(next_block_obj.pointwise.out_channels)
        
    except Exception as e:
        print(f"转换残差块时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return model
    
    return converted_model

class DepthwiseSeparableConv(nn.Module):
    """
    深度可分离卷积模块，替代标准残差块
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        
        # 深度卷积（每个通道单独卷积）- 输入和输出通道数必须相同
        self.depthwise = nn.Conv2d(
            in_channels, 
            in_channels,  # 深度卷积的输出通道数必须等于输入通道数
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            groups=in_channels,
            bias=False
        )
        
        # 逐点卷积（1x1卷积，混合通道信息）
        self.pointwise = nn.Conv2d(
            in_channels, 
            out_channels, 
            kernel_size=1, 
            stride=1, 
            padding=0, 
            bias=False
        )
        
        # 批归一化和激活函数
        self.bn1 = nn.BatchNorm2d(in_channels)  # 对应depthwise的输出
        self.bn2 = nn.BatchNorm2d(out_channels)  # 对应pointwise的输出
        self.relu = nn.ReLU(inplace=True)
        
        # 如果输入输出通道数不同，添加shortcut连接
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        # 深度卷积
        out = self.depthwise(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # 逐点卷积
        out = self.pointwise(out)
        out = self.bn2(out)
        
        # shortcut连接
        if self.shortcut is not None:
            identity = self.shortcut(x)
        
        out += identity
        out = self.relu(out)
        
        return out

def estimate_model_complexity(model, input_size=(1, 3, 224, 224)):
    """
    估计模型的复杂度（FLOPs和参数量）
    
    参数:
        model: 要评估的模型
        input_size: 输入张量的形状
    
    返回:
        flops: 浮点运算次数
        params: 参数数量
    """
    # 计算参数量
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 简单估计FLOPs（实际应使用专门的工具如thop）
    # 这里使用一个简化的估计方法
    flops = params * 2  # 简化估计
    
    return flops, params

def get_inference_time(model, input_size=(1, 3, 224, 224), device='cuda', num_iterations=100):
    """
    测量模型的推理时间
    
    参数:
        model: 要评估的模型
        input_size: 输入张量的形状
        device: 计算设备
        num_iterations: 迭代次数
    
    返回:
        inference_time: 平均推理时间（毫秒）
    """
    model = model.to(device)
    model.eval()
    
    # 创建随机输入
    input_tensor = torch.randn(input_size).to(device)
    
    # 预热
    with torch.no_grad():
        for _ in range(10):
            model(input_tensor)
    
    # 测量推理时间
    start_time = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            model(input_tensor)
    end_time = time.time()
    
    # 计算平均推理时间（毫秒）
    inference_time = (end_time - start_time) * 1000 / num_iterations
    
    return inference_time

def get_layer_info(block):
    """
    获取块的层信息，支持标准卷积块和深度可分离卷积块
    
    参数:
        block: 残差块（标准或深度可分离）
    
    返回:
        first_layer, second_layer: 块中的两个主要层
    """
    if hasattr(block, 'conv1') and hasattr(block, 'conv2'):
        # 标准卷积块
        return block.conv1, block.conv2
    elif hasattr(block, 'depthwise') and hasattr(block, 'pointwise'):
        # 深度可分离卷积块
        return block.depthwise, block.pointwise
    else:
        raise ValueError("Unsupported block type")

def get_state(self):
    """
    获取当前环境状态
    
    返回:
        state: 包含模型各层信息的状态向量
    """
    try:
        # 获取各个块的层信息
        block1_layers = get_layer_info(self.current_model.res_block1)
        block2_layers = get_layer_info(self.current_model.res_block2)
        block3_layers = get_layer_info(self.current_model.res_block3)
        
        # 构建状态向量
        state = [
            block1_layers[0].in_channels, block1_layers[1].out_channels,
            block2_layers[0].in_channels, block2_layers[1].out_channels,
            block3_layers[0].in_channels, block3_layers[1].out_channels,
            self.current_model.fc.in_features, self.current_model.fc.out_features
        ]
        
        return state
    except Exception as e:
        print(f"获取状态时出错: {str(e)}")
        raise 