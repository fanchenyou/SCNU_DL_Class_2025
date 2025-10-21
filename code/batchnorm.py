import torch
import torch.nn as nn
from typing import Tuple, Optional

def batch_norm(
    X: torch.Tensor, 
    gamma: torch.Tensor, 
    beta: torch.Tensor, 
    moving_mean: torch.Tensor, 
    moving_var: torch.Tensor, 
    eps: float = 1e-5, 
    momentum: float = 0.1
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """批量归一化操作的实现
    
    Args:
        X: 输入张量，可以是形状为(批量大小, 特征数)的全连接层输入，
           或形状为(批量大小, 通道数, 高度, 宽度)的卷积层输入
        gamma: 缩放参数，可学习
        beta: 平移参数，可学习
        moving_mean: 全局移动平均均值
        moving_var: 全局移动平均方差
        eps: 防止除零错误的小值
        momentum: 更新移动平均的动量
        
    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: 
            - 归一化、缩放和平移后的输出
            - 更新后的移动平均均值
            - 更新后的移动平均方差
    """
    # 判断是否处于训练模式
    if not torch.is_grad_enabled():
        # 推理模式下使用移动平均得到的均值和方差
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        # 确保输入形状是2维(全连接层)或4维(卷积层)
        assert len(X.shape) in (2, 4), f"输入形状应为2维或4维，而不是{len(X.shape)}维"
        
        if len(X.shape) == 2:
            # 全连接层：在特征维度(轴0)上计算均值和方差
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            # 卷积层：在通道维度(轴1)上计算均值和方差，保持维度以便广播
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        
        # 训练模式下使用当前批次的均值和方差
        X_hat = (X - mean) / torch.sqrt(var + eps)
        
        # 更新移动平均的均值和方差
        moving_mean = (1.0 - momentum) * moving_mean + momentum * mean.detach()
        moving_var = (1.0 - momentum) * moving_var + momentum * var.detach()
    
    # 缩放和平移
    Y = gamma * X_hat + beta
    
    return Y, moving_mean, moving_var


# 测试用例：使用形状为(B=2,C=3,H=10,W=10)的输入
def test_batch_norm():
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    
    # 创建测试输入张量：形状为(2, 3, 10, 10)
    # B=2 (批量大小), C=3 (通道数), H=10 (高度), W=10 (宽度)
    batch_size, channels, height, width = 2, 3, 5, 5
    X = torch.randn(batch_size, channels, height, width)
    
    # 初始化可学习参数 gamma 和 beta
    # gamma 和 beta 的形状应与通道数匹配，卷积层中通常为 (1, channels, 1, 1) 以便广播
    gamma = torch.ones(1, channels, 1, 1)
    beta = torch.zeros(1, channels, 1, 1)
    
    # 初始化移动平均均值和方差
    # 同样，形状应为 (1, channels, 1, 1)
    moving_mean = torch.zeros(1, channels, 1, 1)
    moving_var = torch.ones(1, channels, 1, 1)
    
    # 设置超参数
    eps = 1e-5
    momentum = 0.1
    
    # 打印输入信息
    print(f"输出张量: {X}")
    print(f"输入张量形状: {X.shape}")
    print(f"输入张量均值: {X.mean().item():.4f}")
    print(f"输入张量标准差: {X.std().item():.4f}")
    print("-" * 50)
    
    # 训练模式下的前向传播
    with torch.enable_grad():
        print("训练模式:")
        Y_train, new_moving_mean, new_moving_var = batch_norm(
            X, gamma, beta, moving_mean, moving_var, eps, momentum
        )
        
        print(f"输出张量形状: {Y_train.shape}")
        print(f"输出张量均值: {Y_train.mean().item():.4f}")
        print(f"输出张量标准差: {Y_train.std().item():.4f}")
        # print(f"更新后的移动均值形状: {new_moving_mean.shape}")
        print(f"更新后的移动均值: {new_moving_mean.view(-1).tolist()}")
        # print(f"更新后的移动方差形状: {new_moving_var.shape}")
        print(f"更新后的移动方差: {new_moving_var.view(-1).tolist()}")
        print("-" * 50)
    
    # 推理模式下的前向传播
    with torch.no_grad():
        print("推理模式:")
        Y_infer, _, _ = batch_norm(
            X, gamma, beta, new_moving_mean, new_moving_var, eps, momentum
        )
        
        print(f"输出张量形状: {Y_infer.shape}")
        print(f"输出张量均值: {Y_infer.mean().item():.4f}")
        print(f"输出张量标准差: {Y_infer.std().item():.4f}")
    
    # 验证是否与PyTorch内置的BatchNorm2d结果一致
    print("-" * 50)
    print("与PyTorch内置BatchNorm2d比较:")
    
    # 创建PyTorch内置的BatchNorm2d层
    bn_layer = nn.BatchNorm2d(channels, eps=eps, momentum=momentum)
    # 手动设置gamma和beta的值以匹配我们的实现
    with torch.no_grad():
        bn_layer.weight[:] = gamma.view(-1)
        bn_layer.bias[:] = beta.view(-1)
        bn_layer.running_mean[:] = moving_mean.view(-1)
        bn_layer.running_var[:] = moving_var.view(-1)
    
    # 内置层的前向传播
    Y_builtin = bn_layer(X)
    
    print(f"内置BatchNorm2d输出形状: {Y_builtin.shape}")
    print(f"内置BatchNorm2d输出均值: {Y_builtin.mean().item():.4f}")
    print(f"内置BatchNorm2d输出标准差: {Y_builtin.std().item():.4f}")
    
    # 计算与内置实现的绝对误差
    abs_error = torch.abs(Y_train - Y_builtin).mean().item()
    print(f"与内置实现的平均绝对误差: {abs_error:.6f}")


# 运行测试用例
if __name__ == "__main__":
    test_batch_norm()