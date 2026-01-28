import numpy as np
from scipy import ndimage

def replace_inf_with_local_mean(tensor, kernel_size=3, max_iterations=10):
    """
    使用周围非Inf值的平均值替换4维张量中的Inf值
    
    参数:
        tensor: 4维numpy数组
        kernel_size: 卷积核大小，必须是奇数，默认为3
        max_iterations: 最大迭代次数，用于处理连续Inf区域
    
    返回:
        处理后的4维张量
    """
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size必须是奇数")
    
    # 创建原始张量的副本
    result = tensor.copy().astype(np.float64)
    
    # 创建Inf值掩码
    inf_mask = np.isinf(result)
    
    # 如果没有Inf值，直接返回
    if not np.any(inf_mask):
        return result
    
    # 迭代处理，直到所有Inf值都被替换或达到最大迭代次数
    for iteration in range(max_iterations):
        # 检查是否还有Inf值
        current_inf_mask = np.isinf(result)
        if not np.any(current_inf_mask):
            break
            
        # 创建一个临时数组，将Inf值替换为NaN以便计算
        temp = result.copy()
        temp[current_inf_mask] = np.nan
        
        # 对每个通道和批次分别处理
        for b in range(temp.shape[0]):
            for c in range(temp.shape[1]):
                # 获取当前通道的2D切片
                channel_slice = temp[b, c, :, :]
                
                # 使用均值滤波，忽略NaN值
                mean_filtered = ndimage.generic_filter(
                    channel_slice, 
                    np.nanmean, 
                    size=kernel_size, 
                    mode='constant', 
                    cval=np.nan
                )
                
                # 找到当前通道中的Inf位置
                channel_inf_mask = np.isinf(result[b, c, :, :])
                
                # 用周围非Inf值的平均值替换Inf值
                if np.any(channel_inf_mask):
                    # 只有在均值不是NaN（即周围有非Inf值）时才替换
                    valid_replacements = ~np.isnan(mean_filtered[channel_inf_mask])
                    replace_indices = np.where(channel_inf_mask)
                    valid_indices = (replace_indices[0][valid_replacements], 
                                   replace_indices[1][valid_replacements])
                    
                    if len(valid_indices[0]) > 0:
                        result[b, c][valid_indices] = mean_filtered[valid_indices]
        
        print(f"迭代 {iteration + 1}: 剩余Inf值数量: {np.sum(np.isinf(result))}")
    
    # 如果还有Inf值（可能是孤立的或大块连续Inf区域），用全局非Inf均值替换
    final_inf_mask = np.isinf(result)
    if np.any(final_inf_mask):
        global_mean = np.nanmean(temp)  # temp中Inf已被替换为NaN
        if not np.isnan(global_mean):
            result[final_inf_mask] = global_mean
        else:
            # 如果所有值都是Inf，用0替换
            result[final_inf_mask] = 0.0
        print(f"使用全局均值替换剩余 {np.sum(final_inf_mask)} 个Inf值")
    
    return result

def replace_inf_with_expanding_mean(tensor, max_window_size=11):
    """
    使用扩展窗口方法替换Inf值，逐步扩大搜索范围直到找到非Inf值
    
    参数:
        tensor: 4维numpy数组
        max_window_size: 最大窗口大小，必须是奇数
    
    返回:
        处理后的4维张量
    """
    print("ok")
    result = tensor.copy().astype(np.float64)
    inf_mask = np.isinf(result)
    
    if not np.any(inf_mask):
        return result
    
    # 获取Inf位置的坐标
    inf_coords = np.where(inf_mask)
    
    for idx in range(len(inf_coords[0])):
        print(f"进度: {idx}/{len(inf_coords[0])}")
        b, c, h, w = inf_coords[0][idx], inf_coords[1][idx], inf_coords[2][idx], inf_coords[3][idx]
        
        # 逐步扩大窗口大小，直到找到非Inf值
        for window_size in range(3, max_window_size + 1, 2):
            half_window = window_size // 2
            
            # 计算窗口边界
            h_start = max(0, h - half_window)
            h_end = min(result.shape[2], h + half_window + 1)
            w_start = max(0, w - half_window)
            w_end = min(result.shape[3], w + half_window + 1)
            
            # 提取窗口区域
            window = result[b, c, h_start:h_end, w_start:w_end]
            
            # 找到窗口中的非Inf值
            non_inf_values = window[~np.isinf(window)]
            
            if len(non_inf_values) > 0:
                # 用非Inf值的均值替换当前Inf值
                result[b, c, h, w] = np.mean(non_inf_values)
                break
        else:
            # 如果所有窗口都找不到非Inf值，使用全局均值
            non_inf_global = result[~np.isinf(result)]
            if len(non_inf_global) > 0:
                result[b, c, h, w] = np.mean(non_inf_global)
            else:
                result[b, c, h, w] = 0.0
    
    return result

# 测试函数
def test_inf_replacement():
    """测试Inf值替换函数"""
    # 创建一个包含Inf值的4维张量
    np.random.seed(42)
    tensor = np.random.randn(2, 3, 5, 5).astype(np.float32)
    
    # 添加一些Inf值，包括连续的Inf区域
    tensor[0, 0, 1:4, 1:4] = np.inf  # 3x3的Inf区域
    tensor[0, 1, 2, 2] = np.inf      # 孤立的Inf
    tensor[1, 2, 0, 0] = -np.inf     # 负Inf
    
    print("原始张量中的Inf值数量:", np.sum(np.isinf(tensor)))
    print("原始张量形状:", tensor.shape)
    
    # 使用方法1：局部均值替换
    result1 = replace_inf_with_local_mean(tensor)
    print("方法1处理后Inf值数量:", np.sum(np.isinf(result1)))
    
    # 使用方法2：扩展窗口方法
    result2 = replace_inf_with_expanding_mean(tensor)
    print("方法2处理后Inf值数量:", np.sum(np.isinf(result2)))
    
    return result1, result2

def calculate_model_complexity(model):
    """
    计算模型的参数数量
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("🔍 模型复杂度分析")
    print("=" * 50)
    print(f"总参数量: {total_params:,} 个参数")
    print(f"可训练参数量: {trainable_params:,} 个参数")
    print(f"约: {total_params/1e6:.2f} 百万参数")
    
    # 与经典网络对比
    print("\n📊 与经典网络对比:")
    print(f"LeNet-5: ~60,000 参数")
    print(f"AlexNet: ~60 million 参数") 
    print(f"您的网络: ~{total_params/1e6:.1f} million 参数")
    print(f"参数规模是 LeNet-5 的 {total_params/60000:.1f} 倍")
    
    return total_params, trainable_params

if __name__ == "__main__":
    result1, result2 = test_inf_replacement()