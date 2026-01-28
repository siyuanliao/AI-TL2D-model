# 二维定深声传播损失场预测-轻量化条件平均场网络 LCMF-Net
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from my_functions import calculate_model_complexity


# ==================== 数据工具 ====================
def split_indices_by_year(n_samples: int = 18876, years: int = 13, months: int = 12, points: int = 121):
    """按年份划分训练/验证/测试集索引"""
    assert n_samples == years * months * points, f"样本数 {n_samples} 与 {years}*{months}*{points} 不匹配"
    
    year_ids = np.repeat(np.arange(1, years + 1), months * points)
    train_idx = np.where(np.isin(year_ids, list(range(1, 8)) + [10, 11, 12, 13]))[0]
    val_idx = np.where(year_ids == 8)[0]
    test_idx = np.where(year_ids == 9)[0]
    return train_idx, val_idx, test_idx


def gen_background(target):
    """生成历史平均场: (12, 121, 36, 250)"""
    return target.reshape(13, 12, 121, 36, 250).mean(axis=(0))


# ==================== 数据集 ====================
class SoundField2DDataset(Dataset):
    def __init__(self, input1, input2, target, indices):
        self.input1 = input1[indices].astype(np.float32)
        # 注意：input2现在是二维场 (12, 121, 36, 250)
        month_indices = (indices // 121) % 12
        pos_indices = indices % 121
        self.input2 = input2[month_indices, pos_indices, :, :].astype(np.float32)
        self.target = target[indices].astype(np.float32)
        
    def __len__(self):
        return len(self.input1)
    
    def __getitem__(self, idx):
        x2 = self.input2[idx]  # [36, 250]
        y = self.target[idx]   # [36, 250]
        return torch.from_numpy(self.input1[idx]), torch.from_numpy(x2), torch.from_numpy(y)


# ==================== 模型组件 ====================
class ResBlockFiLM2D(nn.Module):
    def __init__(self, channels, cond_dim, dilation=1, dropout=0.05, gn_groups=8):
        super().__init__()
        # 使用普通二维卷积，可选择是否使用膨胀卷积
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, 
                              padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(gn_groups, channels)
        self.film = nn.Linear(cond_dim, channels * 2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3,
                              padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(gn_groups, channels)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x, cond):
        h = self.norm1(self.conv1(x))
        gamma, beta = self.film(cond).chunk(2, 1)
        h = h * (1 + gamma[..., None, None]) + beta[..., None, None]
        h = self.dropout(F.relu(h, inplace=True))
        h = self.norm2(self.conv2(h))
        return F.relu(x + h, inplace=True)


class SF2DCondResNet(nn.Module):
    def __init__(self, x1_dim=52, base_ch=64, cond_dim=128, dropout=0.05):
        super().__init__()
        # 膨胀率序列，用于不同感受野
        dilations = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8)
        
        # 环境参数编码器(Environmental Parameter Encoder)
        self.cond_mlp = nn.Sequential(
            nn.Linear(x1_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim)
        )
        
        # 条件特征调制器 (Conditional Feature Modulator)
        self.stem = nn.Sequential(
            nn.Conv2d(1, base_ch//2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch//2),
            nn.Conv2d(base_ch//2, base_ch, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch),
            nn.ReLU(inplace=True)
        )
        
        self.blocks = nn.ModuleList([
            ResBlockFiLM2D(base_ch, cond_dim, dilation=d, dropout=dropout)
            for d in dilations
        ])
        
        # 声场重建器 (Sound Field Reconstructor) - 预测二维传播损失场
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 4, 1, kernel_size=1)  # 输出单通道二维场
        )
        nn.init.zeros_(self.head[-1].weight)
        if self.head[-1].bias is not None:
            nn.init.zeros_(self.head[-1].bias)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: 条件向量 [batch, x1_dim]
            x2: 历史平均场输入 [batch, height=36, width=250]
        Returns:
            二维传播损失场 [batch, height=36, width=250]
        """
        # 添加通道维度: [batch, 1, height, width]
        x2 = x2.unsqueeze(1)
        
        cond = self.cond_mlp(x1)
        h = self.stem(x2)
        
        for block in self.blocks:
            h = block(h, cond)
        
        # 输出: [batch, 1, height, width] -> [batch, height, width]
        return self.head(h).squeeze(1)


# ==================== 训练工具 ====================
@dataclass
class Metrics:
    loss: float
    rmse: float


def compute_metrics(pred, y, loss_val):
    diff = pred - y
    return Metrics(float(loss_val), float(diff.abs().mean()), float(torch.sqrt((diff**2).mean())))


def set_seed(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ==================== 主训练函数 ====================
def main():
    # 配置参数
    out_dir = "./outputs_2d"
    epochs, batch_size = 120, 16
    lr, weight_decay = 1e-3, 1e-4
    patience, seed = 20, 42
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    # 加载数据
    input1 = np.load("shareddata/sf_input.npy", mmap_mode="r").astype(np.float32)
    target = np.load("shareddata/sf_res.npy", mmap_mode="r").astype(np.float32)  # 二维数据
    assert len(input1) == len(target), "数据长度不匹配"
    
    # 生成背景场并标准化
    input2 = gen_background(target)
    x1_mean, x1_std = input1.mean(0), input1.std(0) + 1e-6
    t_mean, t_std = target.mean(), target.std() + 1e-6
    
    input1 = (input1 - x1_mean) / x1_std
    input2 = (input2 - t_mean) / t_std
    target = (target - t_mean) / t_std
    
    # 数据划分
    train_idx, val_idx, test_idx = split_indices_by_year(len(input1))
    print(f"数据划分: 训练{len(train_idx)}, 验证{len(val_idx)}, 测试{len(test_idx)}")
    
    # 数据加载器
    train_loader = DataLoader(
        SoundField2DDataset(input1, input2, target, train_idx),
        batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        SoundField2DDataset(input1, input2, target, val_idx),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    test_loader = DataLoader(
        SoundField2DDataset(input1, input2, target, test_idx),
        batch_size=batch_size, shuffle=False, pin_memory=True
    )
    
    # 初始化模型
    model = SF2DCondResNet().to(device)
    total_params, trainable_params = calculate_model_complexity(model)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    #训练时只使用了L1损失函数，mse只用于评估训练效果，而不进入梯度下降过程
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    # ==================== 新增：训练记录字典 ====================
    training_history = {
        'epoch': [],
        'learning_rate': [],
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],  # 标准化尺度
        'val_rmse': [],  # 标准化尺度
        'val_loss_real': [],  # 真实尺度
        'val_rmse_real': []  # 真实尺度
    }
    
    # 训练循环
    os.makedirs(out_dir, exist_ok=True)
    best_val, patience_counter = float("inf"), 0
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_loss, train_rmse = 0, 0
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                loss = criterion(pred, y)
                mse_loss = criterion2(pred, y)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            with torch.no_grad():
                train_loss += loss.item()
                train_rmse += mse_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d}: loss={loss.item():.5f}")
        
        # 验证
        model.eval()
        val_loss, val_rmse = 0, 0
        val_loss_real, val_rmse_real = 0, 0
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    pred = model(x1, x2)
                    # 真实尺度的传播损失误差
                    pred_real = pred * t_std + t_mean
                    y_real = y * t_std + t_mean
                    
                    loss = criterion(pred, y)
                    loss_real = criterion(pred_real, y_real)
                    mse_loss = criterion2(pred, y)
                    mse_loss_real = criterion2(pred_real, y_real)
                
                val_loss += loss.item()
                val_loss_real += loss_real.item()
                val_rmse += mse_loss.item()
                val_rmse_real += mse_loss_real.item()
        
        # 计算平均值
        train_loss_avg = train_loss / len(train_loader)
        train_rmse_avg = np.sqrt(train_rmse / len(train_loader))
        val_loss_avg = val_loss / len(val_loader)
        val_rmse_avg = np.sqrt(val_rmse / len(val_loader))
        val_loss_real_avg = val_loss_real / len(val_loader)
        val_rmse_real_avg = np.sqrt(val_rmse_real / len(val_loader))
        
        # ==================== 新增：记录训练历史 ====================
        training_history['epoch'].append(epoch)
        training_history['learning_rate'].append(optimizer.param_groups[0]['lr'])
        training_history['train_loss'].append(train_loss_avg)
        training_history['train_rmse'].append(train_rmse_avg)
        training_history['val_loss'].append(val_loss_avg)
        training_history['val_rmse'].append(val_rmse_avg)
        training_history['val_loss_real'].append(val_loss_real_avg)
        training_history['val_rmse_real'].append(val_rmse_real_avg)
        
        scheduler.step(val_loss_avg)
        
        # 输出
        print(f"[Epoch {epoch:03d}] lr={optimizer.param_groups[0]['lr']:.2e} | "
              f"train: loss={train_loss_avg:.3f} rmse={train_rmse_avg:.3f} | "
              f"val: loss={val_loss_avg:.3f} rmse={val_rmse_avg:.3f} | "
              f"time={time.time()-t0:.1f}s")
        
        print(f"验证集的真实传播损失 loss={val_loss_real_avg:.3f} rmse={val_rmse_real_avg:.3f}")
        
        # 保存模型
        torch.save({
            "epoch": epoch, "best_val": best_val,
            "model": model.state_dict(), "optimizer": optimizer.state_dict()
        }, os.path.join(out_dir, "last.pt"))
        
        if val_loss_avg < best_val - 1e-6:
            best_val = val_loss_avg
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))
            print(f"✅ 新最佳模型: val_loss={best_val:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发 (patience={patience})")
                break
    
    # ==================== 新增：保存训练历史 ====================
    history_path = os.path.join(out_dir, "training_history.npz")
    np.savez(history_path, **training_history)
    print(f"训练历史已保存到: {history_path}")
    
    
    # 测试
    model.load_state_dict(torch.load(os.path.join(out_dir, "best.pt"), map_location=device))
    model.eval()
    
    test_loss, test_rmse = 0, 0
    test_loss_real, test_rmse_real = 0, 0
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model(x1, x2)
                pred_real = pred * t_std + t_mean
                y_real = y * t_std + t_mean
                
                loss1 = criterion(pred, y)
                loss2 = criterion2(pred, y)
                loss_real1 = criterion(pred_real, y_real)
                loss_real2 = criterion2(pred_real, y_real)
            
            test_loss += loss1.item()
            test_rmse += loss2.item()
            test_loss_real += loss_real1.item()
            test_rmse_real += loss_real2.item()
    
    test_loss_avg = test_loss / len(test_loader)
    test_rmse_avg = np.sqrt(test_rmse / len(test_loader))
    test_loss_real_avg = test_loss_real / len(test_loader)
    test_rmse_real_avg = np.sqrt(test_rmse_real / len(test_loader))
    
    print(f"[TEST] loss={test_loss_avg:.5f} rmse={test_rmse_avg:.5f}")
    print(f"[TEST] 真实传播损失尺度 loss={test_loss_real_avg:.5f} rmse={test_rmse_real_avg:.5f}")
    
    # ==================== 新增：保存测试结果 ====================
    test_results = {
        'test_loss': test_loss_avg,
        'test_rmse': test_rmse_avg,
        'test_loss_real': test_loss_real_avg,
        'test_rmse_real': test_rmse_real_avg
    }
    
    # 将测试结果添加到训练历史中
    training_history['test_results'] = test_results
    np.savez(history_path, **training_history)
    
    print("训练完成")


if __name__ == "__main__":
    main()