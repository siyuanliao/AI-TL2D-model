# 二维定深声传播损失场预测-基准模型：可调通道数的轻量化U-Net
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
        month_indices = (indices // 121) % 12
        pos_indices = indices % 121
        self.input2 = input2[month_indices, pos_indices, :, :].astype(np.float32)
        self.target = target[indices].astype(np.float32)
        
    def __len__(self):
        return len(self.input1)
    
    def __getitem__(self, idx):
        x2 = self.input2[idx]
        y = self.target[idx]
        return torch.from_numpy(self.input1[idx]), torch.from_numpy(x2), torch.from_numpy(y)


# ==================== 基准模型：轻量化U-Net ====================
class DoubleConv(nn.Module):
    """U-Net中的双层卷积块"""
    def __init__(self, in_channels, out_channels, mid_channels=None, dropout=0.05):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, mid_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    """下采样块"""
    def __init__(self, in_channels, out_channels, dropout=0.05):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, dropout=dropout)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """上采样块"""
    def __init__(self, in_channels, out_channels, bilinear=True, dropout=0.05):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, dropout=dropout)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, dropout=dropout)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 处理尺寸对齐问题
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class ConditionEncoder(nn.Module):
    """环境参数编码器"""
    def __init__(self, x1_dim=52, cond_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(x1_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)


class UNet2D(nn.Module):
    """
    基准模型：可调通道数的轻量化U-Net
    参数:
        base_channels: 基础通道数，可调整网络参数量
        cond_dim: 条件向量维度
        bilinear: 是否使用双线性插值上采样
    """
    def __init__(self, x1_dim=52, base_channels=32, cond_dim=128, 
                 bilinear=True, dropout=0.05):
        super().__init__()
        self.base_channels = base_channels
        self.bilinear = bilinear
        self.cond_dim = cond_dim
        
        # 条件编码器
        self.cond_encoder = ConditionEncoder(x1_dim, cond_dim)
        
        # 初始卷积层
        self.inc = DoubleConv(1 + cond_dim // 4, base_channels, dropout=dropout)
        
        # 下采样路径
        self.down1 = DownBlock(base_channels, base_channels * 2, dropout=dropout)
        self.down2 = DownBlock(base_channels * 2, base_channels * 4, dropout=dropout)
        self.down3 = DownBlock(base_channels * 4, base_channels * 8, dropout=dropout)
        
        # 瓶颈层
        factor = 2 if bilinear else 1
        self.bottleneck = DoubleConv(base_channels * 8, base_channels * 16 // factor, dropout=dropout)
        
        # 条件融合模块
        # 瓶颈层输出通道数: base_channels * 16 // factor
        bottleneck_channels = base_channels * 16 // factor
        self.cond_fusion = nn.Sequential(
            nn.Linear(bottleneck_channels + cond_dim, bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout)
        )
        
        # 上采样路径
        # 注意: UpBlock的in_channels是上采样后与对应层拼接后的通道数
        # 对于up1: 输入是融合后的特征(维度=bottleneck_channels) + 下采样特征(维度=base_channels*4)
        self.up1 = UpBlock(bottleneck_channels + base_channels * 4, base_channels * 8, bilinear, dropout=dropout)
        self.up2 = UpBlock(base_channels * 8 + base_channels * 2, base_channels * 4, bilinear, dropout=dropout)
        self.up3 = UpBlock(base_channels * 4 + base_channels, base_channels, bilinear, dropout=dropout)
        
        # 输出层
        self.outc = nn.Conv2d(base_channels, 1, kernel_size=1)
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x1, x2):
        """
        Args:
            x1: 条件向量 [batch, x1_dim]
            x2: 历史平均场输入 [batch, height=36, width=250]
        Returns:
            二维传播损失场 [batch, height=36, width=250]
        """
        batch_size = x1.size(0)
        
        # 添加通道维度
        x2 = x2.unsqueeze(1)  # [batch, 1, 36, 250]
        
        # 编码条件向量
        cond = self.cond_encoder(x1)  # [batch, cond_dim]
        
        # 将条件向量扩展到空间维度
        cond_spatial = cond.unsqueeze(-1).unsqueeze(-1)  # [batch, cond_dim, 1, 1]
        cond_spatial = cond_spatial.expand(-1, -1, 36, 250)  # [batch, cond_dim, 36, 250]
        
        # 选择部分条件通道与输入拼接
        cond_part = cond_spatial[:, :self.cond_dim // 4, :, :]  # 使用1/4的条件通道
        x = torch.cat([x2, cond_part], dim=1)  # [batch, 1+cond_dim//4, 36, 250]
        
        # 编码路径
        x1_enc = self.inc(x)  # [batch, base_channels, 36, 250]
        x2_enc = self.down1(x1_enc)  # [batch, base_channels*2, 18, 125]
        x3_enc = self.down2(x2_enc)  # [batch, base_channels*4, 9, 62]
        x4_enc = self.down3(x3_enc)  # [batch, base_channels*8, 4, 31]
        
        # 瓶颈层
        bottleneck_out = self.bottleneck(x4_enc)  # [batch, base_channels*16//factor, 4, 31]
        
        # 全局平均池化获取全局特征
        global_feat = F.adaptive_avg_pool2d(bottleneck_out, (1, 1))  # [batch, base_channels*16//factor, 1, 1]
        global_feat = global_feat.flatten(1)  # [batch, base_channels*16//factor]
        
        # 条件融合
        fused = torch.cat([global_feat, cond], dim=1)  # [batch, base_channels*16//factor + cond_dim]
        fused = self.cond_fusion(fused)  # [batch, base_channels*16//factor]
        
        # 重塑条件特征
        fused = fused.unsqueeze(-1).unsqueeze(-1)  # [batch, base_channels*16//factor, 1, 1]
        # 扩展回原来的空间尺寸
        fused_spatial = fused.expand(-1, -1, 4, 31)  # [batch, base_channels*16//factor, 4, 31]
        
        # 与瓶颈层输出相加（残差连接）
        x4_fused = bottleneck_out + fused_spatial  # [batch, base_channels*16//factor, 4, 31]
        
        # 解码路径
        x = self.up1(x4_fused, x3_enc)  # [batch, base_channels*8, 9, 62]
        x = self.up2(x, x2_enc)  # [batch, base_channels*4, 18, 125]
        x = self.up3(x, x1_enc)  # [batch, base_channels, 36, 250]
        
        # 输出
        out = self.outc(x)  # [batch, 1, 36, 250]
        return out.squeeze(1)  # [batch, 36, 250]


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
    out_dir = "./outputs_2d_baseline"
    epochs, batch_size = 120, 16
    lr, weight_decay = 1e-3, 1e-4
    patience, seed = 20, 42
    
    # 网络参数（可调整以控制参数量）
    base_channels = 32  # 基础通道数，可调整：16, 32, 64等
    cond_dim = 128  # 条件向量维度
    bilinear = True  # 上采样方式
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    print(f"基准模型参数: base_channels={base_channels}, cond_dim={cond_dim}")
    
    # 加载数据
    input1 = np.load("shareddata/sf_input.npy", mmap_mode="r").astype(np.float32)
    target = np.load("shareddata/sf_res.npy", mmap_mode="r").astype(np.float32)
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
    model = UNet2D(
        x1_dim=52,
        base_channels=base_channels,
        cond_dim=cond_dim,
        bilinear=bilinear,
        dropout=0.05
    ).to(device)
    
    # 测试前向传播
    with torch.no_grad():
        test_x1 = torch.randn(2, 52).to(device)
        test_x2 = torch.randn(2, 36, 250).to(device)
        print(f"测试前向传播:")
        print(f"  输入x1形状: {test_x1.shape}")
        print(f"  输入x2形状: {test_x2.shape}")
        
        output = model(test_x1, test_x2)
        print(f"  输出形状: {output.shape}")
        print(f"  预期输出: [2, 36, 250]")
        
        # 检查每个层的输出尺寸
        print("\n检查各层输出尺寸:")
        x2 = test_x2.unsqueeze(1)
        cond = model.cond_encoder(test_x1)
        print(f"  条件编码后: {cond.shape}")  # [2, 128]
        
        cond_spatial = cond.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 36, 250)
        cond_part = cond_spatial[:, :model.cond_dim // 4, :, :]
        x = torch.cat([x2, cond_part], dim=1)
        print(f"  初始拼接后: {x.shape}")  # [2, 33, 36, 250] (1+128/4=33)
        
        x1_enc = model.inc(x)
        print(f"  inc后: {x1_enc.shape}")  # [2, 32, 36, 250]
        
        x2_enc = model.down1(x1_enc)
        print(f"  down1后: {x2_enc.shape}")  # [2, 64, 18, 125]
        
        x3_enc = model.down2(x2_enc)
        print(f"  down2后: {x3_enc.shape}")  # [2, 128, 9, 62]
        
        x4_enc = model.down3(x3_enc)
        print(f"  down3后: {x4_enc.shape}")  # [2, 256, 4, 31]
        
        bottleneck_out = model.bottleneck(x4_enc)
        print(f"  bottleneck后: {bottleneck_out.shape}")  # [2, 256, 4, 31] (bilinear=True时factor=2)
        
        global_feat = F.adaptive_avg_pool2d(bottleneck_out, (1, 1)).flatten(1)
        print(f"  全局特征: {global_feat.shape}")  # [2, 256]
        
        fused = torch.cat([global_feat, cond], dim=1)
        print(f"  拼接条件后: {fused.shape}")  # [2, 384] (256+128)
        
        fused = model.cond_fusion(fused)
        print(f"  条件融合后: {fused.shape}")  # [2, 256]
        
        fused_spatial = fused.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 4, 31)
        print(f"  重塑扩展后: {fused_spatial.shape}")  # [2, 256, 4, 31]
        
        x4_fused = bottleneck_out + fused_spatial
        print(f"  融合特征: {x4_fused.shape}")  # [2, 256, 4, 31]
        
        x = model.up1(x4_fused, x3_enc)
        print(f"  up1后: {x.shape}")  # [2, 256, 9, 62]
        
        x = model.up2(x, x2_enc)
        print(f"  up2后: {x.shape}")  # [2, 128, 18, 125]
        
        x = model.up3(x, x1_enc)
        print(f"  up3后: {x.shape}")  # [2, 32, 36, 250]
        
        out = model.outc(x)
        print(f"  最终输出: {out.shape}")  # [2, 1, 36, 250]
        
    total_params, trainable_params = calculate_model_complexity(model)
    print(f"模型参数统计: 总参数={total_params:,} 可训练参数={trainable_params:,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.L1Loss()
    criterion2 = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    # 训练记录字典
    training_history = {
        'epoch': [],
        'learning_rate': [],
        'train_loss': [],
        'train_rmse': [],
        'val_loss': [],
        'val_rmse': [],
        'val_loss_real': [],
        'val_rmse_real': [],
        'model_params': {
            'base_channels': base_channels,
            'cond_dim': cond_dim,
            'total_params': total_params,
            'trainable_params': trainable_params
        }
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
        
        # 记录训练历史
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
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "config": {
                "base_channels": base_channels,
                "cond_dim": cond_dim,
                "bilinear": bilinear
            }
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
    
    # 保存训练历史
    history_path = os.path.join(out_dir, "training_history.npz")
    np.savez(history_path, **training_history)
    
    
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
    
    print(f"[基准模型测试结果]")
    print(f"  loss={test_loss_avg:.5f} rmse={test_rmse_avg:.5f}")
    print(f"  真实传播损失尺度 loss={test_loss_real_avg:.5f} rmse={test_rmse_real_avg:.5f}")
    
    # 保存测试结果
    test_results = {
        'test_loss': test_loss_avg,
        'test_rmse': test_rmse_avg,
        'test_loss_real': test_loss_real_avg,
        'test_rmse_real': test_rmse_real_avg
    }
    
    training_history['test_results'] = test_results
    np.savez(history_path, **training_history)
    
    
    print("基准模型训练完成")


if __name__ == "__main__":
    main()