# 二维定深声传播损失场预测-条件生成对抗网络基准模型 cGAN-2D
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


# ==================== 生成器组件 ====================
class ResBlockFiLM2D(nn.Module):
    def __init__(self, channels, cond_dim, dilation=1, dropout=0.05, gn_groups=8):
        super().__init__()
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


class Generator(nn.Module):
    """生成器网络"""
    def __init__(self, x1_dim=52, base_ch=64, cond_dim=128, dropout=0.05):
        super().__init__()
        # 膨胀率序列
        dilations = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8)
        
        # 环境参数编码器
        self.cond_mlp = nn.Sequential(
            nn.Linear(x1_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim)
        )
        
        # 条件特征调制器
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
        
        # 声场重建器
        self.head = nn.Sequential(
            nn.Conv2d(base_ch, base_ch // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 2, base_ch // 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_ch // 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch // 4, 1, kernel_size=1)
        )
        nn.init.xavier_uniform_(self.head[-1].weight, gain=0.1)
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
        # 添加通道维度 [batch, 1, height, width]
        x2 = x2.unsqueeze(1)
        cond = self.cond_mlp(x1)
        h = self.stem(x2)
        
        for block in self.blocks:
            h = block(h, cond)
        
        return self.head(h).squeeze(1)


# ==================== 判别器组件 ====================
class ConditionalDiscriminator(nn.Module):
    """条件判别器网络"""
    def __init__(self, x1_dim=52, base_ch=64, img_height=36, img_width=250):
        super().__init__()
        
        # 计算下采样后的空间维度
        # 经过3次下采样(2x)：36->18->9->5, 250->125->63->32
        down_height = img_height // 8
        down_width = img_width // 8
        
        # 条件编码器 - 生成与下采样后尺寸匹配的特征
        self.cond_encoder = nn.Sequential(
            nn.Linear(x1_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, down_height * down_width),  # 5 * 32 = 160
        )
        
        # 图像特征提取器
        self.conv_layers = nn.ModuleList([
            # 输入: [batch, 1, 36, 250] (声场)
            nn.Conv2d(1, base_ch, kernel_size=4, stride=2, padding=1),  # [18, 125]
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_ch, base_ch*2, kernel_size=4, stride=2, padding=1),  # [9, 63]
            nn.GroupNorm(8, base_ch*2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_ch*2, base_ch*4, kernel_size=4, stride=2, padding=1),  # [5, 32]
            nn.GroupNorm(8, base_ch*4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(base_ch*4, base_ch*8, kernel_size=3, stride=1, padding=1),  # [5, 32]
            nn.GroupNorm(8, base_ch*8),
            nn.LeakyReLU(0.2, inplace=True),
        ])
        
        # 条件特征与图像特征的融合
        self.fusion_layer = nn.Conv2d(base_ch*8 + 1, base_ch*8, kernel_size=1)
        
        # 最终输出层 - 使用线性层输出logits
        self.output_layer = nn.Sequential(
            nn.Conv2d(base_ch*8, 1, kernel_size=3, stride=1, padding=1),  # [5, 32]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # 移除Sigmoid，因为BCEWithLogitsLoss会内部处理
        )
        
    def forward(self, x1, sound_field):
        """
        Args:
            x1: 条件向量 [batch, x1_dim]
            sound_field: 声场 [batch, height, width] 或 [batch, 1, height, width]
        Returns:
            判别器输出logits [batch, 1]
        """
        if sound_field.dim() == 3:
            sound_field = sound_field.unsqueeze(1)
        
        batch_size = sound_field.shape[0]
        
        # 提取声场特征
        features = sound_field
        for layer in self.conv_layers:
            features = layer(features)
        
        # 处理条件信息
        cond_features = self.cond_encoder(x1)
        # reshape条件特征为空间特征图 [batch, 1, 5, 32]
        cond_map = cond_features.view(batch_size, 1, features.size(2), features.size(3))
        
        # 拼接声场特征和条件特征
        combined = torch.cat([features, cond_map], dim=1)
        fused = self.fusion_layer(combined)
        
        # 最终输出logits
        output = self.output_layer(fused)
        return output


# ==================== 条件GAN模型 ====================
class cGAN2D(nn.Module):
    """条件GAN模型"""
    def __init__(self, x1_dim=52, g_base_ch=64, d_base_ch=64, cond_dim=128, dropout=0.05):
        super().__init__()
        self.generator = Generator(x1_dim, g_base_ch, cond_dim, dropout)
        self.discriminator = ConditionalDiscriminator(x1_dim, d_base_ch)
    
    def forward(self, x1, x2):
        """生成阶段前向传播"""
        return self.generator(x1, x2)


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


def gradient_penalty(discriminator, real_data, fake_data, x1, device):
    """计算WGAN-GP的梯度惩罚"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # 插值样本
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # 计算判别器输出
    d_interpolated = discriminator(x1, interpolated)
    
    # 计算梯度
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 计算梯度惩罚
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


# ==================== 主训练函数 ====================
def main():
    # 配置参数
    out_dir = "./outputs_2d_cgan"
    epochs, batch_size = 120, 16
    lr_g, lr_d = 1e-4, 4e-4  # GAN通常使用较低的学习率
    weight_decay = 1e-4
    patience, seed = 20, 42
    lambda_gp = 10  # 梯度惩罚系数
    lambda_l1 = 100  # L1损失系数
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
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
    model = cGAN2D().to(device)
    g_params, g_trainable = calculate_model_complexity(model.generator)
    d_params, d_trainable = calculate_model_complexity(model.discriminator)
    print(f"生成器参数: {g_params:,} (可训练: {g_trainable:,})")
    print(f"判别器参数: {d_params:,} (可训练: {d_trainable:,})")
    print(f"总参数: {g_params + d_params:,}")
    
    # 测试维度
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        x1_test, x2_test, y_test = [x.to(device) for x in test_batch]
        print(f"输入尺寸测试:")
        print(f"  x1: {x1_test.shape}")  # [batch, 52]
        print(f"  x2: {x2_test.shape}")  # [batch, 36, 250]
        print(f"  y: {y_test.shape}")   # [batch, 36, 250]
        
        # 测试生成器
        fake_y = model.generator(x1_test, x2_test)
        print(f"  生成器输出: {fake_y.shape}")  # 应该是 [batch, 36, 250]
        
        # 测试判别器
        d_out_real = model.discriminator(x1_test, y_test.unsqueeze(1))
        d_out_fake = model.discriminator(x1_test, fake_y.unsqueeze(1))
        print(f"  判别器输出(real): {d_out_real.shape}")  # 应该是 [batch, 1]
        print(f"  判别器输出(fake): {d_out_fake.shape}")  # 应该是 [batch, 1]
    
    # 优化器
    optimizer_g = torch.optim.AdamW(
        model.generator.parameters(), 
        lr=lr_g, 
        weight_decay=weight_decay,
        betas=(0.5, 0.999)
    )
    optimizer_d = torch.optim.AdamW(
        model.discriminator.parameters(), 
        lr=lr_d, 
        weight_decay=weight_decay,
        betas=(0.5, 0.999)
    )
    
    # 损失函数 - 使用BCEWithLogitsLoss替代BCELoss
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()  # 使用BCEWithLogitsLoss
    
    # 学习率调度器
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    # 训练记录字典
    training_history = {
        'epoch': [],
        'learning_rate_g': [],
        'learning_rate_d': [],
        'train_g_loss': [],
        'train_d_loss': [],
        'train_l1_loss': [],
        'train_rmse': [],
        'val_g_loss': [],
        'val_d_loss': [],
        'val_l1_loss': [],
        'val_rmse': [],
        'val_l1_loss_real': [],
        'val_rmse_real': [],
        'train_d_acc_real': [],
        'train_d_acc_fake': [],
        'val_d_acc_real': [],
        'val_d_acc_fake': []
    }
    
    # 训练循环
    os.makedirs(out_dir, exist_ok=True)
    best_val, patience_counter = float("inf"), 0
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_g_loss, train_d_loss, train_l1_loss, train_rmse = 0, 0, 0, 0
        train_d_acc_real, train_d_acc_fake = 0, 0
        n_critic = 5  # 每个生成器更新对应的判别器更新次数
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            batch_size = x1.size(0)
            
            # 真实标签和虚假标签
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)
            
            # ==================== 训练判别器 ====================
            model.discriminator.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                # 真实样本
                real_pred = model.discriminator(x1, y.unsqueeze(1))
                d_real_loss = criterion_bce(real_pred, valid)
                
                # 生成虚假样本
                fake_soundfield = model.generator(x1, x2)
                fake_pred = model.discriminator(x1, fake_soundfield.unsqueeze(1).detach())
                d_fake_loss = criterion_bce(fake_pred, fake)
                
                # 判别器总损失
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # 计算判别器准确率
                train_d_acc_real += ((torch.sigmoid(real_pred) > 0.5).float().mean().item())
                train_d_acc_fake += ((torch.sigmoid(fake_pred) < 0.5).float().mean().item())
            
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()
            
            # ==================== 训练生成器 ====================
            # 每n_critic次判别器更新后更新一次生成器
            if batch_idx % n_critic == 0:
                model.generator.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    # 生成虚假样本
                    fake_soundfield = model.generator(x1, x2)
                    
                    # 对抗损失
                    validity = model.discriminator(x1, fake_soundfield.unsqueeze(1))
                    g_adv_loss = criterion_bce(validity, valid)
                    
                    # L1重建损失
                    g_l1_loss = criterion_l1(fake_soundfield, y)
                    g_mse_loss = criterion_mse(fake_soundfield, y)
                    
                    # 生成器总损失
                    g_loss = g_adv_loss + lambda_l1 * g_l1_loss
                
                scaler.scale(g_loss).backward()
                scaler.step(optimizer_g)
                scaler.update()
                
                train_g_loss += g_loss.item()
                train_l1_loss += g_l1_loss.item()
                train_rmse += g_mse_loss.item()
            
            train_d_loss += d_loss.item()
            
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx:3d}/{len(train_loader):3d}: "
                      f"D_loss={d_loss.item():.5f}, D_real={torch.sigmoid(real_pred).mean().item():.3f}, "
                      f"D_fake={torch.sigmoid(fake_pred).mean().item():.3f}, G_loss={g_loss.item() if batch_idx % n_critic == 0 else 'N/A'}")
        
        # 计算训练平均值
        train_d_loss_avg = train_d_loss / len(train_loader)
        train_g_loss_avg = train_g_loss / max(len(train_loader) / n_critic, 1)
        train_l1_loss_avg = train_l1_loss / max(len(train_loader) / n_critic, 1)
        train_rmse_avg = np.sqrt(train_rmse / max(len(train_loader) / n_critic, 1))
        train_d_acc_real_avg = train_d_acc_real / len(train_loader)
        train_d_acc_fake_avg = train_d_acc_fake / len(train_loader)
        
        # ==================== 验证 ====================
        model.eval()
        val_g_loss, val_d_loss, val_l1_loss, val_rmse = 0, 0, 0, 0
        val_l1_loss_real, val_rmse_real = 0, 0
        val_d_acc_real, val_d_acc_fake = 0, 0
        
        with torch.no_grad():
            for x1, x2, y in val_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                batch_size = x1.size(0)
                
                valid = torch.ones(batch_size, 1).to(device)
                fake = torch.zeros(batch_size, 1).to(device)
                
                # 生成器验证
                fake_soundfield = model.generator(x1, x2)
                
                # 对抗损失
                validity = model.discriminator(x1, fake_soundfield.unsqueeze(1))
                g_adv_loss = criterion_bce(validity, valid)
                
                # 重建损失
                g_l1_loss = criterion_l1(fake_soundfield, y)
                g_mse_loss = criterion_mse(fake_soundfield, y)
                
                g_loss = g_adv_loss + lambda_l1 * g_l1_loss
                
                # 判别器验证
                real_pred = model.discriminator(x1, y.unsqueeze(1))
                fake_pred = model.discriminator(x1, fake_soundfield.unsqueeze(1).detach())
                d_real_loss = criterion_bce(real_pred, valid)
                d_fake_loss = criterion_bce(fake_pred, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # 计算判别器准确率
                val_d_acc_real += ((torch.sigmoid(real_pred) > 0.5).float().mean().item())
                val_d_acc_fake += ((torch.sigmoid(fake_pred) < 0.5).float().mean().item())
                
                # 真实尺度的传播损失误差
                pred_real = fake_soundfield * t_std + t_mean
                y_real = y * t_std + t_mean
                l1_loss_real = criterion_l1(pred_real, y_real)
                mse_loss_real = criterion_mse(pred_real, y_real)
                
                val_g_loss += g_loss.item()
                val_d_loss += d_loss.item()
                val_l1_loss += g_l1_loss.item()
                val_rmse += g_mse_loss.item()
                val_l1_loss_real += l1_loss_real.item()
                val_rmse_real += mse_loss_real.item()
        
        # 计算验证平均值
        val_g_loss_avg = val_g_loss / len(val_loader)
        val_d_loss_avg = val_d_loss / len(val_loader)
        val_l1_loss_avg = val_l1_loss / len(val_loader)
        val_rmse_avg = np.sqrt(val_rmse / len(val_loader))
        val_l1_loss_real_avg = val_l1_loss_real / len(val_loader)
        val_rmse_real_avg = np.sqrt(val_rmse_real / len(val_loader))
        val_d_acc_real_avg = val_d_acc_real / len(val_loader)
        val_d_acc_fake_avg = val_d_acc_fake / len(val_loader)
        
        # 记录训练历史
        training_history['epoch'].append(epoch)
        training_history['learning_rate_g'].append(optimizer_g.param_groups[0]['lr'])
        training_history['learning_rate_d'].append(optimizer_d.param_groups[0]['lr'])
        training_history['train_g_loss'].append(train_g_loss_avg)
        training_history['train_d_loss'].append(train_d_loss_avg)
        training_history['train_l1_loss'].append(train_l1_loss_avg)
        training_history['train_rmse'].append(train_rmse_avg)
        training_history['train_d_acc_real'].append(train_d_acc_real_avg)
        training_history['train_d_acc_fake'].append(train_d_acc_fake_avg)
        training_history['val_g_loss'].append(val_g_loss_avg)
        training_history['val_d_loss'].append(val_d_loss_avg)
        training_history['val_l1_loss'].append(val_l1_loss_avg)
        training_history['val_rmse'].append(val_rmse_avg)
        training_history['val_l1_loss_real'].append(val_l1_loss_real_avg)
        training_history['val_rmse_real'].append(val_rmse_real_avg)
        training_history['val_d_acc_real'].append(val_d_acc_real_avg)
        training_history['val_d_acc_fake'].append(val_d_acc_fake_avg)
        
        # 更新学习率
        scheduler_g.step(val_l1_loss_avg)
        scheduler_d.step(val_d_loss_avg)
        
        # 输出训练信息
        print(f"[Epoch {epoch:03d}] "
              f"lr_G={optimizer_g.param_groups[0]['lr']:.2e} "
              f"lr_D={optimizer_d.param_groups[0]['lr']:.2e}")
        print(f"  真实传播损失尺度: L1={val_l1_loss_real_avg:.3f}, RMSE={val_rmse_real_avg:.3f}")
        print(f"  时间: {time.time()-t0:.1f}s")
        
        # 保存模型
        torch.save({
            "epoch": epoch, "best_val": best_val,
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict()
        }, os.path.join(out_dir, "last.pt"))
        
        # 使用L1损失作为早停标准
        if val_l1_loss_avg < best_val - 1e-6:
            best_val = val_l1_loss_avg
            torch.save(model.generator.state_dict(), os.path.join(out_dir, "best_generator.pt"))
            torch.save(model.discriminator.state_dict(), os.path.join(out_dir, "best_discriminator.pt"))
            print(f"✅ 新最佳模型: val_l1_loss={best_val:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"早停触发 (patience={patience})")
                break
    
    # 保存训练历史
    history_path = os.path.join(out_dir, "training_history.npz")
    np.savez(history_path, **training_history)
    print(f"训练历史已保存到: {history_path}")
    
    
    # ==================== 测试 ====================
    print("\n" + "="*50)
    print("测试阶段")
    print("="*50)
    
    # 加载最佳生成器
    model.generator.load_state_dict(
        torch.load(os.path.join(out_dir, "best_generator.pt"), map_location=device)
    )
    model.generator.eval()
    
    test_l1_loss, test_rmse = 0, 0
    test_l1_loss_real, test_rmse_real = 0, 0
    
    with torch.no_grad():
        for x1, x2, y in test_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                pred = model.generator(x1, x2)
                pred_real = pred * t_std + t_mean
                y_real = y * t_std + t_mean
                
                l1_loss = criterion_l1(pred, y)
                mse_loss = criterion_mse(pred, y)
                l1_loss_real = criterion_l1(pred_real, y_real)
                mse_loss_real = criterion_mse(pred_real, y_real)
            
            test_l1_loss += l1_loss.item()
            test_rmse += mse_loss.item()
            test_l1_loss_real += l1_loss_real.item()
            test_rmse_real += mse_loss_real.item()
    
    test_l1_loss_avg = test_l1_loss / len(test_loader)
    test_rmse_avg = np.sqrt(test_rmse / len(test_loader))
    test_l1_loss_real_avg = test_l1_loss_real / len(test_loader)
    test_rmse_real_avg = np.sqrt(test_rmse_real / len(test_loader))
    
    print(f"[TEST] 标准化尺度: L1={test_l1_loss_avg:.5f}, RMSE={test_rmse_avg:.5f}")
    print(f"[TEST] 真实传播损失尺度: L1={test_l1_loss_real_avg:.5f}, RMSE={test_rmse_real_avg:.5f}")
    
    # 保存测试结果
    test_results = {
        'test_l1_loss': test_l1_loss_avg,
        'test_rmse': test_rmse_avg,
        'test_l1_loss_real': test_l1_loss_real_avg,
        'test_rmse_real': test_rmse_real_avg
    }
    
    training_history['test_results'] = test_results
    np.savez(history_path, **training_history)
    
    
    print("GAN训练完成")


if __name__ == "__main__":
    main()