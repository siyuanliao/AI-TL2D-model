# 2D Constant-Depth Sound Propagation Loss Field Prediction - Conditional Generative Adversarial Network Baseline Model cGAN-2D
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from dataclasses import dataclass
from my_functions import calculate_model_complexity


# ==================== Data Utilities ====================
def split_indices_by_year(n_samples: int = 18876, years: int = 13, months: int = 12, points: int = 121):
    """Split indices into train/validation/test sets by year"""
    assert n_samples == years * months * points, f"Number of samples {n_samples} does not match {years}*{months}*{points}"
    
    year_ids = np.repeat(np.arange(1, years + 1), months * points)
    train_idx = np.where(np.isin(year_ids, list(range(1, 8)) + [10, 11, 12, 13]))[0]
    val_idx = np.where(year_ids == 8)[0]
    test_idx = np.where(year_ids == 9)[0]
    return train_idx, val_idx, test_idx


def gen_background(target):
    """Generate historical mean field: (12, 36, 250)"""
    return target.reshape(13, 12, 121, 36, 250).mean(axis=(0, 2))


# ==================== Dataset ====================
class SoundField2DDataset(Dataset):
    def __init__(self, input1, input2, target, indices):
        self.input1 = input1[indices].astype(np.float32)
        # Note: input2 is now a 2D field (12, 121, 36, 250)
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


# ==================== Generator Components ====================
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
    """Generator network"""
    def __init__(self, x1_dim=52, base_ch=64, cond_dim=128, dropout=0.05):
        super().__init__()
        # Dilation rate sequence
        dilations = (1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8)
        
        # Environmental parameter encoder
        self.cond_mlp = nn.Sequential(
            nn.Linear(x1_dim, 128), nn.ReLU(inplace=True),
            nn.Linear(128, 256), nn.ReLU(inplace=True),
            nn.Linear(256, cond_dim)
        )
        
        # Conditional feature modulator
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
        
        # Sound field reconstructor
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
            x1: Condition vector [batch, x1_dim]
            x2: Historical mean field input [batch, height=36, width=250]
        Returns:
            2D propagation loss field [batch, height=36, width=250]
        """
        # Add channel dimension [batch, 1, height, width]
        x2 = x2.unsqueeze(1)
        cond = self.cond_mlp(x1)
        h = self.stem(x2)
        
        for block in self.blocks:
            h = block(h, cond)
        
        return self.head(h).squeeze(1)


# ==================== Discriminator Components ====================
class ConditionalDiscriminator(nn.Module):
    """Conditional discriminator network"""
    def __init__(self, x1_dim=52, base_ch=64, img_height=36, img_width=250):
        super().__init__()
        
        # Compute spatial dimensions after downsampling
        # After 3 downsampling steps (2x): 36->18->9->5, 250->125->63->32
        down_height = img_height // 8
        down_width = img_width // 8
        
        # Condition encoder - produces features matching the downsampled size
        self.cond_encoder = nn.Sequential(
            nn.Linear(x1_dim, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, down_height * down_width),  # 5 * 32 = 160
        )
        
        # Image feature extractor
        self.conv_layers = nn.ModuleList([
            # Input: [batch, 1, 36, 250] (sound field)
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
        
        # Fusion of conditional features and image features
        self.fusion_layer = nn.Conv2d(base_ch*8 + 1, base_ch*8, kernel_size=1)
        
        # Final output layer - uses linear layer to output logits
        self.output_layer = nn.Sequential(
            nn.Conv2d(base_ch*8, 1, kernel_size=3, stride=1, padding=1),  # [5, 32]
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # Sigmoid removed because BCEWithLogitsLoss handles it internally
        )
        
    def forward(self, x1, sound_field):
        """
        Args:
            x1: Condition vector [batch, x1_dim]
            sound_field: Sound field [batch, height, width] or [batch, 1, height, width]
        Returns:
            Discriminator output logits [batch, 1]
        """
        if sound_field.dim() == 3:
            sound_field = sound_field.unsqueeze(1)
        
        batch_size = sound_field.shape[0]
        
        # Extract sound field features
        features = sound_field
        for layer in self.conv_layers:
            features = layer(features)
        
        # Process conditional information
        cond_features = self.cond_encoder(x1)
        # Reshape conditional features to spatial feature map [batch, 1, 5, 32]
        cond_map = cond_features.view(batch_size, 1, features.size(2), features.size(3))
        
        # Concatenate sound field features and conditional features
        combined = torch.cat([features, cond_map], dim=1)
        fused = self.fusion_layer(combined)
        
        # Final output logits
        output = self.output_layer(fused)
        return output


# ==================== Conditional GAN Model ====================
class cGAN2D(nn.Module):
    """Conditional GAN model"""
    def __init__(self, x1_dim=52, g_base_ch=64, d_base_ch=64, cond_dim=128, dropout=0.05):
        super().__init__()
        self.generator = Generator(x1_dim, g_base_ch, cond_dim, dropout)
        self.discriminator = ConditionalDiscriminator(x1_dim, d_base_ch)
    
    def forward(self, x1, x2):
        """Forward pass for generation stage"""
        return self.generator(x1, x2)


# ==================== Training Utilities ====================
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
    """Compute gradient penalty for WGAN-GP"""
    batch_size = real_data.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    # Interpolated samples
    interpolated = alpha * real_data + (1 - alpha) * fake_data
    interpolated.requires_grad_(True)
    
    # Compute discriminator output
    d_interpolated = discriminator(x1, interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    penalty = ((gradient_norm - 1) ** 2).mean()
    
    return penalty


# ==================== Main Training Function ====================
def main():
    # Configuration parameters
    out_dir = "./outputs_2d_cgan"
    epochs, batch_size = 120, 16
    lr_g, lr_d = 1e-4, 4e-4  # GANs typically use lower learning rates
    weight_decay = 1e-4
    patience, seed = 20, 42
    lambda_gp = 10  # Gradient penalty coefficient
    lambda_l1 = 100  # L1 loss coefficient
    
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load data
    input1 = np.load("shareddata/sf_input.npy", mmap_mode="r").astype(np.float32)
    target = np.load("shareddata/sf_res.npy", mmap_mode="r").astype(np.float32)
    assert len(input1) == len(target), "Data length mismatch"
    
    # Generate background field and standardize
    input2 = gen_background(target)
    x1_mean, x1_std = input1.mean(0), input1.std(0) + 1e-6
    t_mean, t_std = target.mean(), target.std() + 1e-6
    
    input1 = (input1 - x1_mean) / x1_std
    input2 = (input2 - t_mean) / t_std
    target = (target - t_mean) / t_std
    
    # Data split
    train_idx, val_idx, test_idx = split_indices_by_year(len(input1))
    print(f"Data split: training {len(train_idx)}, validation {len(val_idx)}, test {len(test_idx)}")
    
    # Data loaders
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
    
    # Initialize model
    model = cGAN2D().to(device)
    g_params, g_trainable = calculate_model_complexity(model.generator)
    d_params, d_trainable = calculate_model_complexity(model.discriminator)
    print(f"Generator parameters: {g_params:,} (trainable: {g_trainable:,})")
    print(f"Discriminator parameters: {d_params:,} (trainable: {d_trainable:,})")
    print(f"Total parameters: {g_params + d_params:,}")
    
    # Test dimensions
    with torch.no_grad():
        test_batch = next(iter(train_loader))
        x1_test, x2_test, y_test = [x.to(device) for x in test_batch]
        print(f"Input size test:")
        print(f"  x1: {x1_test.shape}")  # [batch, 52]
        print(f"  x2: {x2_test.shape}")  # [batch, 36, 250]
        print(f"  y: {y_test.shape}")   # [batch, 36, 250]
        
        # Test generator
        fake_y = model.generator(x1_test, x2_test)
        print(f"  Generator output: {fake_y.shape}")  # Should be [batch, 36, 250]
        
        # Test discriminator
        d_out_real = model.discriminator(x1_test, y_test.unsqueeze(1))
        d_out_fake = model.discriminator(x1_test, fake_y.unsqueeze(1))
        print(f"  Discriminator output (real): {d_out_real.shape}")  # Should be [batch, 1]
        print(f"  Discriminator output (fake): {d_out_fake.shape}")  # Should be [batch, 1]
    
    # Optimizers
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
    
    # Loss functions - using BCEWithLogitsLoss instead of BCELoss
    criterion_l1 = nn.L1Loss()
    criterion_mse = nn.MSELoss()
    criterion_bce = nn.BCEWithLogitsLoss()  # Using BCEWithLogitsLoss
    
    # Learning rate schedulers
    scheduler_g = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_g, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    scheduler_d = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer_d, mode="min", factor=0.5, patience=5, min_lr=1e-6
    )
    
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    
    # Training history dictionary
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
    
    # Training loop
    os.makedirs(out_dir, exist_ok=True)
    best_val, patience_counter = float("inf"), 0
    
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        train_g_loss, train_d_loss, train_l1_loss, train_rmse = 0, 0, 0, 0
        train_d_acc_real, train_d_acc_fake = 0, 0
        n_critic = 5  # Number of discriminator updates per generator update
        
        for batch_idx, (x1, x2, y) in enumerate(train_loader):
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)
            batch_size = x1.size(0)
            
            # Real and fake labels
            valid = torch.ones(batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(batch_size, 1, requires_grad=False).to(device)
            
            # ==================== Train Discriminator ====================
            model.discriminator.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                # Real samples
                real_pred = model.discriminator(x1, y.unsqueeze(1))
                d_real_loss = criterion_bce(real_pred, valid)
                
                # Generate fake samples
                fake_soundfield = model.generator(x1, x2)
                fake_pred = model.discriminator(x1, fake_soundfield.unsqueeze(1).detach())
                d_fake_loss = criterion_bce(fake_pred, fake)
                
                # Total discriminator loss
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # Compute discriminator accuracy
                train_d_acc_real += ((torch.sigmoid(real_pred) > 0.5).float().mean().item())
                train_d_acc_fake += ((torch.sigmoid(fake_pred) < 0.5).float().mean().item())
            
            scaler.scale(d_loss).backward()
            scaler.step(optimizer_d)
            scaler.update()
            
            # ==================== Train Generator ====================
            # Update generator every n_critic discriminator updates
            if batch_idx % n_critic == 0:
                model.generator.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                    # Generate fake samples
                    fake_soundfield = model.generator(x1, x2)
                    
                    # Adversarial loss
                    validity = model.discriminator(x1, fake_soundfield.unsqueeze(1))
                    g_adv_loss = criterion_bce(validity, valid)
                    
                    # L1 reconstruction loss
                    g_l1_loss = criterion_l1(fake_soundfield, y)
                    g_mse_loss = criterion_mse(fake_soundfield, y)
                    
                    # Total generator loss
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
        
        # Compute training averages
        train_d_loss_avg = train_d_loss / len(train_loader)
        train_g_loss_avg = train_g_loss / max(len(train_loader) / n_critic, 1)
        train_l1_loss_avg = train_l1_loss / max(len(train_loader) / n_critic, 1)
        train_rmse_avg = np.sqrt(train_rmse / max(len(train_loader) / n_critic, 1))
        train_d_acc_real_avg = train_d_acc_real / len(train_loader)
        train_d_acc_fake_avg = train_d_acc_fake / len(train_loader)
        
        # ==================== Validation ====================
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
                
                # Generator validation
                fake_soundfield = model.generator(x1, x2)
                
                # Adversarial loss
                validity = model.discriminator(x1, fake_soundfield.unsqueeze(1))
                g_adv_loss = criterion_bce(validity, valid)
                
                # Reconstruction loss
                g_l1_loss = criterion_l1(fake_soundfield, y)
                g_mse_loss = criterion_mse(fake_soundfield, y)
                
                g_loss = g_adv_loss + lambda_l1 * g_l1_loss
                
                # Discriminator validation
                real_pred = model.discriminator(x1, y.unsqueeze(1))
                fake_pred = model.discriminator(x1, fake_soundfield.unsqueeze(1).detach())
                d_real_loss = criterion_bce(real_pred, valid)
                d_fake_loss = criterion_bce(fake_pred, fake)
                d_loss = (d_real_loss + d_fake_loss) / 2
                
                # Compute discriminator accuracy
                val_d_acc_real += ((torch.sigmoid(real_pred) > 0.5).float().mean().item())
                val_d_acc_fake += ((torch.sigmoid(fake_pred) < 0.5).float().mean().item())
                
                # Real-scale propagation loss errors
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
        
        # Compute validation averages
        val_g_loss_avg = val_g_loss / len(val_loader)
        val_d_loss_avg = val_d_loss / len(val_loader)
        val_l1_loss_avg = val_l1_loss / len(val_loader)
        val_rmse_avg = np.sqrt(val_rmse / len(val_loader))
        val_l1_loss_real_avg = val_l1_loss_real / len(val_loader)
        val_rmse_real_avg = np.sqrt(val_rmse_real / len(val_loader))
        val_d_acc_real_avg = val_d_acc_real / len(val_loader)
        val_d_acc_fake_avg = val_d_acc_fake / len(val_loader)
        
        # Record training history
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
        
        # Update learning rates
        scheduler_g.step(val_l1_loss_avg)
        scheduler_d.step(val_d_loss_avg)
        
        # Print training information
        print(f"[Epoch {epoch:03d}] "
              f"lr_G={optimizer_g.param_groups[0]['lr']:.2e} "
              f"lr_D={optimizer_d.param_groups[0]['lr']:.2e}")
        print(f"  Real propagation loss scale: L1={val_l1_loss_real_avg:.3f}, RMSE={val_rmse_real_avg:.3f}")
        print(f"  Time: {time.time()-t0:.1f}s")
        
        # Save model
        torch.save({
            "epoch": epoch, "best_val": best_val,
            "generator": model.generator.state_dict(),
            "discriminator": model.discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(),
            "optimizer_d": optimizer_d.state_dict()
        }, os.path.join(out_dir, "last.pt"))
        
        # Use L1 loss as early stopping criterion
        if val_l1_loss_avg < best_val - 1e-6:
            best_val = val_l1_loss_avg
            torch.save(model.generator.state_dict(), os.path.join(out_dir, "best_generator.pt"))
            torch.save(model.discriminator.state_dict(), os.path.join(out_dir, "best_discriminator.pt"))
            print(f"✅ New best model: val_l1_loss={best_val:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered (patience={patience})")
                break
    
    # Save training history
    history_path = os.path.join(out_dir, "training_history.npz")
    np.savez(history_path, **training_history)
    print(f"Training history saved to: {history_path}")
    
    
    # ==================== Testing ====================
    print("\n" + "="*50)
    print("Testing Phase")
    print("="*50)
    
    # Load best generator
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
    
    print(f"[TEST] Standardized scale: L1={test_l1_loss_avg:.5f}, RMSE={test_rmse_avg:.5f}")
    print(f"[TEST] Real propagation loss scale: L1={test_l1_loss_real_avg:.5f}, RMSE={test_rmse_real_avg:.5f}")
    
    # Save test results
    test_results = {
        'test_l1_loss': test_l1_loss_avg,
        'test_rmse': test_rmse_avg,
        'test_l1_loss_real': test_l1_loss_real_avg,
        'test_rmse_real': test_rmse_real_avg
    }
    
    training_history['test_results'] = test_results
    np.savez(history_path, **training_history)
    
    
    print("GAN training completed")


if __name__ == "__main__":
    main()
