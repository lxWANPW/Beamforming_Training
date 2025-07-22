import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import h5py
import os
import glob
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from tqdm import tqdm
import warnings
from torch.cuda.amp import GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time

# 导入之前的数据集类
class BeamformingDataset(Dataset):
    """优化版天线阵列信道数据集"""
    def __init__(self, channel_folder, angle_folder, transform=None, use_augmentation=True):
        self.channel_folder = channel_folder
        self.angle_folder = angle_folder
        self.transform = transform
        self.use_augmentation = use_augmentation
        
        # 获取信道数据文件列表并排序
        self.channel_files = sorted(glob.glob(os.path.join(channel_folder, '*.mat')))
        print(f"找到信道数据文件: {len(self.channel_files)} 个")
        
        # 加载角度数据
        angle_files = glob.glob(os.path.join(angle_folder, '*.mat'))
        if len(angle_files) > 0:
            try:
                with h5py.File(angle_files[0], 'r') as f:
                    all_theta = np.array(f['all_theta'])
                    all_phi = np.array(f['all_phi'])
                    
                    if all_theta.shape[0] == 50 and all_theta.shape[1] == 1000:
                        self.theta_array = all_theta.T
                        self.phi_array = all_phi.T
                    else:
                        self.theta_array = all_theta
                        self.phi_array = all_phi
                    
                    self.num_samples = self.theta_array.shape[0]
                    self.num_time_samples = self.theta_array.shape[1]
                    
            except Exception as e:
                print(f"h5py 读取失败: {e}")
                try:
                    angle_data = sio.loadmat(angle_files[0])
                    theta_key = None
                    phi_key = None
                    
                    for key in angle_data.keys():
                        if 'theta' in key.lower():
                            theta_key = key
                        if 'phi' in key.lower():
                            phi_key = key
                    
                    if theta_key and phi_key:
                        all_theta = angle_data[theta_key]
                        all_phi = angle_data[phi_key]
                        
                        if all_theta.shape[0] == 50 and all_theta.shape[1] == 1000:
                            self.theta_array = all_theta.T
                            self.phi_array = all_phi.T
                        else:
                            self.theta_array = all_theta
                            self.phi_array = all_phi
                            
                        self.num_samples = self.theta_array.shape[0]
                        self.num_time_samples = self.theta_array.shape[1]
                    else:
                        raise ValueError("未找到theta和phi数据")
                        
                except Exception as e2:
                    print(f"scipy.io 读取也失败: {e2}")
                    raise e2
        
        self.actual_num_samples = min(len(self.channel_files), self.num_samples)
        print(f"实际使用的样本数量: {self.actual_num_samples}")
        
    def __len__(self):
        return self.actual_num_samples
    
    def _add_data_augmentation(self, H_real, H_imag, theta_rad, phi_rad):
        if not self.use_augmentation:
            return H_real, H_imag, theta_rad, phi_rad
            
        noise_std = 0.005
        H_real = H_real + np.random.normal(0, noise_std, H_real.shape)
        H_imag = H_imag + np.random.normal(0, noise_std, H_imag.shape)
        
        angle_noise_std = 0.01
        theta_rad = theta_rad + np.random.normal(0, angle_noise_std, theta_rad.shape)
        phi_rad = phi_rad + np.random.normal(0, angle_noise_std, phi_rad.shape)
        
        return H_real, H_imag, theta_rad, phi_rad
    
    def __getitem__(self, idx):
        if idx >= self.actual_num_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.actual_num_samples}")
            
        try:
            with h5py.File(self.channel_files[idx], 'r') as f:
                H_real = np.array(f['H_real'])
                H_imag = np.array(f['H_imag'])
                H_real = H_real.transpose(1, 2, 0)
                H_imag = H_imag.transpose(1, 2, 0)
        except Exception:
            try:
                channel_data = sio.loadmat(self.channel_files[idx])
                H_real = channel_data['H_real']
                H_imag = channel_data['H_imag']
            except Exception as e2:
                print(f"无法读取信道数据文件 {self.channel_files[idx]}: {e2}")
                raise e2
        
        theta_rad = self.theta_array[idx] * np.pi / 180
        phi_rad = self.phi_array[idx] * np.pi / 180
        
        H_real, H_imag, theta_rad, phi_rad = self._add_data_augmentation(
            H_real, H_imag, theta_rad, phi_rad
        )
        
        H_complex = H_real + 1j * H_imag
        
        H_real_norm = H_real / (np.std(H_real) + 1e-8)
        H_imag_norm = H_imag / (np.std(H_imag) + 1e-8)
        
        H_real_norm = np.clip(H_real_norm, -6, 6)
        H_imag_norm = np.clip(H_imag_norm, -6, 6)
        
        H_input = np.stack([H_real_norm, H_imag_norm], axis=0)
        H_input = torch.FloatTensor(H_input)
        
        theta_norm = np.clip(theta_rad / (np.pi / 2), -1, 1)
        phi_norm = np.clip(phi_rad / np.pi, -1, 1)
        
        theta_label = torch.FloatTensor(theta_norm)
        phi_label = torch.FloatTensor(phi_norm)
        
        return H_input, theta_label, phi_label, H_complex

class SimpleBeamformingNet(nn.Module):
    """简化的波束成形网络，避免DDP问题"""
    def __init__(self, input_dim=256*256*2, d_model=256, nhead=4, num_layers=3, dropout=0.1):
        super(SimpleBeamformingNet, self).__init__()
        
        self.d_model = d_model
        
        # 简单的特征提取
        self.feature_proj = nn.Sequential(
            nn.Linear(input_dim, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.LayerNorm(d_model)
        )
        
        # 位置编码
        pe = torch.zeros(100, d_model)
        position = torch.arange(0, 100, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
        # Transformer编码器
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers)
        
        # 预测头
        self.predictor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 2)
        )
        
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        
        # 重塑输入
        x = x.permute(0, 4, 1, 2, 3)
        x = x.reshape(batch_size, time_steps, -1)
        
        # 特征投影
        x = self.feature_proj(x)
        
        # 位置编码
        x = x + self.pe[:time_steps, :].unsqueeze(0)
        
        # Transformer编码
        x = self.transformer(x)
        
        # 全局平均池化
        x = x.mean(dim=1)
        
        # 预测
        angles = self.predictor(x)
        
        return angles

class SimpleLoss(nn.Module):
    """简化的损失函数"""
    def __init__(self):
        super(SimpleLoss, self).__init__()
        
    def forward(self, predictions, targets):
        target_theta, target_phi = targets
        
        target_theta_last = target_theta[:, -1]
        target_phi_last = target_phi[:, -1]
        
        theta_loss = F.mse_loss(predictions[:, 0], target_theta_last)
        phi_loss = F.mse_loss(predictions[:, 1], target_phi_last)
        
        total_loss = theta_loss + phi_loss
        
        return total_loss, theta_loss, phi_loss

def setup_distributed():
    """设置分布式环境"""
    if 'RANK' in os.environ:
        rank = int(os.environ['RANK'])
        local_rank = int(os.environ['LOCAL_RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        return world_size, rank, local_rank
    else:
        return 1, 0, 0

def train_simple_model(model, train_loader, val_loader, num_epochs=100, lr=1e-3, 
                      device='cuda', world_size=1, rank=0, local_rank=0):
    """简化的训练函数"""
    
    model.to(device)
    
    # DDP包装
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[local_rank], 
            output_device=local_rank,
            find_unused_parameters=False,  # 强制所有参数参与
            static_graph=True              # 静态图优化
        )
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    
    # 损失函数
    criterion = SimpleLoss()
    
    # 混合精度
    scaler = GradScaler()
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    is_main_process = rank == 0
    
    for epoch in range(num_epochs):
        # 设置sampler epoch
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        # 训练
        model.train()
        train_loss_epoch = 0
        num_batches = 0
        
        if is_main_process:
            train_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        else:
            train_progress = train_loader
        
        for H_input, theta_label, phi_label, H_complex in train_progress:
            H_input = H_input.to(device, non_blocking=True)
            theta_label = theta_label.to(device, non_blocking=True)
            phi_label = phi_label.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                predictions = model(H_input)
                loss, theta_loss, phi_loss = criterion(predictions, (theta_label, phi_label))
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_epoch += loss.item()
            num_batches += 1
            
            if is_main_process and isinstance(train_progress, tqdm):
                train_progress.set_postfix({'Loss': f'{loss.item():.5f}'})
        
        scheduler.step()
        
        # 验证
        model.eval()
        val_loss_epoch = 0
        val_batches = 0
        
        with torch.no_grad():
            for H_input, theta_label, phi_label, H_complex in val_loader:
                H_input = H_input.to(device, non_blocking=True)
                theta_label = theta_label.to(device, non_blocking=True)
                phi_label = phi_label.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    predictions = model(H_input)
                    loss, _, _ = criterion(predictions, (theta_label, phi_label))
                
                val_loss_epoch += loss.item()
                val_batches += 1
        
        # 计算平均损失
        train_loss_avg = train_loss_epoch / max(num_batches, 1)
        val_loss_avg = val_loss_epoch / max(val_batches, 1)
        
        # 同步损失（如果是分布式）
        if world_size > 1:
            train_tensor = torch.tensor(train_loss_avg, device=device)
            val_tensor = torch.tensor(val_loss_avg, device=device)
            
            dist.all_reduce(train_tensor, op=dist.ReduceOp.AVG)
            dist.all_reduce(val_tensor, op=dist.ReduceOp.AVG)
            
            train_loss_avg = train_tensor.item()
            val_loss_avg = val_tensor.item()
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        if is_main_process:
            print(f"Epoch {epoch+1:3d}: Train={train_loss_avg:.6f}, Val={val_loss_avg:.6f}")
        
        # 保存最佳模型
        if val_loss_avg < best_val_loss and is_main_process:
            best_val_loss = val_loss_avg
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), 'best_simple_model.pth')
            if is_main_process:
                print(f"✓ 保存最佳模型 (Val Loss: {val_loss_avg:.6f})")
    
    return train_losses, val_losses

def main():
    """主函数"""
    warnings.filterwarnings('ignore')
    
    # 设置环境
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # 分布式设置
    world_size, rank, local_rank = setup_distributed()
    device = f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu'
    
    is_main_process = rank == 0
    
    if is_main_process:
        print("=== 简化DDP训练 ===")
        print(f"World size: {world_size}, Rank: {rank}, Device: {device}")
    
    # 数据路径
    channel_folder = "samples_data/channel_data_opt_20250702_143327"
    angle_folder = "samples_data/angle_data_opt_20250702_143327"
    
    # 加载数据
    if is_main_process:
        print("加载数据集...")
    
    dataset = BeamformingDataset(channel_folder, angle_folder, use_augmentation=True)
    
    # 数据划分
    total_samples = len(dataset)
    train_size = 800
    
    indices = list(range(total_samples))
    np.random.seed(42)
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    val_size = int(train_size * 0.20)
    train_final_indices = train_indices[val_size:]
    val_indices = train_indices[:val_size]
    
    if is_main_process:
        print(f"训练集: {len(train_final_indices)}, 验证集: {len(val_indices)}, 测试集: {len(test_indices)}")
    
    # 创建数据集
    train_dataset = torch.utils.data.Subset(dataset, train_final_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    
    # 创建数据加载器
    batch_size = 4
    
    train_sampler = None
    val_sampler = None
    
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=(train_sampler is None), 
        sampler=train_sampler,
        num_workers=2, 
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        sampler=val_sampler,
        num_workers=2, 
        pin_memory=True
    )
    
    # 创建简化模型
    model = SimpleBeamformingNet(
        input_dim=256*256*2,
        d_model=256,
        nhead=4,
        num_layers=3,
        dropout=0.1
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    if is_main_process:
        print(f"模型参数: {total_params:,} ({total_params/1e6:.1f}M)")
    
    # 训练
    if is_main_process:
        print("开始训练...")
    
    train_losses, val_losses = train_simple_model(
        model, train_loader, val_loader,
        num_epochs=100,
        lr=1e-3,
        device=device,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    
    # 清理分布式
    if world_size > 1:
        dist.destroy_process_group()
    
    if is_main_process:
        print("训练完成！")
        
        # 绘制结果
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Simple Model Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig('simple_training_results.png', dpi=300, bbox_inches='tight')
        print("结果图已保存为 simple_training_results.png")

if __name__ == "__main__":
    main()