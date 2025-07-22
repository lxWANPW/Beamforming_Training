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
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math
from tqdm import tqdm
import warnings

class BeamformingDataset(Dataset):
    """修复版天线阵列信道数据集"""
    def __init__(self, channel_folder, angle_folder, transform=None):
        self.channel_folder = channel_folder
        self.angle_folder = angle_folder
        self.transform = transform
        
        # 获取信道数据文件列表并排序
        self.channel_files = sorted(glob.glob(os.path.join(channel_folder, '*.mat')))
        print(f"找到信道数据文件: {len(self.channel_files)} 个")
        
        # 加载角度数据（所有样本的角度信息在一个文件中）
        angle_files = glob.glob(os.path.join(angle_folder, '*.mat'))
        if len(angle_files) > 0:
            try:
                # 尝试使用 h5py 读取
                with h5py.File(angle_files[0], 'r') as f:
                    all_theta = np.array(f['all_theta'])  # (50, 1000) or (1000, 50)
                    all_phi = np.array(f['all_phi'])      # (50, 1000) or (1000, 50)
                    
                    # 检查维度并调整
                    if all_theta.shape[0] == 50 and all_theta.shape[1] == 1000:
                        self.theta_array = all_theta.T  # (1000, 50)
                        self.phi_array = all_phi.T      # (1000, 50)
                    else:
                        self.theta_array = all_theta    # 假设已经是(1000, 50)
                        self.phi_array = all_phi
                    
                    self.num_samples = self.theta_array.shape[0]
                    self.num_time_samples = self.theta_array.shape[1]
                    
                    print(f"角度数据形状: theta={self.theta_array.shape}, phi={self.phi_array.shape}")
                    
            except Exception as e:
                print(f"h5py 读取失败: {e}")
                # 使用 scipy.io 读取
                try:
                    angle_data = sio.loadmat(angle_files[0])
                    # 检查可能的键名
                    possible_keys = ['all_theta', 'theta_array', 'theta']
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
                    print(f"角度文件内容: {list(angle_data.keys()) if 'angle_data' in locals() else 'Unknown'}")
                    raise e2
        
        # 确保数据一致性，使用较小的数量
        self.actual_num_samples = min(len(self.channel_files), self.num_samples)
        print(f"实际使用的样本数量: {self.actual_num_samples}")
        
    def __len__(self):
        return self.actual_num_samples
    
    def __getitem__(self, idx):
        if idx >= self.actual_num_samples:
            raise IndexError(f"索引 {idx} 超出范围 {self.actual_num_samples}")
            
        # 加载信道数据
        try:
            # 尝试使用 h5py 读取
            with h5py.File(self.channel_files[idx], 'r') as f:
                H_real = np.array(f['H_real'])  # (50, 256, 256)
                H_imag = np.array(f['H_imag'])  # (50, 256, 256)
                
                # 转置以匹配预期格式 [256, 256, 50]
                H_real = H_real.transpose(1, 2, 0)  # (256, 256, 50)
                H_imag = H_imag.transpose(1, 2, 0)  # (256, 256, 50)
                
        except Exception as e:
            # 使用 scipy.io 读取
            try:
                channel_data = sio.loadmat(self.channel_files[idx])
                H_real = channel_data['H_real']  # [256, 256, 50]
                H_imag = channel_data['H_imag']
            except Exception as e2:
                print(f"无法读取信道数据文件 {self.channel_files[idx]}: {e2}")
                raise e2
        
        # 组合实部和虚部
        H_complex = H_real + 1j * H_imag
        
        # 改进的数据归一化（避免极值）
        H_real_std = np.std(H_real)
        H_imag_std = np.std(H_imag)
        
        if H_real_std > 1e-8:
            H_real_norm = (H_real - np.mean(H_real)) / H_real_std
        else:
            H_real_norm = H_real
            
        if H_imag_std > 1e-8:
            H_imag_norm = (H_imag - np.mean(H_imag)) / H_imag_std
        else:
            H_imag_norm = H_imag
        
        # 裁剪极值以避免梯度爆炸
        H_real_norm = np.clip(H_real_norm, -10, 10)
        H_imag_norm = np.clip(H_imag_norm, -10, 10)
        
        # 转换为PyTorch tensor
        H_input = np.stack([H_real_norm, H_imag_norm], axis=0)  # [2, 256, 256, 50]
        H_input = torch.FloatTensor(H_input)
        
        # 改进的角度标签归一化
        theta_rad = self.theta_array[idx] * np.pi / 180  # 转换为弧度
        phi_rad = self.phi_array[idx] * np.pi / 180
        
        # 使用更稳定的归一化方式
        theta_norm = theta_rad / (np.pi / 2)  # 归一化到[-2, 2]范围，然后裁剪
        phi_norm = phi_rad / np.pi  # 归一化到[-1, 1]
        
        theta_norm = np.clip(theta_norm, -1, 1)
        phi_norm = np.clip(phi_norm, -1, 1)
        
        theta_label = torch.FloatTensor(theta_norm)  # [50]
        phi_label = torch.FloatTensor(phi_norm)      # [50]
        
        return H_input, theta_label, phi_label, H_complex

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerBeamformingNet(nn.Module):
    """基于Transformer的波束成形网络，提升角度预测精度"""
    def __init__(self, input_dim=256*256*2, d_model=768, nhead=12, num_layers=8, dropout=0.15):
        super(TransformerBeamformingNet, self).__init__()
        
        self.d_model = d_model
        
        # 高效的特征提取器（使用卷积降维）
        self.feature_extractor = nn.Sequential(
            # 先用卷积降维减少计算量
            nn.Conv1d(input_dim, d_model * 2, kernel_size=1),
            nn.BatchNorm1d(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, max_len=100)
        
        # 更强的Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,  # 更大的FFN
            dropout=dropout,
            activation='gelu',  # 使用GELU激活
            batch_first=True,
            norm_first=True  # Pre-LN架构，更稳定
        )
        self.transformer_encoder = TransformerEncoder(
            encoder_layers, 
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        
        # 多头注意力聚合（替代简单的最后时间步）
        self.attention_pooling = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # 改进的角度预测头（增加dropout防止过拟合）
        self.angle_predictor = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 1.5),  # 更高的dropout
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout * 1.2),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 4, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),  # 输出层前轻微dropout
            nn.Linear(64, 2)  # theta, phi
        )
        
        # 可学习的查询向量用于注意力聚合
        self.query_vector = nn.Parameter(torch.randn(1, 1, d_model))
        
        # 改进的权重初始化
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # 使用更好的初始化
            nn.init.xavier_normal_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)
        elif isinstance(module, nn.Conv1d):
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                    
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        
        # 重塑输入 [batch, time_steps, features]
        x = x.permute(0, 4, 1, 2, 3)  # [batch, time_steps, channels, height, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch, time_steps, features]
        
        # 特征提取（使用1D卷积）
        x = x.permute(0, 2, 1)  # [batch, features, time_steps]
        x = self.feature_extractor(x)  # [batch, d_model, time_steps]
        x = x.permute(0, 2, 1)  # [batch, time_steps, d_model]
        
        # 输入缩放和位置编码
        x = x * math.sqrt(self.d_model)
        x = x.transpose(0, 1)  # [time_steps, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, time_steps, d_model]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # [batch, time_steps, d_model]
        
        # 注意力聚合（更好的时序信息整合）
        query = self.query_vector.expand(batch_size, -1, -1)  # [batch, 1, d_model]
        aggregated, attention_weights = self.attention_pooling(
            query, encoded, encoded
        )  # [batch, 1, d_model]
        aggregated = aggregated.squeeze(1)  # [batch, d_model]
        
        # 角度预测
        angles = self.angle_predictor(aggregated)  # [batch, 2]
        
        return angles

def improved_loss_function(predictions, targets, loss_type='focal_huber'):
    """改进的损失函数，降低角度预测误差"""
    pred_angles = predictions  # [batch, 2] (theta, phi)
    target_theta, target_phi = targets
    
    # 使用最后一个时间步的目标
    target_theta_last = target_theta[:, -1]  # [batch]
    target_phi_last = target_phi[:, -1]      # [batch]
    
    if loss_type == 'focal_huber':
        # 组合Focal Loss和Huber Loss的思想
        # 计算基础误差
        theta_error = torch.abs(pred_angles[:, 0] - target_theta_last)
        phi_error = torch.abs(pred_angles[:, 1] - target_phi_last)
        
        # Focal权重：对难样本加权
        theta_focal_weight = torch.pow(theta_error + 1e-8, 0.5)  # 平方根权重
        phi_focal_weight = torch.pow(phi_error + 1e-8, 0.5)
        
        # Huber损失
        theta_loss = F.huber_loss(pred_angles[:, 0], target_theta_last, delta=0.5, reduction='none')
        phi_loss = F.huber_loss(pred_angles[:, 1], target_phi_last, delta=0.5, reduction='none')
        
        # 应用focal权重
        theta_loss = (theta_focal_weight * theta_loss).mean()
        phi_loss = (phi_focal_weight * phi_loss).mean()
        
    elif loss_type == 'adaptive_mse':
        # 自适应MSE损失
        theta_loss = F.mse_loss(pred_angles[:, 0], target_theta_last)
        phi_loss = F.mse_loss(pred_angles[:, 1], target_phi_last)
        
        # 动态权重平衡
        theta_weight = 1.0 / (theta_loss.detach() + 1e-8)
        phi_weight = 1.0 / (phi_loss.detach() + 1e-8)
        
        total_weight = theta_weight + phi_weight
        theta_weight = theta_weight / total_weight
        phi_weight = phi_weight / total_weight
        
        theta_loss = theta_weight * theta_loss
        phi_loss = phi_weight * phi_loss
        
    else:
        # 标准Huber损失
        theta_loss = F.huber_loss(pred_angles[:, 0], target_theta_last, delta=1.0)
        phi_loss = F.huber_loss(pred_angles[:, 1], target_phi_last, delta=1.0)
    
    total_loss = theta_loss + phi_loss
    
    # 添加正则化项
    l2_reg = 0.0
    for param in pred_angles.requires_grad_(True):
        if param.requires_grad:
            l2_reg += torch.norm(param)
    total_loss += 1e-6 * l2_reg
    
    # 确保损失值在合理范围内
    total_loss = torch.clamp(total_loss, 0, 50)
    
    return total_loss, theta_loss, phi_loss

def train_model(model, train_loader, val_loader, num_epochs=150, lr=3e-4, device='cuda'):
    """改进的训练函数，使用更高学习率和优化策略"""
    model.to(device)
    
    # 使用更激进的优化器设置
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=2e-4,  # 增加正则化防止过拟合
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # 改进的学习率调度策略
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,  # 10%时间warm-up
        div_factor=10,  # 初始lr = max_lr/10
        final_div_factor=100,  # 最终lr = max_lr/100
        anneal_strategy='cos'
    )
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 40  # 增加patience，在低损失时更宽容
    min_improvement = 1e-5  # 最小改进阈值，避免微小波动触发早停
    
    print(f"开始训练，最大学习率: {lr}")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_theta_loss = 0
        train_phi_loss = 0
        num_batches = 0
        
        train_progress = tqdm(train_loader, desc=f"训练 Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_progress):
            try:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                
                # 检查输入数据的有效性
                if torch.isnan(H_input).any() or torch.isinf(H_input).any():
                    print(f"警告: 输入数据包含NaN或Inf，跳过批次 {batch_idx}")
                    continue
                
                optimizer.zero_grad()
                
                # 前向传播
                predictions = model(H_input)
                
                # 计算改进的损失
                loss, theta_loss, phi_loss = improved_loss_function(
                    predictions, (theta_label, phi_label), loss_type='focal_huber'
                )
                
                # 检查损失的有效性
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 损失为NaN或Inf，跳过批次 {batch_idx}")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（适应更高学习率）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                
                optimizer.step()
                scheduler.step()  # OneCycleLR需要每个step调用
                
                train_loss_epoch += loss.item()
                train_theta_loss += theta_loss.item()
                train_phi_loss += phi_loss.item()
                num_batches += 1
                
                # 更新进度条
                current_lr = optimizer.param_groups[0]['lr']
                train_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'θ': f'{theta_loss.item():.4f}',
                    'φ': f'{phi_loss.item():.4f}',
                    'LR': f'{current_lr:.2e}'
                })
                
            except Exception as e:
                print(f"训练批次 {batch_idx} 出错: {e}")
                continue
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_theta_loss = 0
        val_phi_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            val_progress = tqdm(val_loader, desc="验证", leave=False)
            for H_input, theta_label, phi_label, H_complex in val_progress:
                try:
                    H_input = H_input.to(device)
                    theta_label = theta_label.to(device)
                    phi_label = phi_label.to(device)
                    
                    predictions = model(H_input)
                    loss, theta_loss, phi_loss = improved_loss_function(
                        predictions, (theta_label, phi_label), loss_type='focal_huber'
                    )
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss_epoch += loss.item()
                        val_theta_loss += theta_loss.item()
                        val_phi_loss += phi_loss.item()
                        val_batches += 1
                        
                except Exception as e:
                    continue
        
        # 计算平均损失
        if num_batches > 0:
            train_loss_avg = train_loss_epoch / num_batches
            train_theta_avg = train_theta_loss / num_batches
            train_phi_avg = train_phi_loss / num_batches
        else:
            train_loss_avg = float('inf')
            train_theta_avg = float('inf')
            train_phi_avg = float('inf')
            
        if val_batches > 0:
            val_loss_avg = val_loss_epoch / val_batches
            val_theta_avg = val_theta_loss / val_batches
            val_phi_avg = val_phi_loss / val_batches
        else:
            val_loss_avg = float('inf')
            val_theta_avg = float('inf')
            val_phi_avg = float('inf')
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # 在极低损失时使用更宽松的早停策略
        current_early_stop_patience = early_stop_patience
        if val_loss_avg < 0.001:  # 当验证损失很低时
            current_early_stop_patience = 60  # 更大的patience
        
        # 打印epoch结果 - 增加更多监控信息
        current_lr = optimizer.param_groups[0]['lr']
        train_val_gap = val_loss_avg - train_loss_avg
        print(f"Epoch {epoch+1:2d}: Train={train_loss_avg:.6f} (θ:{train_theta_avg:.6f}, φ:{train_phi_avg:.6f}), "
              f"Val={val_loss_avg:.6f} (θ:{val_theta_avg:.6f}, φ:{val_phi_avg:.6f}), "
              f"Gap={train_val_gap:.6f}, LR={current_lr:.2e}, Patience={patience_counter}/{current_early_stop_patience}")
        
        # 早停和模型保存 - 改进策略
        improvement = best_val_loss - val_loss_avg
        if improvement > min_improvement:  # 只有明显改进才重置计数器
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), 'best_beamforming_model_fixed.pth')
            print(f"✓ 保存最佳模型 (Val Loss: {val_loss_avg:.6f}, 改进: {improvement:.6f})")
        else:
            patience_counter += 1
            
        if patience_counter >= current_early_stop_patience:
            print(f"早停触发！在epoch {epoch+1}停止训练 (patience: {patience_counter}/{current_early_stop_patience})")
            break
    
    return train_losses, val_losses

def test_model(model, test_loader, device='cuda'):
    """测试模型性能"""
    model.to(device)
    model.eval()
    
    all_predictions = []
    angle_errors = []
    
    with torch.no_grad():
        test_progress = tqdm(test_loader, desc="测试进度")
        
        for H_input, theta_label, phi_label, H_complex in test_progress:
            try:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                
                # 预测
                predictions = model(H_input)
                
                # 转换为度数（反归一化）
                pred_angles = predictions.cpu().numpy()
                pred_theta_norm = pred_angles[:, 0]  # [-1, 1]
                pred_phi_norm = pred_angles[:, 1]    # [-1, 1]
                
                # 反归一化到度数（修正范围）
                pred_theta_deg = pred_theta_norm * 90  # [-90, 90] 对应 [-1, 1]
                pred_phi_deg = pred_phi_norm * 180     # [-180, 180] 对应 [-1, 1]
                
                # 目标角度（使用最后时间步）
                target_theta_norm = theta_label[:, -1].cpu().numpy()
                target_phi_norm = phi_label[:, -1].cpu().numpy()
                
                target_theta_deg = target_theta_norm * 90
                target_phi_deg = target_phi_norm * 180
                
                # 计算角度误差
                theta_error = np.abs(pred_theta_deg - target_theta_deg)
                phi_error = np.abs(pred_phi_deg - target_phi_deg)
                
                # 处理phi角度的周期性
                phi_error = np.minimum(phi_error, 360 - phi_error)
                
                total_error = theta_error + phi_error
                angle_errors.extend(total_error)
                
                for i in range(len(pred_angles)):
                    all_predictions.append({
                        'predicted_theta': pred_theta_deg[i],
                        'predicted_phi': pred_phi_deg[i],
                        'target_theta': target_theta_deg[i],
                        'target_phi': target_phi_deg[i],
                        'theta_error': theta_error[i],
                        'phi_error': phi_error[i],
                        'total_error': total_error[i]
                    })
                    
            except Exception as e:
                print(f"测试批次错误: {e}")
                continue
    
    # 计算统计指标
    if len(angle_errors) > 0:
        mean_error = np.mean(angle_errors)
        std_error = np.std(angle_errors)
        median_error = np.median(angle_errors)
        
        print(f"\n📊 测试结果:")
        print(f"平均角度误差: {mean_error:.2f}°")
        print(f"角度误差标准差: {std_error:.2f}°")
        print(f"角度误差中位数: {median_error:.2f}°")
        print(f"有效预测数量: {len(all_predictions)}")
        
        if len(all_predictions) > 0:
            theta_errors = [p['theta_error'] for p in all_predictions]
            phi_errors = [p['phi_error'] for p in all_predictions]
            print(f"θ误差: {np.mean(theta_errors):.2f}° ± {np.std(theta_errors):.2f}°")
            print(f"φ误差: {np.mean(phi_errors):.2f}° ± {np.std(phi_errors):.2f}°")
    else:
        print("没有有效的测试结果")
    
    return all_predictions

def main():
    """主函数"""
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径
    channel_folder = "samples_data/channel_data_opt_20250702_143327"
    angle_folder = "samples_data/angle_data_opt_20250702_143327"
    
    print("正在加载数据集...")
    
    # 创建数据集
    dataset = BeamformingDataset(channel_folder, angle_folder)
    
    # 按要求划分数据集：800训练，200测试
    total_samples = len(dataset)
    train_size = 800
    test_size = total_samples - train_size
    
    indices = list(range(total_samples))
    np.random.seed(42)  # 设置随机种子确保结果可重现
    np.random.shuffle(indices)
    
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    # 从训练集中划分验证集 - 增加验证集大小提高稳定性
    val_size = int(train_size * 0.25)  # 25%作为验证集，提高稳定性
    train_final_indices = train_indices[val_size:]
    val_indices = train_indices[:val_size]
    
    print(f"数据集划分:")
    print(f"  训练集: {len(train_final_indices)} 样本")
    print(f"  验证集: {len(val_indices)} 样本") 
    print(f"  测试集: {len(test_indices)} 样本")
    
    # 创建数据子集
    train_dataset = torch.utils.data.Subset(dataset, train_final_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    # 创建数据加载器（提高批次大小利用Transformer并行性）
    batch_size = 6  # 适度增加批次大小
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    # 创建改进的Transformer模型
    model = TransformerBeamformingNet(
        input_dim=256*256*2,
        d_model=512,  # 适度减小模型容量，防止过拟合
        nhead=8,      # 减少注意力头数
        num_layers=6, # 减少层数
        dropout=0.15  # 增加dropout
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("\n🚀 开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=150,  # 增加训练轮数
        lr=3e-4,         # 提高学习率
        device=device
    )
    
    # 加载最佳模型
    if os.path.exists('best_beamforming_model_fixed.pth'):
        model.load_state_dict(torch.load('best_beamforming_model_fixed.pth'))
        print("✓ 加载最佳模型完成")
    
    # 测试模型
    print("\n🧪 开始测试...")
    predictions = test_model(model, test_loader, device=device)
    
    # 绘制结果
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 训练损失曲线
    axes[0, 0].plot(train_losses, label='Training Loss', alpha=0.8)
    axes[0, 0].plot(val_losses, label='Validation Loss', alpha=0.8)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Progress')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_yscale('log')
    
    # 角度预测散点图
    if len(predictions) > 0:
        pred_theta = [p['predicted_theta'] for p in predictions]
        target_theta = [p['target_theta'] for p in predictions]
        pred_phi = [p['predicted_phi'] for p in predictions]
        target_phi = [p['target_phi'] for p in predictions]
        
        # θ角度预测
        axes[0, 1].scatter(target_theta, pred_theta, alpha=0.6, s=20)
        min_theta = min(min(target_theta), min(pred_theta))
        max_theta = max(max(target_theta), max(pred_theta))
        axes[0, 1].plot([min_theta, max_theta], [min_theta, max_theta], 'r--', label='Perfect')
        axes[0, 1].set_xlabel('True θ (degrees)')
        axes[0, 1].set_ylabel('Predicted θ (degrees)')
        axes[0, 1].set_title('θ Angle Prediction')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # φ角度预测
        axes[1, 0].scatter(target_phi, pred_phi, alpha=0.6, s=20)
        min_phi = min(min(target_phi), min(pred_phi))
        max_phi = max(max(target_phi), max(pred_phi))
        axes[1, 0].plot([min_phi, max_phi], [min_phi, max_phi], 'r--', label='Perfect')
        axes[1, 0].set_xlabel('True φ (degrees)')
        axes[1, 0].set_ylabel('Predicted φ (degrees)')
        axes[1, 0].set_title('φ Angle Prediction')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 误差分布
        errors = [p['total_error'] for p in predictions]
        axes[1, 1].hist(errors, bins=30, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(np.mean(errors), color='red', linestyle='--', label=f'Mean: {np.mean(errors):.1f}°')
        axes[1, 1].set_xlabel('Total Angle Error (degrees)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results_fixed.png', dpi=300, bbox_inches='tight')
    print(f"\n📈 训练结果图表已保存为 'training_results_fixed.png'")
    
    # 打印训练总结
    print(f"\n📋 训练总结:")
    if train_losses:
        print(f"• 最终训练损失: {train_losses[-1]:.4f}")
        print(f"• 最终验证损失: {val_losses[-1]:.4f}")
        print(f"• 总训练轮数: {len(train_losses)}")
        print(f"• 损失改善: {((train_losses[0] - train_losses[-1]) / train_losses[0] * 100):.1f}%")
    
    return model, predictions

if __name__ == "__main__":
    main() 