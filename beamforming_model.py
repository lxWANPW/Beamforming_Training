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
    """天线阵列信道数据集"""
    def __init__(self, channel_folder, angle_folder, transform=None):
        self.channel_folder = channel_folder
        self.angle_folder = angle_folder
        self.transform = transform
        
        # 获取信道数据文件列表
        self.channel_files = sorted(glob.glob(os.path.join(channel_folder, '*.mat')))
        
        # 加载角度数据
        angle_files = glob.glob(os.path.join(angle_folder, '*.mat'))
        if len(angle_files) > 0:
            try:
                # 尝试使用 h5py 读取
                with h5py.File(angle_files[0], 'r') as f:
                    all_theta = np.array(f['all_theta'])  # (50, 1000)
                    all_phi = np.array(f['all_phi'])      # (50, 1000)
                    
                    # 转置以匹配预期形状 [num_samples, num_time_samples]
                    self.theta_array = all_theta.T  # (1000, 50)
                    self.phi_array = all_phi.T      # (1000, 50)
                    
                    self.num_samples = int(np.array(f['num_samples'])[0, 0])
                    self.num_time_samples = int(np.array(f['num_time_samples'])[0, 0])
                    
                    print(f"角度数据形状: theta={self.theta_array.shape}, phi={self.phi_array.shape}")
                    print(f"信道数据文件数量: {len(self.channel_files)}")
                    
            except Exception as e:
                print(f"h5py 读取失败: {e}")
                # 如果 h5py 失败，尝试使用 scipy.io
                try:
                    angle_data = sio.loadmat(angle_files[0])
                    self.theta_array = angle_data['theta_array']  # [num_samples, num_time_samples]
                    self.phi_array = angle_data['phi_array']
                    self.num_samples = angle_data['num_samples'][0][0]
                    self.num_time_samples = angle_data['num_time_samples'][0][0]
                except Exception as e2:
                    print(f"scipy.io 读取也失败: {e2}")
                    raise e2
        
        # 确保角度数据和信道数据的样本数量匹配
        # 使用较小的数量作为实际的数据集大小
        self.actual_num_samples = min(len(self.channel_files), self.theta_array.shape[0])
        print(f"实际使用的样本数量: {self.actual_num_samples}")
        
    def __len__(self):
        return self.actual_num_samples
    
    def __getitem__(self, idx):
        # 确保索引在有效范围内
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
            print(f"h5py 读取信道数据失败: {e}")
            # 如果 h5py 失败，尝试使用 scipy.io
            try:
                channel_data = sio.loadmat(self.channel_files[idx])
                H_real = channel_data['H_real']  # [256, 256, 50]
                H_imag = channel_data['H_imag']
            except Exception as e2:
                print(f"scipy.io 读取信道数据也失败: {e2}")
                raise e2
        
        # 组合实部和虚部
        H_complex = H_real + 1j * H_imag
        
        # 数据归一化处理
        H_real_norm = (H_real - np.mean(H_real)) / (np.std(H_real) + 1e-8)
        H_imag_norm = (H_imag - np.mean(H_imag)) / (np.std(H_imag) + 1e-8)
        
        # 转换为PyTorch tensor
        H_input = np.stack([H_real_norm, H_imag_norm], axis=0)  # [2, 256, 256, 50]
        H_input = torch.FloatTensor(H_input)
        
        # 角度标签归一化（转换为弧度并归一化到[-1,1]）
        theta_norm = (self.theta_array[idx] * np.pi / 180) / (np.pi/2)  # 归一化到[-1,1]
        phi_norm = (self.phi_array[idx] * np.pi / 180) / np.pi  # 归一化到[-1,1]
        
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

class MultiTaskBeamformingTransformer(nn.Module):
    """多任务波束成形Transformer模型"""
    def __init__(self, input_dim=256*256*2, d_model=512, nhead=8, num_layers=6, 
                 num_subarrays=16, dropout=0.1):
        super(MultiTaskBeamformingTransformer, self).__init__()
        
        self.d_model = d_model
        self.num_subarrays = num_subarrays
        
        # 输入投影层
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layers = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_layers)
        
        # 任务特定头部
        # 1. 角度预测头
        self.angle_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # theta, phi
        )
        
        # 2. 波束成形权重头
        self.beamforming_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256*2)  # 复数权重的实部和虚部
        )
        
        # 3. 子阵列划分头
        self.subarray_head = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_subarrays)  # 每个天线元素属于哪个子阵列
        )
        
        # 4. 波束图质量预测头
        self.beam_quality_head = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 3)  # 主瓣宽度, 旁瓣抑制量, SNR增益
        )
        
    def forward(self, x):
        batch_size, channels, height, width, time_steps = x.shape
        print(f"输入形状: {x.shape}")
        
        # 重塑输入 [batch, time_steps, features]
        x = x.permute(0, 4, 1, 2, 3)  # [batch, time_steps, channels, height, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch, time_steps, channels*height*width]
        print(f"重塑后形状: {x.shape}")
        
        # 输入投影
        x = self.input_projection(x) * math.sqrt(self.d_model)
        
        # 位置编码
        x = x.transpose(0, 1)  # [time_steps, batch, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [batch, time_steps, d_model]
        
        # Transformer编码
        encoded = self.transformer_encoder(x)  # [batch, time_steps, d_model]
        
        # 使用最后一个时间步的输出进行预测
        last_output = encoded[:, -1, :]  # [batch, d_model]
        
        # 多任务输出
        angles = self.angle_head(last_output)  # [batch, 2]
        beamforming_weights = self.beamforming_head(last_output)  # [batch, 512]
        subarray_assignment = self.subarray_head(last_output)  # [batch, num_subarrays]
        beam_quality = self.beam_quality_head(last_output)  # [batch, 3]
        
        return {
            'angles': angles,
            'beamforming_weights': beamforming_weights,
            'subarray_assignment': subarray_assignment,
            'beam_quality': beam_quality
        }

def calculate_beam_pattern(weights, angles, antenna_positions):
    """计算波束图"""
    theta_range = np.linspace(0, 90, 91)
    phi_range = np.linspace(0, 360, 361)
    
    beam_pattern = np.zeros((len(theta_range), len(phi_range)))
    
    for i, theta in enumerate(theta_range):
        for j, phi in enumerate(phi_range):
            # 计算导向矢量
            az = phi * np.pi / 180
            el = theta * np.pi / 180
            k = (2*np.pi/0.0857) * np.array([np.cos(el)*np.cos(az), np.cos(el)*np.sin(az)])
            
            steering_vector = np.exp(1j * (antenna_positions @ k))
            
            # 计算波束响应
            response = np.abs(np.vdot(weights, steering_vector))**2
            beam_pattern[i, j] = response
    
    return beam_pattern, theta_range, phi_range

def compute_beam_metrics(beam_pattern, target_theta, target_phi):
    """计算波束图性能指标"""
    # 主瓣宽度计算（3dB带宽）
    max_gain = np.max(beam_pattern)
    half_power = max_gain / 2
    
    # 找到最大值位置
    max_idx = np.unravel_index(np.argmax(beam_pattern), beam_pattern.shape)
    
    # 计算主瓣宽度（简化版本）
    main_lobe_width = np.sum(beam_pattern > half_power) * (360 / beam_pattern.shape[1])
    
    # 旁瓣抑制量
    # 排除主瓣区域
    main_lobe_mask = np.zeros_like(beam_pattern)
    theta_idx, phi_idx = max_idx
    main_lobe_mask[max(0, theta_idx-5):min(beam_pattern.shape[0], theta_idx+6),
                   max(0, phi_idx-10):min(beam_pattern.shape[1], phi_idx+11)] = 1
    
    side_lobe_region = beam_pattern * (1 - main_lobe_mask)
    max_side_lobe = np.max(side_lobe_region)
    side_lobe_suppression = 10 * np.log10(max_gain / (max_side_lobe + 1e-10))
    
    return main_lobe_width, side_lobe_suppression

def multi_task_loss(predictions, targets, alpha=1.0, beta=0.1, gamma=0.01, delta=0.01):
    """多任务损失函数"""
    pred_angles = predictions['angles']
    pred_beamforming = predictions['beamforming_weights']
    pred_subarray = predictions['subarray_assignment']
    pred_beam_quality = predictions['beam_quality']
    
    target_theta, target_phi, H_complex = targets
    
    # 1. 角度预测损失（使用平滑L1损失，更稳定）
    angle_loss = F.smooth_l1_loss(pred_angles[:, 0], target_theta[:, -1]) + \
                 F.smooth_l1_loss(pred_angles[:, 1], target_phi[:, -1])
    
    # 2. 波束成形损失（基于SNR最大化，添加数值稳定性）
    batch_size = pred_beamforming.shape[0]
    beamforming_loss = 0
    
    for i in range(batch_size):
        try:
            # 重构复数权重
            weights_real = pred_beamforming[i, :256]
            weights_imag = pred_beamforming[i, 256:]
            weights = weights_real + 1j * weights_imag
            
            # 添加数值稳定性检查
            if torch.any(torch.isnan(weights)) or torch.any(torch.isinf(weights)):
                beamforming_loss += 1.0  # 惩罚不稳定的权重
                continue
                
            weights = weights / (torch.norm(weights) + 1e-8)  # 安全归一化
            
            # 计算输出SNR（简化版本，避免复杂计算）
            H_i = H_complex[i][:, :, -1]  # 使用最后一个时间步
            H_i = H_i.to(weights.dtype)
            
            # 安全的SNR计算
            signal_power = torch.abs(torch.vdot(weights, H_i.flatten()))**2
            noise_power = torch.norm(weights)**2 + 1e-8
            snr = signal_power / noise_power
            
            # 使用log1p而不是log，避免数值问题
            beamforming_loss += -torch.log1p(snr)
            
        except Exception as e:
            # 如果计算失败，添加一个惩罚项
            beamforming_loss += 1.0
    
    beamforming_loss /= batch_size
    
    # 3. 子阵列划分损失（正则化项）
    subarray_loss = torch.var(torch.sum(F.softmax(pred_subarray, dim=-1), dim=0))
    
    # 4. 波束质量损失（辅助监督）
    beam_quality_loss = torch.mean(pred_beam_quality**2)  # 简化的正则化项
    
    # 确保所有损失都是有限的
    angle_loss = torch.clamp(angle_loss, 0, 100)
    beamforming_loss = torch.clamp(beamforming_loss, 0, 100)
    subarray_loss = torch.clamp(subarray_loss, 0, 10)
    beam_quality_loss = torch.clamp(beam_quality_loss, 0, 10)
    
    total_loss = alpha * angle_loss + beta * beamforming_loss + \
                 gamma * subarray_loss + delta * beam_quality_loss
    
    return total_loss, angle_loss, beamforming_loss, subarray_loss, beam_quality_loss

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-5, device='cuda'):
    """训练函数"""
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience_counter = 0
    early_stop_patience = 15
    
    print(f"开始训练，学习率: {lr}")
    
    # 训练进度条
    epoch_progress = tqdm(range(num_epochs), desc="训练进度", unit="epoch")
    
    for epoch in epoch_progress:
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        train_metrics = {'angle': 0, 'beam': 0, 'sub': 0, 'quality': 0}
        
        # 批次进度条
        batch_progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                            leave=False, unit="batch")
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(batch_progress):
            try:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                H_complex = H_complex.to(device)
                
                optimizer.zero_grad()
                
                # 前向传播
                predictions = model(H_input)
                
                # 计算损失
                loss, angle_loss, beam_loss, sub_loss, quality_loss = multi_task_loss(
                    predictions, (theta_label, phi_label, H_complex)
                )
                
                # 检查损失是否有效
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"警告: 在Epoch {epoch}, Batch {batch_idx} 检测到无效损失，跳过此批次")
                    continue
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪（更保守的设置）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                
                optimizer.step()
                
                train_loss_epoch += loss.item()
                train_metrics['angle'] += angle_loss.item()
                train_metrics['beam'] += beam_loss.item()
                train_metrics['sub'] += sub_loss.item()
                train_metrics['quality'] += quality_loss.item()
                
                # 更新批次进度条
                batch_progress.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Angle': f'{angle_loss.item():.4f}',
                    'Beam': f'{beam_loss.item():.4f}'
                })
                
            except Exception as e:
                print(f"训练批次错误: {e}")
                continue
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        val_metrics = {'angle': 0, 'beam': 0, 'sub': 0, 'quality': 0}
        
        with torch.no_grad():
            for H_input, theta_label, phi_label, H_complex in val_loader:
                try:
                    H_input = H_input.to(device)
                    theta_label = theta_label.to(device)
                    phi_label = phi_label.to(device)
                    H_complex = H_complex.to(device)
                    
                    predictions = model(H_input)
                    loss, angle_loss, beam_loss, sub_loss, quality_loss = multi_task_loss(
                        predictions, (theta_label, phi_label, H_complex)
                    )
                    
                    if not (torch.isnan(loss) or torch.isinf(loss)):
                        val_loss_epoch += loss.item()
                        val_metrics['angle'] += angle_loss.item()
                        val_metrics['beam'] += beam_loss.item()
                        val_metrics['sub'] += sub_loss.item()
                        val_metrics['quality'] += quality_loss.item()
                        
                except Exception as e:
                    print(f"验证批次错误: {e}")
                    continue
        
        # 计算平均损失
        train_loss_avg = train_loss_epoch / len(train_loader) if len(train_loader) > 0 else float('inf')
        val_loss_avg = val_loss_epoch / len(val_loader) if len(val_loader) > 0 else float('inf')
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        # 学习率调度
        scheduler.step(val_loss_avg)
        
        # 更新epoch进度条
        epoch_progress.set_postfix({
            'Train Loss': f'{train_loss_avg:.4f}',
            'Val Loss': f'{val_loss_avg:.4f}',
            'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'
        })
        
        # 早停机制
        if val_loss_avg < best_val_loss:
            best_val_loss = val_loss_avg
            patience_counter = 0
            torch.save(model.state_dict(), 'best_beamforming_model.pth')
            tqdm.write(f"✓ 保存最佳模型 (Val Loss: {val_loss_avg:.4f})")
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            tqdm.write(f"早停触发！在epoch {epoch+1}停止训练")
            break
        
        # 详细日志（每5个epoch）
        if (epoch + 1) % 5 == 0:
            tqdm.write(f"Epoch {epoch+1}: Train={train_loss_avg:.4f}, Val={val_loss_avg:.4f}, "
                      f"Angle={train_metrics['angle']/len(train_loader):.4f}, "
                      f"Beam={train_metrics['beam']/len(train_loader):.4f}")
    
    return train_losses, val_losses

def test_model(model, test_loader, device='cuda'):
    """测试函数"""
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    angle_errors = []
    
    antenna_positions = np.array([[i, j] for i in range(16) for j in range(16)]) * 0.0428
    
    test_progress = tqdm(test_loader, desc="测试进度", unit="batch")
    
    with torch.no_grad():
        for H_input, theta_label, phi_label, H_complex in test_progress:
            try:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                
                # 预测
                predictions = model(H_input)
                
                # 移动到CPU进行后处理
                pred_angles = predictions['angles'].cpu().numpy()
                pred_beamforming = predictions['beamforming_weights'].cpu().numpy()
                pred_subarray = predictions['subarray_assignment'].cpu().numpy()
                
                # 将归一化的角度转换回度数
                # theta从[-1,1]转换回[0,90]度
                pred_theta_deg = (pred_angles[:, 0] * (np.pi/2)) * 180 / np.pi
                pred_phi_deg = (pred_angles[:, 1] * np.pi) * 180 / np.pi
                
                target_theta_deg = (theta_label[:, -1].cpu().numpy() * (np.pi/2)) * 180 / np.pi
                target_phi_deg = (phi_label[:, -1].cpu().numpy() * np.pi) * 180 / np.pi
                
                # 计算角度误差
                theta_error = np.abs(pred_theta_deg - target_theta_deg)
                phi_error = np.abs(pred_phi_deg - target_phi_deg)
                angle_errors.extend(theta_error + phi_error)
                
                # 分析波束成形性能
                for i in range(len(pred_angles)):
                    try:
                        # 重构复数权重
                        weights_real = pred_beamforming[i, :256]
                        weights_imag = pred_beamforming[i, 256:]
                        weights = weights_real + 1j * weights_imag
                        
                        # 检查权重的有效性
                        if np.any(np.isnan(weights)) or np.any(np.isinf(weights)):
                            weights = np.ones(256, dtype=complex) / np.sqrt(256)  # 使用默认权重
                        else:
                            weights = weights / (np.linalg.norm(weights) + 1e-8)
                        
                        # 计算波束图
                        beam_pattern, _, _ = calculate_beam_pattern(
                            weights, [pred_theta_deg[i], pred_phi_deg[i]], antenna_positions
                        )
                        
                        # 计算性能指标
                        main_lobe_width, side_lobe_suppression = compute_beam_metrics(
                            beam_pattern, target_theta_deg[i], target_phi_deg[i]
                        )
                        
                        # 子阵列划分
                        subarray_assignment = np.argmax(pred_subarray[i])
                        
                        all_predictions.append({
                            'predicted_theta': pred_theta_deg[i],
                            'predicted_phi': pred_phi_deg[i],
                            'target_theta': target_theta_deg[i],
                            'target_phi': target_phi_deg[i],
                            'main_lobe_width': main_lobe_width,
                            'side_lobe_suppression': side_lobe_suppression,
                            'subarray_assignment': subarray_assignment,
                            'beamforming_weights': weights
                        })
                        
                    except Exception as e:
                        print(f"处理样本 {i} 时出错: {e}")
                        continue
                        
            except Exception as e:
                print(f"测试批次错误: {e}")
                continue
    
    # 计算统计指标
    if len(angle_errors) > 0:
        mean_angle_error = np.mean(angle_errors)
        std_angle_error = np.std(angle_errors)
        
        print(f"\n测试结果:")
        print(f"平均角度误差: {mean_angle_error:.2f}°")
        print(f"角度误差标准差: {std_angle_error:.2f}°")
        print(f"有效预测数量: {len(all_predictions)}")
    else:
        print("没有有效的测试结果")
        mean_angle_error = float('inf')
        std_angle_error = float('inf')
    
    return all_predictions

def main():
    """主函数"""
    # 抑制一些警告
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 数据路径（需要根据实际路径修改）
    channel_folder = "samples_data/channel_data_*"
    angle_folder = "samples_data/angle_data_*"
    
    # 查找数据文件夹
    channel_folders = glob.glob(channel_folder)
    angle_folders = glob.glob(angle_folder)
    
    if len(channel_folders) == 0 or len(angle_folders) == 0:
        print("未找到数据文件夹，请先运行数据生成代码")
        return
    
    print(f"找到数据文件夹: {len(channel_folders)} 个信道文件夹, {len(angle_folders)} 个角度文件夹")
    
    # 创建数据集
    dataset = BeamformingDataset(channel_folders[0], angle_folders[0])
    
    # 数据分割
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)
    
    print(f"数据集划分: 训练={len(train_indices)}, 验证={len(val_indices)}, 测试={len(test_indices)}")
    
    # 创建数据加载器（使用更小的批次大小）
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)
    
    # 创建模型（稍微减小模型复杂度）
    model = MultiTaskBeamformingTransformer(
        input_dim=256*256*2,
        d_model=256,  # 从512减小到256
        nhead=8,
        num_layers=4,  # 从6减小到4
        num_subarrays=16,
        dropout=0.2  # 增加dropout
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型（使用更保守的参数）
    print("\n开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=30,  # 减少训练轮数
        lr=5e-6,       # 降低学习率
        device=device
    )
    
    # 加载最佳模型
    if os.path.exists('best_beamforming_model.pth'):
        model.load_state_dict(torch.load('best_beamforming_model.pth'))
        print("加载最佳模型完成")
    
    # 测试模型
    print("\n开始测试...")
    predictions = test_model(model, test_loader, device=device)
    
    # 绘制训练曲线
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
    
    plt.figure(figsize=(15, 5))
    
    # 训练损失曲线
    plt.subplot(1, 3, 1)
    if len(train_losses) > 0 and len(val_losses) > 0:
        plt.plot(train_losses, label='Training Loss', alpha=0.8)
        plt.plot(val_losses, label='Validation Loss', alpha=0.8)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training Process')
        plt.yscale('log')  # 使用对数尺度
    
    # 角度预测结果
    plt.subplot(1, 3, 2)
    if len(predictions) > 0:
        pred_theta = [p['predicted_theta'] for p in predictions if 'predicted_theta' in p]
        target_theta = [p['target_theta'] for p in predictions if 'target_theta' in p]
        
        if len(pred_theta) > 0 and len(target_theta) > 0:
            plt.scatter(target_theta, pred_theta, alpha=0.6, s=30)
            min_angle, max_angle = min(min(target_theta), min(pred_theta)), max(max(target_theta), max(pred_theta))
            plt.plot([min_angle, max_angle], [min_angle, max_angle], 'r--', label='Perfect Prediction')
            plt.xlabel('True Angle θ (degrees)')
            plt.ylabel('Predicted Angle θ (degrees)')
            plt.legend()
            plt.title('Angle Prediction Results')
    
    # 损失分解
    plt.subplot(1, 3, 3)
    if len(train_losses) > 1:
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    print(f"\n训练结果图表已保存为 'training_results.png'")
    
    # 显示训练总结
    print(f"\n📊 训练总结:")
    print(f"• 最终训练损失: {train_losses[-1]:.4f}" if train_losses else "• 无训练损失记录")
    print(f"• 最终验证损失: {val_losses[-1]:.4f}" if val_losses else "• 无验证损失记录")
    print(f"• 总训练轮数: {len(train_losses)}")
    print(f"• 有效预测数量: {len(predictions)}")
    
    try:
        plt.show()
    except:
        print("无法显示图表，但已保存到文件")

if __name__ == "__main__":
    main() 