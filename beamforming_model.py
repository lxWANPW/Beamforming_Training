import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import scipy.io as sio
import os
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import math

class BeamformingDataset(Dataset):
    """天线阵列信道数据集"""
    def __init__(self, channel_folder, angle_folder, transform=None):
        self.channel_folder = channel_folder
        self.angle_folder = angle_folder
        self.transform = transform
        
        # 加载角度数据
        angle_files = glob.glob(os.path.join(angle_folder, '*.mat'))
        if len(angle_files) > 0:
            angle_data = sio.loadmat(angle_files[0])
            self.theta_array = angle_data['theta_array']  # [num_samples, num_time_samples]
            self.phi_array = angle_data['phi_array']
            self.num_samples = angle_data['num_samples'][0][0]
            self.num_time_samples = angle_data['num_time_samples'][0][0]
        
        # 获取信道数据文件列表
        self.channel_files = sorted(glob.glob(os.path.join(channel_folder, '*.mat')))
        
    def __len__(self):
        return len(self.channel_files)
    
    def __getitem__(self, idx):
        # 加载信道数据
        channel_data = sio.loadmat(self.channel_files[idx])
        H_real = channel_data['H_real']  # [256, 256, 3000]
        H_imag = channel_data['H_imag']
        
        # 组合实部和虚部
        H_complex = H_real + 1j * H_imag
        
        # 转换为PyTorch tensor
        H_input = np.stack([H_real, H_imag], axis=0)  # [2, 256, 256, 3000]
        H_input = torch.FloatTensor(H_input)
        
        # 角度标签
        theta_label = torch.FloatTensor(self.theta_array[idx])  # [3000]
        phi_label = torch.FloatTensor(self.phi_array[idx])
        
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
        
        # 重塑输入 [batch, time_steps, features]
        x = x.permute(0, 4, 1, 2, 3)  # [batch, time_steps, channels, height, width]
        x = x.reshape(batch_size, time_steps, -1)  # [batch, time_steps, channels*height*width]
        
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

def multi_task_loss(predictions, targets, alpha=1.0, beta=1.0, gamma=1.0, delta=1.0):
    """多任务损失函数"""
    pred_angles = predictions['angles']
    pred_beamforming = predictions['beamforming_weights']
    pred_subarray = predictions['subarray_assignment']
    pred_beam_quality = predictions['beam_quality']
    
    target_theta, target_phi, H_complex = targets
    
    # 1. 角度预测损失
    angle_loss = F.mse_loss(pred_angles[:, 0], target_theta[:, -1]) + \
                 F.mse_loss(pred_angles[:, 1], target_phi[:, -1])
    
    # 2. 波束成形损失（基于SNR最大化）
    batch_size = pred_beamforming.shape[0]
    beamforming_loss = 0
    
    for i in range(batch_size):
        # 重构复数权重
        weights_real = pred_beamforming[i, :256]
        weights_imag = pred_beamforming[i, 256:]
        weights = weights_real + 1j * weights_imag
        weights = weights / torch.norm(weights)  # 归一化
        
        # 计算输出SNR
        H_i = H_complex[i][:, :, -1]  # 使用最后一个时间步
        signal_power = torch.abs(torch.vdot(weights, H_i @ weights))**2
        noise_power = torch.norm(weights)**2
        snr = signal_power / (noise_power + 1e-10)
        
        beamforming_loss += -torch.log(snr + 1e-10)  # 最大化SNR
    
    beamforming_loss /= batch_size
    
    # 3. 子阵列划分损失（正则化项）
    subarray_loss = torch.var(torch.sum(F.softmax(pred_subarray, dim=-1), dim=0))
    
    # 4. 波束质量损失（辅助监督）
    beam_quality_loss = F.mse_loss(pred_beam_quality, 
                                   torch.zeros_like(pred_beam_quality))  # placeholder
    
    total_loss = alpha * angle_loss + beta * beamforming_loss + \
                 gamma * subarray_loss + delta * beam_quality_loss
    
    return total_loss, angle_loss, beamforming_loss, subarray_loss, beam_quality_loss

def train_model(model, train_loader, val_loader, num_epochs=100, lr=1e-4, device='cuda'):
    """训练函数"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss_epoch = 0
        
        for batch_idx, (H_input, theta_label, phi_label, H_complex) in enumerate(train_loader):
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
            
            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss_epoch += loss.item()
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, '
                      f'Total Loss: {loss.item():.4f}, '
                      f'Angle Loss: {angle_loss.item():.4f}, '
                      f'Beam Loss: {beam_loss.item():.4f}')
        
        # 验证阶段
        model.eval()
        val_loss_epoch = 0
        
        with torch.no_grad():
            for H_input, theta_label, phi_label, H_complex in val_loader:
                H_input = H_input.to(device)
                theta_label = theta_label.to(device)
                phi_label = phi_label.to(device)
                H_complex = H_complex.to(device)
                
                predictions = model(H_input)
                loss, _, _, _, _ = multi_task_loss(
                    predictions, (theta_label, phi_label, H_complex)
                )
                val_loss_epoch += loss.item()
        
        train_loss_avg = train_loss_epoch / len(train_loader)
        val_loss_avg = val_loss_epoch / len(val_loader)
        
        train_losses.append(train_loss_avg)
        val_losses.append(val_loss_avg)
        
        scheduler.step(val_loss_avg)
        
        print(f'Epoch {epoch+1}/{num_epochs}, '
              f'Train Loss: {train_loss_avg:.4f}, '
              f'Val Loss: {val_loss_avg:.4f}')
        
        # 保存最佳模型
        if epoch == 0 or val_loss_avg < min(val_losses[:-1]):
            torch.save(model.state_dict(), 'best_beamforming_model.pth')
    
    return train_losses, val_losses

def test_model(model, test_loader, device='cuda'):
    """测试函数"""
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_targets = []
    angle_errors = []
    
    antenna_positions = np.array([[i, j] for i in range(16) for j in range(16)]) * 0.0428
    
    with torch.no_grad():
        for H_input, theta_label, phi_label, H_complex in test_loader:
            H_input = H_input.to(device)
            theta_label = theta_label.to(device)
            phi_label = phi_label.to(device)
            
            # 预测
            predictions = model(H_input)
            
            # 移动到CPU进行后处理
            pred_angles = predictions['angles'].cpu().numpy()
            pred_beamforming = predictions['beamforming_weights'].cpu().numpy()
            pred_subarray = predictions['subarray_assignment'].cpu().numpy()
            
            target_theta = theta_label[:, -1].cpu().numpy()
            target_phi = phi_label[:, -1].cpu().numpy()
            
            # 计算角度误差
            theta_error = np.abs(pred_angles[:, 0] - target_theta)
            phi_error = np.abs(pred_angles[:, 1] - target_phi)
            angle_errors.extend(theta_error + phi_error)
            
            # 分析波束成形性能
            for i in range(len(pred_angles)):
                # 重构复数权重
                weights_real = pred_beamforming[i, :256]
                weights_imag = pred_beamforming[i, 256:]
                weights = weights_real + 1j * weights_imag
                weights = weights / np.linalg.norm(weights)
                
                # 计算波束图
                beam_pattern, _, _ = calculate_beam_pattern(
                    weights, pred_angles[i], antenna_positions
                )
                
                # 计算性能指标
                main_lobe_width, side_lobe_suppression = compute_beam_metrics(
                    beam_pattern, target_theta[i], target_phi[i]
                )
                
                # 子阵列划分
                subarray_assignment = np.argmax(pred_subarray[i])
                
                all_predictions.append({
                    'predicted_theta': pred_angles[i, 0],
                    'predicted_phi': pred_angles[i, 1],
                    'target_theta': target_theta[i],
                    'target_phi': target_phi[i],
                    'main_lobe_width': main_lobe_width,
                    'side_lobe_suppression': side_lobe_suppression,
                    'subarray_assignment': subarray_assignment,
                    'beamforming_weights': weights
                })
    
    # 计算统计指标
    mean_angle_error = np.mean(angle_errors)
    std_angle_error = np.std(angle_errors)
    
    print(f"测试结果:")
    print(f"平均角度误差: {mean_angle_error:.2f}°")
    print(f"角度误差标准差: {std_angle_error:.2f}°")
    
    return all_predictions

def main():
    """主函数"""
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
    
    # 创建数据集
    dataset = BeamformingDataset(channel_folders[0], angle_folders[0])
    
    # 数据分割
    indices = list(range(len(dataset)))
    train_indices, test_indices = train_test_split(indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.2, random_state=42)
    
    # 创建数据加载器
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    # 创建模型
    model = MultiTaskBeamformingTransformer(
        input_dim=256*256*2,
        d_model=512,
        nhead=8,
        num_layers=6,
        num_subarrays=16
    )
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 训练模型
    print("开始训练...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, 
        num_epochs=50, lr=1e-4, device=device
    )
    
    # 加载最佳模型
    model.load_state_dict(torch.load('best_beamforming_model.pth'))
    
    # 测试模型
    print("开始测试...")
    predictions = test_model(model, test_loader, device=device)
    
    # 绘制训练曲线
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='训练损失')
    plt.plot(val_losses, label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('训练过程')
    
    # 绘制角度预测结果
    plt.subplot(1, 2, 2)
    pred_theta = [p['predicted_theta'] for p in predictions]
    target_theta = [p['target_theta'] for p in predictions]
    plt.scatter(target_theta, pred_theta, alpha=0.6)
    plt.plot([0, 90], [0, 90], 'r--', label='理想预测')
    plt.xlabel('真实角度 θ (度)')
    plt.ylabel('预测角度 θ (度)')
    plt.legend()
    plt.title('角度预测结果')
    
    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main() 