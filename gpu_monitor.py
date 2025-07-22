#!/usr/bin/env python3
"""
GPU性能监控脚本
在训练过程中监控GPU使用情况
"""

import subprocess
import time
import json
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np
import threading
import os

class GPUMonitor:
    def __init__(self, log_file="gpu_monitor.log", plot_file="gpu_usage.png"):
        self.log_file = log_file
        self.plot_file = plot_file
        self.monitoring = False
        self.data = {
            'timestamp': [],
            'gpu_utilization': [],
            'memory_used': [],
            'memory_total': [],
            'temperature': [],
            'power_draw': []
        }
        
    def get_gpu_info(self):
        """获取GPU信息"""
        try:
            cmd = [
                'nvidia-smi',
                '--query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                gpu_info = []
                
                for line in lines:
                    parts = [part.strip() for part in line.split(',')]
                    if len(parts) >= 6:
                        try:
                            info = {
                                'timestamp': parts[0],
                                'utilization': float(parts[1]) if parts[1] != '[Not Supported]' else 0,
                                'memory_used': float(parts[2]),
                                'memory_total': float(parts[3]),
                                'temperature': float(parts[4]) if parts[4] != '[Not Supported]' else 0,
                                'power_draw': float(parts[5].split()[0]) if '[Not Supported]' not in parts[5] else 0
                            }
                            gpu_info.append(info)
                        except (ValueError, IndexError):
                            continue
                
                return gpu_info
        except Exception as e:
            print(f"获取GPU信息失败: {e}")
            return []
        
        return []
    
    def log_gpu_status(self):
        """记录GPU状态"""
        gpu_info = self.get_gpu_info()
        
        if gpu_info:
            timestamp = datetime.now()
            
            # 计算平均值（多GPU情况）
            avg_utilization = np.mean([info['utilization'] for info in gpu_info])
            total_memory_used = sum([info['memory_used'] for info in gpu_info])
            total_memory = sum([info['memory_total'] for info in gpu_info])
            avg_temperature = np.mean([info['temperature'] for info in gpu_info])
            total_power = sum([info['power_draw'] for info in gpu_info])
            
            # 存储数据
            self.data['timestamp'].append(timestamp)
            self.data['gpu_utilization'].append(avg_utilization)
            self.data['memory_used'].append(total_memory_used)
            self.data['memory_total'].append(total_memory)
            self.data['temperature'].append(avg_temperature)
            self.data['power_draw'].append(total_power)
            
            # 写入日志文件
            log_entry = {
                'timestamp': timestamp.isoformat(),
                'avg_gpu_utilization': avg_utilization,
                'total_memory_used_mb': total_memory_used,
                'total_memory_gb': total_memory / 1024,
                'avg_temperature': avg_temperature,
                'total_power_w': total_power,
                'individual_gpus': gpu_info
            }
            
            with open(self.log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')
            
            return log_entry
        
        return None
    
    def start_monitoring(self, interval=5):
        """开始监控"""
        print(f"🔍 开始GPU监控，每{interval}秒记录一次")
        print(f"📝 日志文件: {self.log_file}")
        
        self.monitoring = True
        
        def monitor_loop():
            while self.monitoring:
                status = self.log_gpu_status()
                if status:
                    print(f"⏰ {datetime.now().strftime('%H:%M:%S')} - "
                          f"GPU: {status['avg_gpu_utilization']:.1f}% | "
                          f"内存: {status['total_memory_used_mb']/1024:.1f}GB | "
                          f"温度: {status['avg_temperature']:.1f}°C | "
                          f"功耗: {status['total_power_w']:.1f}W")
                
                time.sleep(interval)
        
        self.monitor_thread = threading.Thread(target=monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控"""
        print("🛑 停止GPU监控")
        self.monitoring = False
        if hasattr(self, 'monitor_thread'):
            self.monitor_thread.join(timeout=1)
    
    def plot_usage(self):
        """绘制使用情况图表"""
        if not self.data['timestamp']:
            print("❌ 没有数据可绘制")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # GPU利用率
        axes[0, 0].plot(self.data['timestamp'], self.data['gpu_utilization'], 'b-', linewidth=2)
        axes[0, 0].set_title('GPU利用率')
        axes[0, 0].set_ylabel('利用率 (%)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim(0, 100)
        
        # 内存使用
        memory_used_gb = [m/1024 for m in self.data['memory_used']]
        memory_total_gb = [m/1024 for m in self.data['memory_total']]
        
        axes[0, 1].plot(self.data['timestamp'], memory_used_gb, 'g-', linewidth=2, label='已使用')
        axes[0, 1].plot(self.data['timestamp'], memory_total_gb, 'r--', linewidth=2, label='总容量')
        axes[0, 1].set_title('显存使用')
        axes[0, 1].set_ylabel('显存 (GB)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 温度
        axes[1, 0].plot(self.data['timestamp'], self.data['temperature'], 'r-', linewidth=2)
        axes[1, 0].set_title('GPU温度')
        axes[1, 0].set_ylabel('温度 (°C)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 功耗
        axes[1, 1].plot(self.data['timestamp'], self.data['power_draw'], 'm-', linewidth=2)
        axes[1, 1].set_title('功耗')
        axes[1, 1].set_ylabel('功耗 (W)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 格式化时间轴
        for ax in axes.flat:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 GPU使用图表已保存为: {self.plot_file}")
    
    def get_summary(self):
        """获取监控摘要"""
        if not self.data['timestamp']:
            return "没有监控数据"
        
        duration = (self.data['timestamp'][-1] - self.data['timestamp'][0]).total_seconds()
        
        summary = {
            'duration_minutes': duration / 60,
            'avg_gpu_utilization': np.mean(self.data['gpu_utilization']),
            'max_memory_used_gb': max(self.data['memory_used']) / 1024,
            'avg_temperature': np.mean(self.data['temperature']),
            'avg_power_draw': np.mean(self.data['power_draw']),
            'total_energy_kwh': sum(self.data['power_draw']) * 5 / 3600 / 1000  # 假设5秒间隔
        }
        
        return summary

def main():
    """独立运行GPU监控"""
    monitor = GPUMonitor()
    
    try:
        monitor.start_monitoring(interval=5)
        
        print("按 Ctrl+C 停止监控...")
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🛑 用户中断监控")
    finally:
        monitor.stop_monitoring()
        monitor.plot_usage()
        
        summary = monitor.get_summary()
        print(f"\n📋 监控摘要:")
        print(f"   • 监控时长: {summary['duration_minutes']:.1f} 分钟")
        print(f"   • 平均GPU利用率: {summary['avg_gpu_utilization']:.1f}%")
        print(f"   • 最大显存使用: {summary['max_memory_used_gb']:.1f}GB")
        print(f"   • 平均温度: {summary['avg_temperature']:.1f}°C")
        print(f"   • 平均功耗: {summary['avg_power_draw']:.1f}W")
        print(f"   • 总耗电量: {summary['total_energy_kwh']:.3f} kWh")

if __name__ == "__main__":
    main()
