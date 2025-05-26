#!/usr/bin/env python3
import psutil
import GPUtil
import time
import sys
import subprocess
from datetime import datetime
import matplotlib.pyplot as plt
from collections import deque

class MemoryMonitor:
    def __init__(self, process_name="omoknuni_cli_final", log_file="memory_usage.log"):
        self.process_name = process_name
        self.log_file = log_file
        self.data = {
            'time': deque(maxlen=1000),
            'cpu_percent': deque(maxlen=1000),
            'ram_mb': deque(maxlen=1000),
            'vram_mb': deque(maxlen=1000),
            'gpu_util': deque(maxlen=1000)
        }
        self.start_time = time.time()
        
    def find_process(self):
        """Find the process by name"""
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if self.process_name in proc.info['name'] or \
                   any(self.process_name in arg for arg in (proc.info['cmdline'] or [])):
                    return proc
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return None
        
    def get_gpu_memory_usage(self):
        """Get GPU memory usage"""
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]  # Assuming single GPU
                return gpu.memoryUsed, gpu.load * 100
        except:
            pass
        return 0, 0
        
    def monitor(self, interval=1.0):
        """Monitor memory usage"""
        print(f"Monitoring {self.process_name}...")
        print("Time | CPU% | RAM MB | VRAM MB | GPU% | Status")
        print("-" * 60)
        
        with open(self.log_file, 'w') as f:
            f.write("timestamp,cpu_percent,ram_mb,vram_mb,gpu_util\n")
            
            while True:
                try:
                    proc = self.find_process()
                    if not proc:
                        print(f"Process {self.process_name} not found. Waiting...")
                        time.sleep(interval)
                        continue
                    
                    # Get metrics
                    cpu_percent = proc.cpu_percent(interval=0.1)
                    ram_mb = proc.memory_info().rss / 1024 / 1024
                    vram_mb, gpu_util = self.get_gpu_memory_usage()
                    
                    # Record data
                    current_time = time.time() - self.start_time
                    self.data['time'].append(current_time)
                    self.data['cpu_percent'].append(cpu_percent)
                    self.data['ram_mb'].append(ram_mb)
                    self.data['vram_mb'].append(vram_mb)
                    self.data['gpu_util'].append(gpu_util)
                    
                    # Log to file
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    f.write(f"{timestamp},{cpu_percent:.1f},{ram_mb:.1f},{vram_mb:.1f},{gpu_util:.1f}\n")
                    f.flush()
                    
                    # Determine status
                    status = "OK"
                    if ram_mb > 32000:  # 32GB warning
                        status = "HIGH RAM!"
                    if vram_mb > 6000:  # 6GB warning
                        status = "HIGH VRAM!"
                    if cpu_percent < 30 and gpu_util > 80:
                        status = "CPU BOTTLENECK"
                    
                    # Print status
                    print(f"{current_time:5.0f}s | {cpu_percent:4.0f}% | {ram_mb:7.0f} | {vram_mb:7.0f} | {gpu_util:3.0f}% | {status}")
                    
                    time.sleep(interval)
                    
                except KeyboardInterrupt:
                    print("\nMonitoring stopped.")
                    self.plot_results()
                    break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(interval)
                    
    def plot_results(self):
        """Plot monitoring results"""
        if not self.data['time']:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        # CPU Usage
        ax1.plot(self.data['time'], self.data['cpu_percent'], 'b-')
        ax1.set_title('CPU Usage %')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('CPU %')
        ax1.grid(True)
        
        # RAM Usage
        ax2.plot(self.data['time'], self.data['ram_mb'], 'g-')
        ax2.set_title('RAM Usage (MB)')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('RAM (MB)')
        ax2.grid(True)
        
        # VRAM Usage
        ax3.plot(self.data['time'], self.data['vram_mb'], 'r-')
        ax3.set_title('VRAM Usage (MB)')
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('VRAM (MB)')
        ax3.grid(True)
        
        # GPU Utilization
        ax4.plot(self.data['time'], self.data['gpu_util'], 'm-')
        ax4.set_title('GPU Utilization %')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('GPU %')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('memory_usage_plot.png')
        print(f"Plot saved to memory_usage_plot.png")

if __name__ == "__main__":
    monitor = MemoryMonitor()
    monitor.monitor()