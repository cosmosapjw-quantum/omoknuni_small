#!/usr/bin/env python3
import subprocess
import time
import sys
import statistics

def get_gpu_utilization():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        return int(result.stdout.strip())
    except:
        return 0

def get_cpu_utilization():
    try:
        result = subprocess.run(['top', '-bn1'], capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'Cpu(s)' in line:
                # Extract overall CPU usage
                parts = line.split()
                for i, part in enumerate(parts):
                    if 'id,' in part:
                        idle = float(parts[i-1].replace('%', ''))
                        return 100 - idle
        return 0
    except:
        return 0

def main():
    print("Monitoring CPU and GPU utilization...")
    print("Press Ctrl+C to stop\n")
    
    cpu_samples = []
    gpu_samples = []
    
    try:
        while True:
            cpu = get_cpu_utilization()
            gpu = get_gpu_utilization()
            
            cpu_samples.append(cpu)
            gpu_samples.append(gpu)
            
            # Keep only last 60 samples (2 minutes)
            if len(cpu_samples) > 60:
                cpu_samples.pop(0)
                gpu_samples.pop(0)
            
            # Calculate averages
            cpu_avg = statistics.mean(cpu_samples)
            gpu_avg = statistics.mean(gpu_samples)
            
            # Calculate min/max for last 10 samples
            recent_cpu = cpu_samples[-10:] if len(cpu_samples) >= 10 else cpu_samples
            recent_gpu = gpu_samples[-10:] if len(gpu_samples) >= 10 else gpu_samples
            
            cpu_min = min(recent_cpu)
            cpu_max = max(recent_cpu)
            gpu_min = min(recent_gpu) 
            gpu_max = max(recent_gpu)
            
            status = ""
            if cpu_avg >= 70 and gpu_avg >= 70:
                status = "✅ TARGET ACHIEVED!"
            elif cpu_avg >= 70 or gpu_avg >= 70:
                status = "⚠️  Partial"
            else:
                status = "❌ Below target"
            
            print(f"\r🖥️  CPU: {cpu:.1f}% (avg: {cpu_avg:.1f}%, range: {cpu_min:.0f}-{cpu_max:.0f}%) | " + 
                  f"🎮 GPU: {gpu}% (avg: {gpu_avg:.1f}%, range: {gpu_min}-{gpu_max}%) | " +
                  f"{status}    ", end='', flush=True)
            
            time.sleep(2)
            
    except KeyboardInterrupt:
        print(f"\n\nFinal Statistics:")
        print(f"CPU Average: {statistics.mean(cpu_samples):.1f}%")
        print(f"GPU Average: {statistics.mean(gpu_samples):.1f}%")
        print(f"CPU Std Dev: {statistics.stdev(cpu_samples):.1f}%")
        print(f"GPU Std Dev: {statistics.stdev(gpu_samples):.1f}%")

if __name__ == "__main__":
    main()