#!/usr/bin/env python3
import subprocess
import time
import sys

def get_gpu_stats():
    """Get GPU utilization and memory stats using nvidia-smi"""
    try:
        result = subprocess.run([
            'nvidia-smi', 
            '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw',
            '--format=csv,noheader,nounits'
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            stats = result.stdout.strip().split(', ')
            return {
                'gpu_util': float(stats[0]),
                'memory_used': float(stats[1]),
                'memory_total': float(stats[2]),
                'temperature': float(stats[3]),
                'power': float(stats[4])
            }
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
    return None

def main():
    print("GPU Utilization Monitor - Press Ctrl+C to stop")
    print("=" * 60)
    print(f"{'Time':8} {'GPU%':6} {'Mem Used':10} {'Mem Total':10} {'Temp':6} {'Power':8}")
    print("=" * 60)
    
    samples = []
    start_time = time.time()
    
    try:
        while True:
            stats = get_gpu_stats()
            if stats:
                current_time = time.time() - start_time
                print(f"{current_time:8.1f} {stats['gpu_util']:6.1f} "
                      f"{stats['memory_used']:10.0f} {stats['memory_total']:10.0f} "
                      f"{stats['temperature']:6.1f} {stats['power']:8.1f}")
                samples.append(stats['gpu_util'])
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n" + "=" * 60)
        if samples:
            avg_util = sum(samples) / len(samples)
            max_util = max(samples)
            min_util = min(samples)
            print(f"Summary: Avg={avg_util:.1f}%, Max={max_util:.1f}%, Min={min_util:.1f}%")
        print("Monitor stopped.")

if __name__ == "__main__":
    main()