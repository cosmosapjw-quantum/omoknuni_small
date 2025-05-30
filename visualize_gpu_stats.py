#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import sys
import os

def visualize_gpu_stats(csv_file):
    """Visualize GPU statistics from nvidia-smi monitoring."""
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Parse values
    if 'utilization.gpu [%]' in df.columns:
        df['gpu_util'] = df['utilization.gpu [%]'].str.strip(' %').astype(float)
    if 'utilization.memory [%]' in df.columns:
        df['mem_util'] = df['utilization.memory [%]'].str.strip(' %').astype(float)
    if 'memory.used [MiB]' in df.columns:
        df['mem_used'] = df['memory.used [MiB]'].str.strip(' MiB').astype(float)
    if 'temperature.gpu' in df.columns:
        df['temp'] = df['temperature.gpu'].astype(float)
    
    # Create timestamp index
    df['time'] = pd.to_datetime(df['timestamp'])
    df['seconds'] = (df['time'] - df['time'].iloc[0]).dt.total_seconds()
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('GPU Utilization During MCTS Benchmark', fontsize=16)
    
    # GPU Utilization
    if 'gpu_util' in df.columns:
        ax = axes[0, 0]
        ax.plot(df['seconds'], df['gpu_util'], 'b-', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('GPU Utilization (%)')
        ax.set_title('GPU Core Utilization')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        # Add average line
        avg_gpu = df['gpu_util'].mean()
        ax.axhline(y=avg_gpu, color='r', linestyle='--', 
                   label=f'Average: {avg_gpu:.1f}%')
        ax.legend()
    
    # Memory Utilization
    if 'mem_util' in df.columns:
        ax = axes[0, 1]
        ax.plot(df['seconds'], df['mem_util'], 'g-', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Utilization (%)')
        ax.set_title('GPU Memory Utilization')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 105)
        
        avg_mem = df['mem_util'].mean()
        ax.axhline(y=avg_mem, color='r', linestyle='--', 
                   label=f'Average: {avg_mem:.1f}%')
        ax.legend()
    
    # Memory Usage
    if 'mem_used' in df.columns:
        ax = axes[1, 0]
        ax.plot(df['seconds'], df['mem_used'] / 1024, 'm-', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Memory Used (GB)')
        ax.set_title('GPU Memory Usage')
        ax.grid(True, alpha=0.3)
        
        avg_mem_gb = df['mem_used'].mean() / 1024
        ax.axhline(y=avg_mem_gb, color='r', linestyle='--', 
                   label=f'Average: {avg_mem_gb:.1f} GB')
        ax.legend()
    
    # Temperature
    if 'temp' in df.columns:
        ax = axes[1, 1]
        ax.plot(df['seconds'], df['temp'], 'r-', linewidth=2)
        ax.set_xlabel('Time (seconds)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title('GPU Temperature')
        ax.grid(True, alpha=0.3)
        
        avg_temp = df['temp'].mean()
        ax.axhline(y=avg_temp, color='b', linestyle='--', 
                   label=f'Average: {avg_temp:.1f}°C')
        ax.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_file = csv_file.replace('.csv', '_visualization.png')
    plt.savefig(output_file, dpi=150)
    print(f"Visualization saved to: {output_file}")
    
    # Print statistics
    print("\nGPU Statistics Summary:")
    print("-" * 40)
    if 'gpu_util' in df.columns:
        print(f"GPU Utilization:")
        print(f"  Average: {df['gpu_util'].mean():.1f}%")
        print(f"  Peak: {df['gpu_util'].max():.1f}%")
        print(f"  Min: {df['gpu_util'].min():.1f}%")
    
    if 'mem_util' in df.columns:
        print(f"\nMemory Utilization:")
        print(f"  Average: {df['mem_util'].mean():.1f}%")
        print(f"  Peak: {df['mem_util'].max():.1f}%")
    
    if 'mem_used' in df.columns:
        print(f"\nMemory Usage:")
        print(f"  Average: {df['mem_used'].mean() / 1024:.2f} GB")
        print(f"  Peak: {df['mem_used'].max() / 1024:.2f} GB")
    
    if 'temp' in df.columns:
        print(f"\nTemperature:")
        print(f"  Average: {df['temp'].mean():.1f}°C")
        print(f"  Peak: {df['temp'].max():.1f}°C")
    
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python visualize_gpu_stats.py <gpu_stats.csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    
    visualize_gpu_stats(csv_file)