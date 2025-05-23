#!/usr/bin/env python3
"""
Performance analysis script for AlphaZero self-play logs
"""
import re
import sys
from collections import defaultdict
import statistics

def parse_log_file(filename):
    """Parse the AlphaZero log file and extract performance metrics."""
    metrics = defaultdict(list)
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Extract batch processing times
    batch_pattern = r'✅ Batch \d+: (\d+) states in ([\d.]+)ms'
    for match in re.finditer(batch_pattern, content):
        batch_size = int(match.group(1))
        time_ms = float(match.group(2))
        metrics['batch_times'].append(time_ms)
        metrics['batch_sizes'].append(batch_size)
    
    # Extract throughput
    throughput_pattern = r'Final throughput: ([\d.]+) sims/sec'
    for match in re.finditer(throughput_pattern, content):
        throughput = float(match.group(1))
        metrics['throughput'].append(throughput)
    
    # Extract search completion times
    search_pattern = r'✅ Search completed in (\d+)ms'
    for match in re.finditer(search_pattern, content):
        search_time = int(match.group(1))
        metrics['search_times'].append(search_time)
    
    # Extract GPU utilization estimates
    gpu_pattern = r'Est\. GPU: ([\d.]+)%'
    for match in re.finditer(gpu_pattern, content):
        gpu_util = float(match.group(1))
        metrics['gpu_estimates'].append(gpu_util)
    
    return metrics

def analyze_metrics(metrics):
    """Analyze and print performance statistics."""
    print("🔍 AlphaZero Performance Analysis")
    print("=" * 50)
    
    if metrics['batch_times']:
        print(f"\n📊 Batch Processing:")
        print(f"  Average batch time: {statistics.mean(metrics['batch_times']):.2f}ms")
        print(f"  Min batch time: {min(metrics['batch_times']):.2f}ms")
        print(f"  Max batch time: {max(metrics['batch_times']):.2f}ms")
        print(f"  Std deviation: {statistics.stdev(metrics['batch_times']):.2f}ms")
    
    if metrics['throughput']:
        print(f"\n🚀 Throughput:")
        print(f"  Average: {statistics.mean(metrics['throughput']):.2f} sims/sec")
        print(f"  Min: {min(metrics['throughput']):.2f} sims/sec")
        print(f"  Max: {max(metrics['throughput']):.2f} sims/sec")
    
    if metrics['search_times']:
        print(f"\n⏱️  Search Times:")
        print(f"  Average: {statistics.mean(metrics['search_times']):.0f}ms")
        print(f"  Min: {min(metrics['search_times'])}ms")
        print(f"  Max: {max(metrics['search_times'])}ms")
    
    if metrics['gpu_estimates']:
        print(f"\n🎮 GPU Utilization (Estimated):")
        print(f"  Average: {statistics.mean(metrics['gpu_estimates']):.1f}%")
        print(f"  Min: {min(metrics['gpu_estimates']):.1f}%")
        print(f"  Max: {max(metrics['gpu_estimates']):.1f}%")
    
    # Performance rating
    avg_throughput = statistics.mean(metrics['throughput']) if metrics['throughput'] else 0
    print(f"\n🏆 Performance Rating:")
    if avg_throughput >= 190:
        print(f"  ✅ EXCELLENT - Target achieved! ({avg_throughput:.0f} sims/sec)")
    elif avg_throughput >= 150:
        print(f"  🔶 GOOD - Close to target ({avg_throughput:.0f} sims/sec)")
    else:
        print(f"  ❌ NEEDS IMPROVEMENT ({avg_throughput:.0f} sims/sec)")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        filename = sys.argv[1]
    else:
        # Try to find the most recent log
        import glob
        import os
        logs = glob.glob("*.log")
        if logs:
            filename = max(logs, key=os.path.getctime)
            print(f"Using most recent log: {filename}\n")
        else:
            print("Usage: python analyze_performance.py <logfile>")
            sys.exit(1)
    
    metrics = parse_log_file(filename)
    analyze_metrics(metrics)