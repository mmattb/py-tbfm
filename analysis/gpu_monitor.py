#!/usr/bin/env python3
"""
GPU monitoring script - sends Pushover notifications about GPU status.

Usage:
    # Check GPU status once
    python gpu_monitor.py

    # Monitor continuously (check every 5 minutes)
    python gpu_monitor.py --interval 300

    # Alert when GPU memory usage drops below threshold (job finished)
    python gpu_monitor.py --alert-on-free --threshold 90
"""

import argparse
import os
import sys
import subprocess
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import notifications


def get_gpu_status():
    """Get GPU status using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line:
                idx, name, util, mem_used, mem_total, temp = line.split(', ')
                gpus.append({
                    'id': int(idx),
                    'name': name,
                    'utilization': int(util),
                    'memory_used': int(mem_used),
                    'memory_total': int(mem_total),
                    'memory_percent': int(mem_used) / int(mem_total) * 100,
                    'temperature': int(temp)
                })
        
        return gpus
    except Exception as e:
        return None


def format_gpu_status(gpus):
    """Format GPU status as a readable message."""
    if not gpus:
        return "❌ Could not read GPU status"
    
    lines = []
    for gpu in gpus:
        lines.append(f"GPU {gpu['id']}: {gpu['name']}")
        lines.append(f"  Utilization: {gpu['utilization']}%")
        lines.append(f"  Memory: {gpu['memory_used']}MB / {gpu['memory_total']}MB ({gpu['memory_percent']:.1f}%)")
        lines.append(f"  Temp: {gpu['temperature']}°C")
        lines.append("")
    
    return "\n".join(lines)


def check_gpu_free(gpus, threshold=90):
    """Check if any GPU has memory usage below threshold."""
    if not gpus:
        return False
    
    for gpu in gpus:
        if gpu['memory_percent'] < threshold:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description='GPU monitoring with Pushover notifications')
    parser.add_argument('--interval', type=int, default=0,
                       help='Check interval in seconds (0 = check once and exit)')
    parser.add_argument('--alert-on-free', action='store_true',
                       help='Send alert when GPU memory usage drops below threshold')
    parser.add_argument('--threshold', type=float, default=90,
                       help='Memory usage threshold percentage for alerts (default: 90)')
    parser.add_argument('--quiet', action='store_true',
                       help='Only send notifications on alerts, not status updates')
    
    args = parser.parse_args()
    
    if not notifications.is_enabled():
        print("⚠️  Notifications not enabled. Set NOTIFY_METHOD to receive alerts.")
        print("Status will be printed to console only.\n")
    
    alerted = False  # Track if we've already sent a "GPU free" alert
    
    while True:
        gpus = get_gpu_status()
        status_msg = format_gpu_status(gpus)
        
        if args.alert_on_free and check_gpu_free(gpus, args.threshold) and not alerted:
            # GPU became available
            notifications.notify_training_complete(
                model_name="GPU Monitor",
                metrics={},
                output_dir=None
            )
            notifications._notifier.send(
                "🎉 GPU Available!",
                f"GPU memory usage dropped below {args.threshold}%\n\n{status_msg}",
                is_error=False
            )
            alerted = True
            print(f"[Alert Sent] GPU available\n{status_msg}")
        elif not args.quiet and notifications.is_enabled():
            # Regular status update
            notifications._notifier.send(
                "📊 GPU Status",
                status_msg,
                is_error=False
            )
            print(f"[Status Sent]\n{status_msg}")
        else:
            # Just print to console
            print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}]")
            print(status_msg)
        
        # Reset alert flag if GPU is busy again
        if alerted and not check_gpu_free(gpus, args.threshold):
            alerted = False
            print("[GPU busy again, will alert on next free]")
        
        if args.interval <= 0:
            break
        
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
