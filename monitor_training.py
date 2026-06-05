#!/usr/bin/env python
"""
monitor_training.py

Script per monitorare il progresso del training in tempo reale.
Mostra log Celery, progress trial, e statistiche aggiornate.
"""

import os
import sys
import time
import json
import glob
from datetime import datetime
from collections import defaultdict

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

try:
    from state_manager import PERSISTENT_PATHS
except ImportError:
    # Fallback if can't import
    PERSISTENT_PATHS = {
        "trials_output_dir": "processed_data/trials",
        "logs_dir": "logs"
    }

def clear_screen():
    """Clear terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_celery_log():
    """Find the most recent Celery worker log."""
    log_dir = PERSISTENT_PATHS.get("logs_dir", "logs")
    pattern = os.path.join(log_dir, "celery_worker_*.log")
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def tail_file(filepath, num_lines=20):
    """Read last N lines of a file."""
    if not os.path.exists(filepath):
        return []
    
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
        return lines[-num_lines:]

def get_recent_trials(limit=10):
    """Get most recent trial results."""
    trials_dir = PERSISTENT_PATHS.get("trials_output_dir", "processed_data/trials")
    if not os.path.exists(trials_dir):
        return []
    
    pattern = os.path.join(trials_dir, "*.json")
    files = glob.glob(pattern)
    
    # Sort by modification time
    files.sort(key=os.path.getmtime, reverse=True)
    
    trials = []
    for filepath in files[:limit]:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
                mtime = os.path.getmtime(filepath)
                data['_filepath'] = filepath
                data['_modified'] = datetime.fromtimestamp(mtime).strftime('%Y-%m-%d %H:%M:%S')
                trials.append(data)
        except:
            continue
    
    return trials

def get_training_stats():
    """Get statistics from all trials."""
    trials_dir = PERSISTENT_PATHS.get("trials_output_dir", "processed_data/trials")
    if not os.path.exists(trials_dir):
        return None
    
    pattern = os.path.join(trials_dir, "*.json")
    files = glob.glob(pattern)
    
    stats = {
        'total_trials': len(files),
        'by_architecture': defaultdict(int),
        'by_status': defaultdict(int),
        'best_accuracy': 0.0,
        'best_trial': None,
        'recent_count_1h': 0,
        'recent_count_24h': 0
    }
    
    now = time.time()
    one_hour_ago = now - 3600
    one_day_ago = now - 86400
    
    for filepath in files:
        try:
            mtime = os.path.getmtime(filepath)
            
            if mtime > one_hour_ago:
                stats['recent_count_1h'] += 1
            if mtime > one_day_ago:
                stats['recent_count_24h'] += 1
            
            with open(filepath, 'r') as f:
                data = json.load(f)
                
                arch = data.get('architecture', 'unknown')
                stats['by_architecture'][arch] += 1
                
                status = data.get('status', 'unknown')
                stats['by_status'][status] += 1
                
                acc = data.get('mean_accuracy', 0.0)
                if acc > stats['best_accuracy']:
                    stats['best_accuracy'] = acc
                    stats['best_trial'] = os.path.basename(filepath)
        except:
            continue
    
    return stats

def format_trial_row(trial, width=120):
    """Format a single trial for display."""
    status = trial.get('status', 'unknown')
    acc = trial.get('mean_accuracy', 0.0)
    arch = trial.get('architecture', 'unknown')
    manifest = trial.get('manifest_name', 'unknown')
    modified = trial.get('_modified', 'unknown')
    
    # Status icon
    status_icon = {
        'success': '✅',
        'error': '❌',
        'running': '🔄',
        'pending': '⏳'
    }.get(status, '❓')
    
    # Format accuracy with color
    if acc >= 0.95:
        acc_str = f"\033[92m{acc:.4f}\033[0m"  # Green
    elif acc >= 0.90:
        acc_str = f"\033[93m{acc:.4f}\033[0m"  # Yellow
    else:
        acc_str = f"{acc:.4f}"
    
    return f"{status_icon} {modified} | {arch:15s} | {manifest:20s} | Acc: {acc_str}"

def monitor_loop(refresh_interval=5):
    """Main monitoring loop."""
    print("🔍 Training Monitor Started")
    print(f"Refresh interval: {refresh_interval}s")
    print("Press Ctrl+C to exit\n")
    
    try:
        while True:
            clear_screen()
            
            # Header
            print("=" * 120)
            print(f"🚀 SERS Training Monitor - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 120)
            
            # Statistics
            stats = get_training_stats()
            if stats:
                print("\n📊 STATISTICS")
                print("-" * 120)
                print(f"Total Trials: {stats['total_trials']}")
                print(f"Last 1 hour: {stats['recent_count_1h']} | Last 24h: {stats['recent_count_24h']}")
                print(f"Best Accuracy: {stats['best_accuracy']:.4f} ({stats['best_trial']})")
                
                print("\nBy Architecture:")
                for arch, count in sorted(stats['by_architecture'].items(), key=lambda x: -x[1]):
                    print(f"  - {arch}: {count} trials")
                
                print("\nBy Status:")
                for status, count in sorted(stats['by_status'].items()):
                    print(f"  - {status}: {count} trials")
            
            # Recent trials
            print("\n📋 RECENT TRIALS (Last 10)")
            print("-" * 120)
            trials = get_recent_trials(limit=10)
            if trials:
                for trial in trials:
                    print(format_trial_row(trial))
            else:
                print("No trials found yet.")
            
            # Celery worker log (last 15 lines)
            print("\n📜 CELERY WORKER LOG (Last 15 lines)")
            print("-" * 120)
            celery_log = get_latest_celery_log()
            if celery_log:
                log_lines = tail_file(celery_log, num_lines=15)
                for line in log_lines:
                    # Highlight errors and warnings
                    line = line.rstrip()
                    if 'ERROR' in line or 'Error' in line:
                        print(f"\033[91m{line}\033[0m")  # Red
                    elif 'WARNING' in line or 'Warning' in line:
                        print(f"\033[93m{line}\033[0m")  # Yellow
                    elif 'SUCCESS' in line or 'Succeeded' in line:
                        print(f"\033[92m{line}\033[0m")  # Green
                    else:
                        print(line)
            else:
                print("No Celery log found. Check if worker is running.")
            
            # Footer
            print("\n" + "=" * 120)
            print(f"Next update in {refresh_interval}s... (Ctrl+C to exit)")
            
            time.sleep(refresh_interval)
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitor stopped by user.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor SERS training progress in real-time")
    parser.add_argument('--interval', type=int, default=5, help='Refresh interval in seconds (default: 5)')
    parser.add_argument('--trials', type=int, default=10, help='Number of recent trials to show (default: 10)')
    
    args = parser.parse_args()
    
    monitor_loop(refresh_interval=args.interval)
