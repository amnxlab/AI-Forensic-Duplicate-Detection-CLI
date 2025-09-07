# tracker.py
# Live monitoring module for real-time file duplicate/tampering alerts

import os
import time
from datetime import datetime

import numpy as np  # kept if scan_duplicates needs it; otherwise safe to remove
from loader import daily_snapshot, scan_duplicates

CHECK_INTERVAL = 30  # seconds between checks (can be adjusted)

# Base paths (tracker-specific logs still in reports/)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
ALERT_LOG = os.path.join(BASE_DIR, "reports", "tracker_alerts.txt")

# Use the snapshot dir defined by daily_snapshot so files land in the same place
SNAPSHOT_DIR = daily_snapshot.SNAPSHOT_DIR
BASELINE_NAME = "tracker_baseline_snapshot.txt"
BASELINE_PATH = os.path.join(SNAPSHOT_DIR, BASELINE_NAME)


def status(msg):
    print(f"[*] {msg}")


def info(msg):
    print(f"[+] {msg}")


def warning(msg):
    print(f"[!] {msg}")


def _ensure_reports_dirs():
    # daily_snapshot already ensures its own dirs, but no harm in double safety
    os.makedirs(os.path.dirname(ALERT_LOG), exist_ok=True)
    os.makedirs(SNAPSHOT_DIR, exist_ok=True)


def monitor_folder(folder_path):
    _ensure_reports_dirs()

    status(f"Tracker started on folder: {folder_path}")
    status(f"Checking every {CHECK_INTERVAL} seconds\n")

    # 1) Build or load baseline snapshot (stored under reports/snapshots/)
    if not os.path.exists(BASELINE_PATH):
        info("No previous baseline snapshot found. Generating baseline...")
        baseline = daily_snapshot.generate_snapshot(folder_path)
        daily_snapshot.save_snapshot(baseline, BASELINE_NAME)
        info(f"Baseline snapshot saved: {BASELINE_PATH}")
        status("Waiting for changes...")
        time.sleep(CHECK_INTERVAL)

    baseline = daily_snapshot.load_snapshot(BASELINE_NAME)
    if baseline is None:
        warning("Failed to load baseline snapshot (None returned). Regenerating.")
        baseline = daily_snapshot.generate_snapshot(folder_path)
        daily_snapshot.save_snapshot(baseline, BASELINE_NAME)

    # 2) Main loop
    while True:
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            current_snapshot = daily_snapshot.generate_snapshot(folder_path)

            # Use byte-level + optional embedding compare (sim_threshold just below 1.0)
            changes = daily_snapshot.compare_snapshots(
                baseline, current_snapshot, sim_threshold=0.999999
            )

            if not changes:
                status(f"{timestamp}: No snapshot changes.")
                time.sleep(CHECK_INTERVAL)
                continue

            # If we got here, snapshot changed. Optionally rescan duplicates.
            warning(f"{timestamp}: Snapshot changed. Re-scanning duplicates...")
            try:
                new_duplicates = scan_duplicates.scan_folder_for_duplicates(folder_path)
            except Exception as e:
                new_duplicates = []
                warning(f"Duplicate scan failed: {e}")

            # Build set of old duplicate paths (not persisted previously; keep simple)
            # If you later persist baseline duplicates, load them here to compute delta properly.
            old_dup_paths = set()
            new_only = []
            for f1, f2, tag in new_duplicates:
                if f1 not in old_dup_paths or f2 not in old_dup_paths:
                    new_only.append((f1, f2, tag))

            # Log alert
            with open(ALERT_LOG, "a") as f:
                f.write(f"\n=== ALERT [{timestamp}] ===\n")
                if changes:
                    f.write("[Snapshot Changes Detected]:\n")
                    for path, change_type in changes:
                        f.write(f"{path} ==> {change_type}\n")
                if new_only:
                    f.write("[New Duplicate Files Detected]:\n")
                    for f1, f2, tag in new_only:
                        f.write(f"{tag}:\n → {f1}\n → {f2}\n")

            warning(f"ALERT logged ({len(changes)} changes, {len(new_only)} new duplicates).")

            # Update baseline to current snapshot
            daily_snapshot.save_snapshot(current_snapshot, BASELINE_NAME)
            baseline = current_snapshot

        except Exception as e:
            warning(f"Error during monitoring: {e}")

        time.sleep(CHECK_INTERVAL)


# CLI entry point
if __name__ == "__main__":
    import argparse
    # 'keyboard' was imported before but never used; removing it avoids extra dependency.
    # If you actually want ESC-to-stop, handle it differently or re-add with proper try/except.

    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", required=True, help="Target folder to monitor continuously")
    args = parser.parse_args()

    try:
        info("Press Ctrl+C to stop the tracker.")
        monitor_folder(args.folder)
    except KeyboardInterrupt:
        print("\n[!] Tracker stopped by user.")