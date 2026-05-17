"""
performance_monitor.py  —  ARYAN  (Task 4)
Tracks speed, memory, and generates performance logs + reports.
"""

import time
import os
import json
import psutil
import platform
from datetime import datetime
from dataclasses import dataclass, field, asdict


LOG_FILE    = "results/performance_log.json"
REPORT_FILE = "results/performance_report.txt"


@dataclass
class PerformanceSnapshot:
    timestamp:       str
    batch_num:       int
    traits_processed:int
    batch_time_sec:  float
    traits_per_sec:  float
    memory_mb:       float
    cpu_percent:     float
    cumulative_total:int = 0


class PerformanceMonitor:
    """
    Tracks speed, memory, CPU during batch simulation.
    Call .start() before simulation, .record() after each batch,
    .report() at the end.
    """

    def __init__(self):
        os.makedirs("results", exist_ok=True)
        self.snapshots   = []
        self.start_time  = None
        self.process     = psutil.Process()
        self.total_traits= 0

    def start(self):
        self.start_time = time.perf_counter()
        print(f"📊 Performance monitor started")
        print(f"   Platform : {platform.system()} {platform.machine()}")
        print(f"   CPU cores: {psutil.cpu_count()}")
        print(f"   RAM total: {psutil.virtual_memory().total / 1e9:.1f} GB")

    def record(self, batch_num: int, batch_size: int, elapsed_sec: float):
        """Call after each batch completes."""
        self.total_traits += batch_size
        mem_mb     = self.process.memory_info().rss / 1e6
        cpu_pct    = self.process.cpu_percent(interval=0.1)
        traits_sec = batch_size / elapsed_sec if elapsed_sec > 0 else 0

        snap = PerformanceSnapshot(
            timestamp        = datetime.now().isoformat(),
            batch_num        = batch_num,
            traits_processed = batch_size,
            batch_time_sec   = round(elapsed_sec, 4),
            traits_per_sec   = round(traits_sec, 1),
            memory_mb        = round(mem_mb, 1),
            cpu_percent      = round(cpu_pct, 1),
            cumulative_total = self.total_traits,
        )
        self.snapshots.append(snap)
        return snap

    def report(self) -> dict:
        """Generate final performance report."""
        if not self.snapshots:
            return {"error": "No data recorded"}

        total_elapsed = time.perf_counter() - self.start_time if self.start_time else 0
        speeds        = [s.traits_per_sec for s in self.snapshots]
        memories      = [s.memory_mb for s in self.snapshots]

        avg_speed   = sum(speeds) / len(speeds)
        peak_mem    = max(memories)
        est_1m_hrs  = (1_000_000 / avg_speed / 3600) if avg_speed > 0 else 999

        summary = {
            "total_traits":     self.total_traits,
            "total_elapsed_sec":round(total_elapsed, 2),
            "avg_speed_per_sec":round(avg_speed, 1),
            "peak_memory_mb":   round(peak_mem, 1),
            "est_1M_hours":     round(est_1m_hrs, 2),
            "target_4hr_met":   est_1m_hrs <= 4.0,
            "batches_recorded": len(self.snapshots),
        }

        # Save JSON log
        with open(LOG_FILE, "w") as f:
            json.dump({
                "summary":   summary,
                "snapshots": [asdict(s) for s in self.snapshots]
            }, f, indent=2)

        # Save text report
        lines = [
            "=" * 55,
            "  ARYAN — PERFORMANCE REPORT",
            f"  Generated: {datetime.now().isoformat()[:19]}",
            "=" * 55,
            f"  Total traits processed : {self.total_traits:,}",
            f"  Total time             : {total_elapsed:.1f}s",
            f"  Average speed          : {avg_speed:,.0f} traits/sec",
            f"  Peak memory usage      : {peak_mem:.0f} MB",
            f"  Est. time for 1M traits: {est_1m_hrs:.2f} hours",
            f"  4-hour target met      : {'YES' if est_1m_hrs <= 4 else 'NO - needs GPU'}",
            "=" * 55,
        ]
        with open(REPORT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

        print("\n".join(lines))
        print(f"\n📄 Log saved:    {LOG_FILE}")
        print(f"📄 Report saved: {REPORT_FILE}")
        return summary


if __name__ == "__main__":
    import random
    monitor = PerformanceMonitor()
    monitor.start()

    for i in range(1, 6):
        time.sleep(0.05)
        monitor.record(batch_num=i, batch_size=5000, elapsed_sec=0.05 + random.uniform(0, 0.02))

    monitor.report()
