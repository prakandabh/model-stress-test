#!/usr/bin/env python3
"""
gpu_logger.py - Log GPU + CPU + RAM + VRAM metrics at a fixed sampling
interval to CSV. Fully standalone: it does NOT launch or depend on any
workload. Start it, run your load however you like (e.g. a JMeter test
triggered from a different machine's UI), then stop it.

Columns logged:
    - GPU utilization (%)
    - VRAM usage (used GB / total GB / %)
    - CPU utilization (%)
    - RAM usage (used GB / total GB / %)
    - Power draw (W)
    - Per-process GPU SM %, VRAM MB, CPU %, RAM MB (when --pid is given)

Usage examples:
    # Log everything, name the output yourself, stop with Ctrl+C when your
    # JMeter test finishes
    python gpu_logger.py --interval-ms 250 --name asr_load_test1

    # Same, but also scope per-process columns to your already-running
    # pipeline server (find its PID first: pgrep -f pipeline.py)
    python gpu_logger.py --interval-ms 250 --name asr_load_test1 --pid 48213

    # Stop automatically after a fixed duration instead of Ctrl+C
    # (useful if you know how long the JMeter test plan runs)
    python gpu_logger.py --interval-ms 250 --name asr_load_test1 --duration-s 120

    # Or give the exact output path yourself instead of --name
    python gpu_logger.py --interval-ms 250 --output gpu_metrics/asr_run1_250ms.csv

Requires: pip install nvidia-ml-py psutil --break-system-packages
"""

import argparse
import csv
import os
import signal
import sys
import time
from datetime import datetime

try:
    import pynvml
except ImportError:
    print("ERROR: pynvml not installed. Run: pip install nvidia-ml-py --break-system-packages",
          file=sys.stderr)
    sys.exit(1)

try:
    import psutil
except ImportError:
    print("ERROR: psutil not installed. Run: pip install psutil --break-system-packages",
          file=sys.stderr)
    sys.exit(1)


GB = 1024 ** 3
MB = 1024 ** 2

FIELDNAMES = [
    "timestamp",
    "elapsed_s",
    "interval_ms",
    "gpu_id",
    "pid",
    # GPU
    "gpu_util_percent",
    "vram_used_gb",
    "vram_total_gb",
    "vram_percent",
    "power_w",
    # System CPU / RAM
    "cpu_percent",
    "ram_used_gb",
    "ram_total_gb",
    "ram_percent",
    # Per-process (only populated when --pid / a detected process is tracked)
    "process_gpu_sm_percent",
    "process_vram_mb",
    "process_cpu_percent",
    "process_ram_mb",
]

_stop = False


def _handle_sigint(signum, frame):
    global _stop
    _stop = True


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def interval_label(interval_ms):
    return f"{interval_ms//1000}s" if interval_ms >= 1000 else f"{interval_ms}ms"


def get_process_sm_util(handle, target_pid=None):
    """
    Returns {pid: sm_util_percent} using NVML's per-process utilization
    sampling. NOT supported on all GPUs/drivers -- falls back to {} if
    unavailable.
    """
    try:
        samples = pynvml.nvmlDeviceGetProcessUtilization(handle, 0)
    except pynvml.NVMLError:
        return {}
    result = {}
    for s in samples:
        if target_pid is not None and s.pid != target_pid:
            continue
        result[s.pid] = s.smUtil
    return result


def get_process_vram_mb(handle, target_pid=None):
    """Returns {pid: used_vram_mb} from the running-process list."""
    result = {}
    procs = []
    try:
        procs += pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
    except pynvml.NVMLError:
        pass
    try:
        procs += pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
    except pynvml.NVMLError:
        pass
    for p in procs:
        if target_pid is not None and p.pid != target_pid:
            continue
        mem = p.usedGpuMemory
        result[p.pid] = (mem / MB) if mem not in (None, 0xFFFFFFFFFFFFFFFF) else 0
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Log GPU/CPU/RAM/VRAM metrics at a fixed sampling interval. "
                    "Standalone -- does not launch or require a workload command."
    )
    parser.add_argument("--interval-ms", type=int, required=True,
                         help="Sampling interval in ms (e.g. 100, 250, 500, 1000, 5000)")
    parser.add_argument("--name", default=None,
                         help="Test name for the output file, e.g. --name asr_load_test1 "
                              "-> gpu_metrics/asr_load_test1_250ms_<timestamp>.csv")
    parser.add_argument("--output", default=None,
                         help="Exact output CSV path. Overrides --name if both are given.")
    parser.add_argument("--duration-s", type=float, default=None,
                         help="Stop after N seconds (default: run until Ctrl+C or --watch-pid exits)")
    parser.add_argument("--gpu-id", type=int, default=None,
                         help="Only log this GPU index (default: all GPUs)")
    parser.add_argument("--pid", type=int, default=None,
                         help="Track this PID's per-process GPU SM%%, VRAM, CPU%%, RAM "
                              "(e.g. the PID of your already-running pipeline server -- "
                              "find it with: pgrep -f pipeline.py)")
    parser.add_argument("--watch-pid", type=int, default=None,
                         help="Automatically stop logging once this PID exits "
                              "(only useful if the tracked process is expected to end -- "
                              "leave unset for a long-running server)")
    args = parser.parse_args()

    if not args.output and not args.name:
        parser.error("Provide either --name (auto-named file) or --output (exact path).")

    if not args.output:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"gpu_metrics/{args.name}_{interval_label(args.interval_ms)}_{ts}.csv"

    signal.signal(signal.SIGINT, _handle_sigint)
    signal.signal(signal.SIGTERM, _handle_sigint)

    pynvml.nvmlInit()
    device_count = pynvml.nvmlDeviceGetCount()
    gpu_ids = [args.gpu_id] if args.gpu_id is not None else list(range(device_count))
    handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i) for i in gpu_ids}

    # Set up per-process CPU tracking (psutil requires a throwaway priming
    # call -- the first cpu_percent() reading is always meaningless).
    proc = None
    if args.pid is not None:
        try:
            proc = psutil.Process(args.pid)
            proc.cpu_percent(interval=None)
        except psutil.NoSuchProcess:
            print(f"WARNING: PID {args.pid} not found at startup -- per-process columns will be blank "
                  f"unless it appears later.", file=sys.stderr)
    psutil.cpu_percent(interval=None)  # prime system-wide CPU reading

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir or ".", exist_ok=True)
    new_file = not os.path.exists(args.output)

    print(f"Logging GPU {gpu_ids} + CPU/RAM at {args.interval_ms}ms -> {args.output}")
    if args.pid:
        print(f"Tracking PID {args.pid} for per-process columns")
    if args.duration_s:
        print(f"Will stop after {args.duration_s}s")
    elif args.watch_pid:
        print(f"Will stop when PID {args.watch_pid} exits")
    else:
        print("Run your test now. Press Ctrl+C here when it's done to stop logging.")
    print()

    start = time.monotonic()
    interval_s = args.interval_ms / 1000.0

    with open(args.output, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            writer.writeheader()

        next_tick = time.monotonic()
        while not _stop:
            now = time.monotonic()
            elapsed = now - start

            if args.duration_s is not None and elapsed >= args.duration_s:
                break
            if args.watch_pid is not None and not pid_alive(args.watch_pid):
                print(f"PID {args.watch_pid} exited, stopping.")
                break

            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            # --- System-wide CPU / RAM (same for every row this tick) ---
            cpu_percent = psutil.cpu_percent(interval=None)
            vm = psutil.virtual_memory()
            ram_used_gb = vm.used / GB
            ram_total_gb = vm.total / GB
            ram_percent = vm.percent

            # --- Per-process CPU / RAM ---
            process_cpu_percent = ""
            process_ram_mb = ""
            if proc is not None:
                try:
                    process_cpu_percent = round(proc.cpu_percent(interval=None), 1)
                    process_ram_mb = round(proc.memory_info().rss / MB, 1)
                except psutil.NoSuchProcess:
                    proc = None

            for gpu_id in gpu_ids:
                handle = handles[gpu_id]
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_util = util.gpu
                except pynvml.NVMLError:
                    gpu_util = ""

                try:
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    vram_used_gb = meminfo.used / GB
                    vram_total_gb = meminfo.total / GB
                    vram_percent = (meminfo.used / meminfo.total * 100) if meminfo.total else ""
                except pynvml.NVMLError:
                    vram_used_gb = vram_total_gb = vram_percent = ""

                try:
                    power_w = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                except pynvml.NVMLError:
                    power_w = ""

                sm_by_pid = get_process_sm_util(handle, args.pid)
                vram_by_pid = get_process_vram_mb(handle, args.pid)
                pids = set(sm_by_pid) | set(vram_by_pid)
                if args.pid is not None:
                    pids = {args.pid}

                base_row = {
                    "timestamp": ts,
                    "elapsed_s": round(elapsed, 3),
                    "interval_ms": args.interval_ms,
                    "gpu_id": gpu_id,
                    "gpu_util_percent": gpu_util,
                    "vram_used_gb": round(vram_used_gb, 3) if vram_used_gb != "" else "",
                    "vram_total_gb": round(vram_total_gb, 3) if vram_total_gb != "" else "",
                    "vram_percent": round(vram_percent, 1) if vram_percent != "" else "",
                    "power_w": round(power_w, 1) if power_w != "" else "",
                    "cpu_percent": cpu_percent,
                    "ram_used_gb": round(ram_used_gb, 3),
                    "ram_total_gb": round(ram_total_gb, 3),
                    "ram_percent": ram_percent,
                }

                if pids:
                    for pid in pids:
                        row = dict(base_row)
                        row.update({
                            "pid": pid,
                            "process_gpu_sm_percent": sm_by_pid.get(pid, ""),
                            "process_vram_mb": round(vram_by_pid.get(pid, 0), 1),
                            "process_cpu_percent": process_cpu_percent if pid == args.pid else "",
                            "process_ram_mb": process_ram_mb if pid == args.pid else "",
                        })
                        writer.writerow(row)
                else:
                    # No per-process GPU detail available -- log device/system-level row only
                    row = dict(base_row)
                    row.update({
                        "pid": "",
                        "process_gpu_sm_percent": "",
                        "process_vram_mb": "",
                        "process_cpu_percent": "",
                        "process_ram_mb": "",
                    })
                    writer.writerow(row)

            f.flush()

            next_tick += interval_s
            sleep_time = next_tick - time.monotonic()
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                next_tick = time.monotonic()  # fell behind; resync to avoid drift

    pynvml.nvmlShutdown()
    print(f"\nDone. Wrote to {args.output}")


if __name__ == "__main__":
    main()