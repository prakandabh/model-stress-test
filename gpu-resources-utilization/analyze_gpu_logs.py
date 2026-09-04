#!/usr/bin/env python3
"""
analyze_gpu_logs.py - Compute summary statistics (mean/p95/peak) for GPU,
VRAM, CPU, and RAM from the CSV files produced by gpu_logger.py, and print
the sampling-interval comparison tables.

Usage:
    python analyze_gpu_logs.py gpu_metrics/*.csv
"""
import argparse
import csv
import glob
import sys

# (CSV column, display label, unit suffix for printing)
METRICS = [
    ("gpu_util_percent", "GPU Utilization", "%"),
    ("vram_percent", "VRAM Utilization", "%"),
    ("vram_used_gb", "VRAM Used", "GB"),
    ("cpu_percent", "CPU Utilization", "%"),
    ("ram_percent", "RAM Utilization", "%"),
    ("ram_used_gb", "RAM Used", "GB"),
]


def percentile(values, pct):
    if not values:
        return None
    values = sorted(values)
    k = (len(values) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(values) - 1)
    if f == c:
        return values[f]
    return values[f] + (values[c] - values[f]) * (k - f)


def analyze_file(path):
    """Returns {interval_ms, n, metric_col: {mean, p95, peak}, ...}"""
    columns = {col: [] for col, _, _ in METRICS}
    interval_ms = None
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            interval_ms = row.get("interval_ms") or interval_ms
            for col in columns:
                v = row.get(col)
                if v not in (None, ""):
                    try:
                        columns[col].append(float(v))
                    except ValueError:
                        pass

    stats = {"interval_ms": interval_ms}
    n_ref = None
    for col, values in columns.items():
        if values:
            stats[col] = {
                "mean": sum(values) / len(values),
                "p95": percentile(values, 95),
                "peak": max(values),
            }
            n_ref = n_ref or len(values)
    stats["n"] = n_ref or 0
    return stats if n_ref else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+", help="CSV files from gpu_logger.py (globs OK)")
    args = parser.parse_args()

    paths = []
    for pattern in args.files:
        paths.extend(sorted(glob.glob(pattern)))
    if not paths:
        print("No files matched.", file=sys.stderr)
        sys.exit(1)

    results = [r for r in (analyze_file(p) for p in paths) if r]

    def interval_key(r):
        try:
            return int(r["interval_ms"])
        except (TypeError, ValueError):
            return 0
    results.sort(key=interval_key)

    def label_for(ms_raw):
        ms = int(ms_raw) if ms_raw else 0
        return f"{ms/1000:g}s" if ms >= 1000 else f"{ms}ms"

    for col, title, unit in METRICS:
        if not any(col in r for r in results):
            continue
        print(f"\n{title} ({unit})")
        print(f"{'Sampling':<12}{'N':<8}{'Mean':<10}{'P95':<10}{'Peak':<8}")
        print("-" * 48)
        for r in results:
            if col not in r:
                continue
            s = r[col]
            print(f"{label_for(r['interval_ms']):<12}{r['n']:<8}"
                  f"{s['mean']:<10.2f}{s['p95']:<10.2f}{s['peak']:<8.2f}")


if __name__ == "__main__":
    main()