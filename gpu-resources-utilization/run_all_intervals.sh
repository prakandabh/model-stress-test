#!/usr/bin/env bash
# run_all_intervals.sh
# Walks through all 5 sampling intervals as SEPARATE runs. For each one it
# starts gpu_logger.py, waits for you to trigger your JMeter test from
# wherever you run it (even a different machine hitting this server over the
# network), then stops logging when you press Enter.
#
# This script does NOT launch or depend on JMeter/your pipeline in any way --
# it only controls the logger.
#
# Usage:
#   ./run_all_intervals.sh <test-name> [pid]
#
# Examples:
#   ./run_all_intervals.sh asr_load_test
#   ./run_all_intervals.sh asr_load_test 48213   # also track that PID's per-process columns
#
set -euo pipefail

NAME="${1:?Usage: $0 <test-name> [pid]}"
PID="${2:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTDIR="$SCRIPT_DIR/gpu_metrics"
mkdir -p "$OUTDIR"

INTERVALS_MS=(100 250 500 1000 5000)

for ms in "${INTERVALS_MS[@]}"; do
    if [ "$ms" -ge 1000 ]; then
        label="$((ms/1000))s"
    else
        label="${ms}ms"
    fi
    outfile="${OUTDIR}/${NAME}_${label}.csv"

    echo "=================================================="
    echo "Interval: ${ms}ms  ->  ${outfile}"
    echo "=================================================="

    PID_ARGS=()
    if [ -n "$PID" ]; then
        PID_ARGS=(--pid "$PID")
    fi

    python3 "$SCRIPT_DIR/gpu_logger.py" --interval-ms "$ms" --output "$outfile" "${PID_ARGS[@]}" &
    LOGGER_PID=$!

    # Give the logger a moment to initialize NVML
    sleep 1

    echo ""
    echo ">>> Logger is running. Go trigger your JMeter test now (from any device)."
    read -r -p ">>> Press ENTER once the JMeter test has finished... " _

    kill -INT "$LOGGER_PID" 2>/dev/null || true
    wait "$LOGGER_PID" 2>/dev/null || true

    echo "Saved ${outfile}"
    echo ""

    # Cooldown between runs so one run's tail doesn't bleed into the next
    sleep 2
done

echo "All runs complete. Files in ${OUTDIR}/"
ls -la "$OUTDIR"