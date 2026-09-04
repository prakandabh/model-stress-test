# GPU Resource Utilization Logging

Standalone logger that captures CPU, GPU, RAM, and VRAM usage on **this**
machine while you drive load against it however you like -- e.g. a JMeter
test plan you run from your own laptop's UI against this server. The logger
does not launch, wrap, or depend on JMeter or your pipeline in any way; you
just start it, run your test, and stop it.

Logs the following, at whichever sampling interval you choose (100ms, 250ms,
500ms, 1s, 5s are the recommended set for a bursty inference workload):

- **CPU utilization** (%, system-wide + optionally per-process)
- **GPU utilization** (%, per-GPU + optionally per-process SM%)
- **RAM (system memory)**: used GB, total GB, and utilization %
- **VRAM (GPU memory)**: used GB, total GB, and utilization %
- Power draw (W), plus process RAM/VRAM in MB as a bonus

## Folder contents

```
gpu-resources-utilization/
├── gpu_logger.py           # standalone logger, one CSV row per sample
├── run_all_intervals.sh    # interactive helper: steps through all 5 intervals
├── analyze_gpu_logs.py     # computes mean/P95/peak stats table from the CSVs
├── README.md
└── gpu_metrics/              # auto-created, holds output CSVs after a run
```

## 1. One-time setup

```bash
pip install nvidia-ml-py psutil --break-system-packages
```

## 2. How the workflow actually works

Since JMeter runs from a UI on a different machine, this script never starts
your load for you. Instead, for each test:

1. Start the logger on **this** machine (the one being tested).
2. Trigger the JMeter test from wherever you normally run it.
3. When the test finishes, stop the logger.

### Option A -- one interval at a time, manual

```bash
cd model-stress-test/gpu-resources-utilization

python3 gpu_logger.py --interval-ms 250 --name asr_load_test1
```

This writes to `gpu_metrics/asr_load_test1_250ms_<timestamp>.csv`. Go run
your JMeter test now; when it's done, press `Ctrl+C` in this terminal to stop
logging. Repeat with `--interval-ms 100`, `500`, `1000`, `5000` for the other
runs (each is a separate JMeter run, per professional benchmarking practice
-- don't run five loggers at once against one test).

If you'd rather not rely on `Ctrl+C` timing, and you know roughly how long
the JMeter test plan takes, use `--duration-s` instead:

```bash
python3 gpu_logger.py --interval-ms 250 --name asr_load_test1 --duration-s 120
```

### Option B -- walk through all 5 intervals interactively

```bash
./run_all_intervals.sh asr_load_test1
```

For each interval it starts the logger, prints `Go trigger your JMeter test
now`, and waits for you to press **Enter** once that JMeter run has finished
before moving to the next interval. Output files:

```
gpu_metrics/
├── asr_load_test1_100ms.csv
├── asr_load_test1_250ms.csv
├── asr_load_test1_500ms.csv
├── asr_load_test1_1s.csv
└── asr_load_test1_5s.csv
```

## 3. Naming your log files

- `--name asr_load_test1` -> auto-builds `gpu_metrics/asr_load_test1_250ms_<timestamp>.csv`
  (timestamp avoids overwriting a previous run with the same name).
- `--output gpu_metrics/my_custom_filename.csv` -> use this exact path instead,
  no auto-naming.
- With `run_all_intervals.sh <test-name>`, that name is reused for all 5
  files, suffixed with the interval (no timestamp, since each interval only
  runs once per invocation).

Use a name that identifies the pipeline and scenario, e.g.
`asr_50users`, `tts_ramp_test`, `translation_soak_test`.

## 4. Tracking a specific process (optional)

Your pipeline server is presumably already running before JMeter hits it.
If you want the per-process columns (`process_gpu_sm_percent`,
`process_vram_mb`, `process_cpu_percent`, `process_ram_mb`) populated for
that specific process rather than left blank, find its PID first:

```bash
pgrep -f pipeline.py
# e.g. -> 48213
```

Then pass it in:

```bash
python3 gpu_logger.py --interval-ms 250 --name asr_load_test1 --pid 48213

# or with the interactive helper:
./run_all_intervals.sh asr_load_test1 48213
```

Without `--pid`, the logger still records every process NVML can see on the
GPU (one row per detected PID per sample), it just won't single one out for
the `process_*` columns.

`--watch-pid` (available on `gpu_logger.py` directly, not used by
`run_all_intervals.sh`) auto-stops logging when a PID exits -- only useful
if the tracked process is expected to end during the run. Leave it unset for
a long-running server, which is the normal case here.

## 5. Generate the summary table

```bash
python3 analyze_gpu_logs.py gpu_metrics/asr_load_test1_*.csv
```

Prints a separate mean/P95/peak table for GPU Utilization, VRAM Utilization,
VRAM Used, CPU Utilization, RAM Utilization, and RAM Used, one row per
sampling interval:

```
GPU Utilization (%)
Sampling    N       Mean      P95       Peak
------------------------------------------------
100ms       612     78.40     96.00     100.00
250ms       245     76.90     94.00     100.00
500ms       123     74.20     91.00     98.00
1s          61      71.80     87.00     94.00
5s          12      63.50     76.00     82.00

VRAM Utilization (%)
...

RAM Utilization (%)
...
```

## CSV columns

| Column | Meaning |
|---|---|
| `gpu_util_percent` | GPU compute utilization (%) |
| `vram_used_gb` / `vram_total_gb` / `vram_percent` | GPU memory used / total / % |
| `power_w` | GPU power draw (W) |
| `cpu_percent` | System-wide CPU utilization (%) |
| `ram_used_gb` / `ram_total_gb` / `ram_percent` | System RAM used / total / % |
| `process_gpu_sm_percent` | Tracked PID's share of GPU SM usage (%) |
| `process_vram_mb` | Tracked PID's VRAM usage (MB) |
| `process_cpu_percent` | Tracked PID's CPU usage (%, can exceed 100% on multi-core) |
| `process_ram_mb` | Tracked PID's RAM usage (RSS, MB) |

## Notes

- `process_gpu_sm_percent` needs a recent NVIDIA driver and isn't supported
  on every card -- if unavailable it falls back to device-level
  `gpu_util_percent` only, with process columns left blank.
- `cpu_percent` / `process_cpu_percent` are `psutil` readings taken once per
  sampling tick, so at 100ms/250ms they reflect a fairly instantaneous
  snapshot rather than a smoothed average -- the same "short bursts vs.
  smoothed averages" tradeoff that motivates testing multiple intervals in
  the first place.
- Nothing here needs to run on the machine JMeter's UI is on -- only on the
  server actually being tested.