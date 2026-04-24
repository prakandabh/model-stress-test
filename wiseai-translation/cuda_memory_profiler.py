"""
CUDA Memory Profiler
====================
Tracks GPU VRAM usage before and after each translation inference call.
Renders a professional Rich table with per-sentence deltas and a
summary panel showing overall memory behaviour.

Designed to be imported by translation_pipeline.py, but can also be
run as a standalone script for a quick GPU status check.

Usage (standalone):
    python cuda_memory_profiler.py
"""

from dataclasses import dataclass
from typing import List

import pynvml
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

# Configuration
DEFAULT_GPU_INDEX = 0
BYTES_PER_MB = 1024 * 1024
TEXT_PREVIEW_LEN = 38  # max chars shown in the "Text" column


# Colour palette (Rich markup)
COLOR_INCREASE = "red"
COLOR_DECREASE = "green"
COLOR_NEUTRAL = "dim white"
COLOR_HEADER = "bold cyan"
COLOR_SUBTLE = "dim"
COLOR_HIGHLIGHT = "bold white"
COLOR_WARNING = "yellow"
COLOR_OK = "green"


# Dataclasses
@dataclass(frozen=True)
class MemorySnapshot:
    """Point-in-time GPU memory reading."""

    label: str
    used_mb: float
    free_mb: float
    total_mb: float

    @property
    def utilization_pct(self) -> float:
        return (self.used_mb / self.total_mb * 100) if self.total_mb else 0.0


@dataclass(frozen=True)
class MemoryEvent:
    """Paired before/after snapshots for one inference call."""

    index: int
    text_preview: str
    before: MemorySnapshot
    after: MemorySnapshot

    @property
    def delta_mb(self) -> float:
        return self.after.used_mb - self.before.used_mb

    @property
    def delta_tag(self) -> str:
        """Rich-markup coloured delta string."""
        delta = self.delta_mb
        color = (
            COLOR_INCREASE
            if delta > 0
            else (COLOR_DECREASE if delta < 0 else COLOR_NEUTRAL)
        )
        sign = "+" if delta > 0 else ""
        return "[{color}]{sign}{val:.1f}[/{color}]".format(
            color=color, sign=sign, val=delta
        )


# NVML helpers
def init_nvml() -> None:
    """Initialise pynvml. Raises if NVML is unavailable."""
    pynvml.nvmlInit()


def shutdown_nvml() -> None:
    """Cleanly shut down pynvml."""
    pynvml.nvmlShutdown()


def snapshot(label: str, gpu_index: int = DEFAULT_GPU_INDEX) -> MemorySnapshot:
    """Capture current GPU memory state and return an immutable snapshot."""
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    info = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return MemorySnapshot(
        label=label,
        used_mb=info.used / BYTES_PER_MB,
        free_mb=info.free / BYTES_PER_MB,
        total_mb=info.total / BYTES_PER_MB,
    )


def _gpu_name(gpu_index: int) -> str:
    """Return the GPU device name string."""
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        return pynvml.nvmlDeviceGetName(handle)
    except Exception:
        return "Unknown GPU"


# Event builder
def record_event(
    index: int,
    text: str,
    before: MemorySnapshot,
    after: MemorySnapshot,
) -> MemoryEvent:
    """Construct a MemoryEvent from a pair of snapshots."""
    preview = text[:TEXT_PREVIEW_LEN] + "…" if len(text) > TEXT_PREVIEW_LEN else text
    return MemoryEvent(index=index, text_preview=preview, before=before, after=after)


# Display helpers
def _mb(value: float) -> str:
    return "{:.1f}".format(value)


def _pct(value: float) -> str:
    return "{:.1f}%".format(value)


def _utilization_color(pct: float) -> str:
    if pct >= 90:
        return COLOR_INCREASE
    if pct >= 70:
        return COLOR_WARNING
    return COLOR_OK


def _build_per_sentence_table(events: List[MemoryEvent]) -> Table:
    """Render one row per inference call showing before/after memory."""
    table = Table(
        title="Per-Sentence CUDA Memory Snapshots",
        title_style=COLOR_HEADER,
        box=box.SIMPLE_HEAD,
        show_lines=True,
        padding=(0, 1),
        expand=False,
    )

    table.add_column("#", style=COLOR_SUBTLE, width=4, justify="right")
    table.add_column("Text Preview", style="white", min_width=30, max_width=40)
    table.add_column("Before (MB)", style="cyan", width=12, justify="right")
    table.add_column("After (MB)", style="cyan", width=11, justify="right")
    table.add_column("Δ (MB)", style="default", width=10, justify="right")
    table.add_column("Free After", style=COLOR_SUBTLE, width=11, justify="right")
    table.add_column("Util After", style="default", width=10, justify="right")

    for ev in events:
        util_after = ev.after.utilization_pct
        util_color = _utilization_color(util_after)
        table.add_row(
            str(ev.index),
            ev.text_preview,
            _mb(ev.before.used_mb),
            _mb(ev.after.used_mb),
            ev.delta_tag,
            _mb(ev.after.free_mb) + " MB",
            "[{c}]{p}[/{c}]".format(c=util_color, p=_pct(util_after)),
        )

    return table


def _build_summary_panel(events: List[MemoryEvent], gpu_name: str) -> Panel:
    """Render aggregate statistics across all inference events."""
    if not events:
        return Panel("[dim]No events recorded.[/dim]", title="Summary")

    deltas = [ev.delta_mb for ev in events]
    total_delta = sum(deltas)
    max_delta = max(deltas)
    min_delta = min(deltas)
    avg_delta = total_delta / len(deltas)

    peak_used = max(ev.after.used_mb for ev in events)
    min_free = min(ev.after.free_mb for ev in events)
    total_mb = events[0].after.total_mb

    leak_flag = total_delta > 50  # >50 MB net growth is suspicious
    leak_msg = (
        "[{c}]  ▲ Possible memory inflation detected ({v:.1f} MB net increase)[/{c}]".format(
            c=COLOR_INCREASE, v=total_delta
        )
        if leak_flag
        else "[{c}]  ✓ Memory appears stable across all calls[/{c}]".format(c=COLOR_OK)
    )

    lines = [
        "",
        "  [bold]GPU[/bold]           {name}".format(name=gpu_name),
        "  [bold]Total VRAM[/bold]    {t} MB".format(t=_mb(total_mb)),
        "  [bold]Sentences[/bold]     {n}".format(n=len(events)),
        "",
        "  [bold cyan]Memory Deltas (MB)[/bold cyan]",
        "  ├─ Total Δ      {v:+.1f}".format(v=total_delta),
        "  ├─ Average Δ    {v:+.1f}".format(v=avg_delta),
        "  ├─ Max spike    {v:+.1f}".format(v=max_delta),
        "  └─ Max reclaim  {v:+.1f}".format(v=min_delta),
        "",
        "  [bold cyan]Peak Usage[/bold cyan]",
        "  ├─ Peak used    {v} MB  ({p})".format(
            v=_mb(peak_used), p=_pct(peak_used / total_mb * 100)
        ),
        "  └─ Min free     {v} MB".format(v=_mb(min_free)),
        "",
        leak_msg,
        "",
    ]

    return Panel(
        "\n".join(lines),
        title="[bold]Profiling Summary[/bold]",
        border_style="cyan",
        expand=False,
    )


# Public display entrypoint
def display_summary(
    events: List[MemoryEvent],
    gpu_index: int = DEFAULT_GPU_INDEX,
) -> None:
    """
    Print the full memory profile report to the terminal.

    Call this once after the pipeline has completed.
    """
    console = Console()
    gpu_name = _gpu_name(gpu_index)

    console.print()
    console.print(
        Rule(
            "[bold cyan]CUDA Memory Profile — GPU {idx}: {name}[/bold cyan]".format(
                idx=gpu_index, name=gpu_name
            )
        )
    )
    console.print()

    if not events:
        console.print("[dim]No memory events were recorded.[/dim]")
        return

    console.print(_build_per_sentence_table(events))
    console.print()
    console.print(_build_summary_panel(events, gpu_name))
    console.print()


# Standalone quick-check
def _standalone_check(gpu_index: int = DEFAULT_GPU_INDEX) -> None:
    """Print a one-shot GPU memory status panel (no pipeline needed)."""
    console = Console()
    init_nvml()
    snap = snapshot("now", gpu_index)
    gpu_name = _gpu_name(gpu_index)
    shutdown_nvml()

    console.print(
        Panel(
            "\n".join(
                [
                    "",
                    "  [bold]GPU[/bold]         {name}".format(name=gpu_name),
                    "  [bold]Used[/bold]        {u} MB  ({p})".format(
                        u=_mb(snap.used_mb), p=_pct(snap.utilization_pct)
                    ),
                    "  [bold]Free[/bold]        {f} MB".format(f=_mb(snap.free_mb)),
                    "  [bold]Total[/bold]       {t} MB".format(t=_mb(snap.total_mb)),
                    "",
                ]
            ),
            title="[bold cyan]GPU {idx} Memory Status[/bold cyan]".format(
                idx=gpu_index
            ),
            border_style="cyan",
            expand=False,
        )
    )


if __name__ == "__main__":
    _standalone_check()
