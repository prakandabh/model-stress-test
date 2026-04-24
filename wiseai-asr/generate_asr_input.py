"""
ASR Input CSV Generator
=======================
Scans a folder of audio files and generates the input CSVs
required by asr_pipeline.py.

Only generates asr_input_stream.csv (transcribe-from-stream).
The transcribe-from-url endpoint requires MinIO URLs which must
be provided manually — a template CSV is generated for that instead.

============================================================
INPUT FORMAT  (what this script produces / pipeline expects)
============================================================

  asr_input_stream.csv
  --------------------
  Columns : audio_file_path, language
  Example :
      audio_file_path,language
      /data/audio/clip_001.wav,nepali
      /data/audio/clip_002.mp3,english

  asr_input_url.csv
  -----------------
  Columns : input_minio_url, language
  Example :
      input_minio_url,language
      http://minio.wiseyak.com/asr-input/clip_001.wav,nepali
      http://minio.wiseyak.com/asr-input/clip_002.mp3,english

============================================================
OUTPUT FORMAT  (what asr_pipeline.py produces)
============================================================

  asr_output_stream.csv  and  asr_output_url.csv
  -----------------------------------------------
  Columns : audio_file_path (or input_minio_url), language,
            transcription, output_minio_url

  When OUTPUT_TYPE = "text":
      audio_file_path,language,transcription,output_minio_url
      /data/audio/clip_001.wav,nepali,यो एउटा परीक्षण हो।,

  When OUTPUT_TYPE = "url":
      audio_file_path,language,transcription,output_minio_url
      /data/audio/clip_001.wav,nepali,,http://minio.wiseyak.com/asr-output/clip_001.txt

============================================================

Usage:
    python generate_asr_input.py
"""

import csv
import logging
import sys
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

# Root folder containing audio files (searched recursively if RECURSIVE = True)
AUDIO_BASE_PATH = "/home/oem/wiseyak/prakanda/stree-test/wiseai-asr/audio_from_text"

# Language to assign to all discovered audio files
# Override per-file logic is not supported — run the script separately
# for each language folder and concatenate CSVs if needed.
LANGUAGE = "nepali"

# File extensions to include (lowercase, with leading dot)
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")

# Whether to search subdirectories recursively
RECURSIVE = True

# Output CSV paths (must match INPUT_STREAM_CSV / INPUT_URL_CSV in asr_pipeline.py)
OUTPUT_STREAM_CSV = "asr_input_stream.csv"

# Whether to overwrite existing CSVs or append to them
CSV_OVERWRITE = True

# MinIO URL prefix used when generating the URL template CSV
# Final URL per file = MINIO_URL_PREFIX + "/" + filename
MINIO_URL_PREFIX = "http://minio.wiseyak.com/asr-input"

# ---------------------------------------------------------------------------
# Column names — must match asr_pipeline.py constants exactly
# ---------------------------------------------------------------------------

COL_AUDIO_FILE_PATH = "audio_file_path"
COL_INPUT_MINIO_URL = "input_minio_url"
COL_LANGUAGE        = "language"

# ---------------------------------------------------------------------------
# Log message templates
# ---------------------------------------------------------------------------

LOG_SCAN_START       = "Scanning: {path} (recursive={recursive})"
LOG_FOUND_FILES      = "Found {count} audio file(s) with extensions: {exts}"
LOG_NO_FILES_FOUND   = "No audio files found in: {path}"
LOG_WROTE_STREAM_CSV = "Wrote stream input CSV ({count} rows): {path}"
LOG_WROTE_URL_CSV    = "Wrote URL template CSV ({count} rows): {path}"
LOG_SKIPPED_EXT      = "Skipped {count} file(s) with unsupported extensions"
LOG_DONE             = "Done. Run asr_pipeline.py to start transcription."

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt = "%H:%M:%S",
    handlers = [logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(Path(__file__).stem)

# ---------------------------------------------------------------------------
# Pure functions
# ---------------------------------------------------------------------------


def collect_audio_files(base_path: str, recursive: bool) -> List[Path]:
    """
    Return all audio files under base_path matching AUDIO_EXTENSIONS.
    Results are sorted for deterministic output.
    """
    root = Path(base_path)
    glob_fn = root.rglob if recursive else root.glob

    return sorted(
        p for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _minio_url(file_path: Path) -> str:
    """Construct a MinIO URL from the filename."""
    return "{prefix}/{filename}".format(
        prefix   = MINIO_URL_PREFIX.rstrip("/"),
        filename = file_path.name,
    )


def build_stream_rows(files: List[Path], language: str) -> List[dict]:
    """Build row dicts for asr_input_stream.csv."""
    return [
        {COL_AUDIO_FILE_PATH: str(f.resolve()), COL_LANGUAGE: language}
        for f in files
    ]


def build_url_rows(files: List[Path], language: str) -> List[dict]:
    """
    Build row dicts for asr_input_url.csv.
    URLs are template values — fill in the correct MinIO paths before running
    the pipeline if MINIO_URL_PREFIX doesn't match your actual bucket layout.
    """
    return [
        {COL_INPUT_MINIO_URL: _minio_url(f), COL_LANGUAGE: language}
        for f in files
    ]


def write_csv(rows: List[dict], fieldnames: List[str], csv_path: str, overwrite: bool) -> None:
    """Write rows to CSV, respecting the overwrite flag."""
    mode = "w" if overwrite else "a"
    file_exists  = Path(csv_path).exists()
    write_header = overwrite or not file_exists

    with open(csv_path, mode, newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def count_skipped(base_path: str, recursive: bool) -> int:
    """Count files that exist but were excluded due to unsupported extensions."""
    root    = Path(base_path)
    glob_fn = root.rglob if recursive else root.glob
    total   = sum(1 for p in glob_fn("*") if p.is_file())
    kept    = sum(
        1 for p in glob_fn("*")
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )
    return total - kept


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def generate() -> None:
    """Entry point — scan folder and write both input CSVs."""
    logger.info(LOG_SCAN_START.format(path=AUDIO_BASE_PATH, recursive=RECURSIVE))

    files = collect_audio_files(AUDIO_BASE_PATH, RECURSIVE)

    if not files:
        logger.warning(LOG_NO_FILES_FOUND.format(path=AUDIO_BASE_PATH))
        return

    logger.info(LOG_FOUND_FILES.format(
        count = len(files),
        exts  = ", ".join(AUDIO_EXTENSIONS),
    ))

    skipped = count_skipped(AUDIO_BASE_PATH, RECURSIVE)
    if skipped:
        logger.info(LOG_SKIPPED_EXT.format(count=skipped))

    # Stream input CSV
    stream_rows = build_stream_rows(files, LANGUAGE)
    write_csv(stream_rows, [COL_AUDIO_FILE_PATH, COL_LANGUAGE], OUTPUT_STREAM_CSV, CSV_OVERWRITE)
    logger.info(LOG_WROTE_STREAM_CSV.format(count=len(stream_rows), path=OUTPUT_STREAM_CSV))

    logger.info(LOG_DONE)


if __name__ == "__main__":
    generate()