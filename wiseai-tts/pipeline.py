"""
TTS Pipeline
============
Calls the deployed TTS service for each row in an input CSV.

Endpoint selection  (set ACTIVE_ENDPOINTS below):
  "text"  — only generate_from_text
  "url"   — only generate_from_url
  "both"  — both endpoints in sequence

Output behaviour:
  output_type = "audio"
      generate_from_text  →  saves WAV to  tts_audio_output/audio_from_text/
      generate_from_url   →  saves WAV to  tts_audio_output/audio_from_url/
                             also writes tts_output_url.csv (output_audio_path populated)

  output_type = "url"
      generate_from_text  →  writes tts_output_text.csv
                             (output_minio_url_key = bucket/org/service/user/filename)
      generate_from_url   →  writes tts_output_url.csv
                             (output_minio_url_key + estimated_seconds from response)

Usage:
    python tts_pipeline.py
"""

import csv
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests
from cuda_memory_profiler import (
    MemoryEvent,
    display_summary,
    init_nvml,
    record_event,
    shutdown_nvml,
    snapshot,
)

# Endpoint selection — set to "text", "url", or "both"
ACTIVE_ENDPOINTS = "text"


# Service configuration
BASE_URL = "http://192.168.88.10:9999"
ENDPOINT_FROM_TEXT = "/generate_from_text"
ENDPOINT_FROM_URL = "/generate_from_url"
REQUEST_TIMEOUT_SEC = 120
GPU_INDEX = 0


# TTS generation parameters
OUTPUT_TYPE = "audio"  # "audio" or "url"
AUDIO_SPEED = 1.0
REFERENCE_AUDIO_ID = "traindatashuvani_2"
TARGET_SAMPLE_RATE = None  # set to int (e.g. 22050) or leave None

# Required when OUTPUT_TYPE = "url" and endpoint = generate_from_text
URL_OUTPUT_BUCKET = "tts-output"
URL_OUTPUT_ORGANIZATION = "wiseyak"
URL_OUTPUT_SERVICE = "tts"
URL_OUTPUT_USER_ID = "pipeline"


# File paths
INPUT_TEXT_CSV = "tts_input_text.csv"
INPUT_URL_CSV = "tts_input_url.csv"
AUDIO_OUTPUT_DIR = "tts_audio_output"
AUDIO_SUBDIR_TEXT = "audio_from_text"
AUDIO_SUBDIR_URL = "audio_from_url"
OUTPUT_CSV_TEXT = "tts_output_text.csv"
OUTPUT_CSV_URL = "tts_output_url.csv"
LOG_FILE_PATH = "tts_pipeline.log"


# Write mode controls

# True  → append new runs to existing output CSVs
# False → overwrite output CSVs on each run
CSV_APPEND_MODE = True

# True  → append new runs to existing log file
# False → overwrite log file on each run
LOG_APPEND_MODE = True


# CSV column names — input
COL_INPUT_TEXT = "input_text"
COL_INPUT_MINIO_URL = "input_minio_url"
COL_LANGUAGE = "language"


# CSV column names — output: generate_from_text (both modes)
COL_TXT_INPUT_TEXT = "input_text"
COL_TXT_MINIO_KEY = "output_minio_url_key"  # full key when url; local path when audio


# CSV column names — output: generate_from_url (always all four columns)
COL_URL_INPUT_URL = "input_minio_url"
COL_URL_MINIO_KEY = "output_minio_url_key"
COL_URL_EST_SECS = "estimated_seconds"
COL_URL_AUDIO_PATH = "output_audio_path"


# Form field names (must match FastAPI endpoint signatures)
FORM_TEXT = "text"
FORM_MINIO_URL = "minio_url"
FORM_LANGUAGE = "language"
FORM_OUTPUT_TYPE = "output_type"
FORM_AUDIO_SPEED = "audio_speed"
FORM_REFERENCE_AUDIO_ID = "reference_audio_id"
FORM_TARGET_SAMPLE_RATE = "target_sample_rate"
FORM_BUCKET_NAME = "bucket_name"
FORM_ORGANIZATION = "organization"
FORM_SERVICE = "service"
FORM_USER_ID = "user_id"


# Response keys
RESPONSE_MINIO_URL_KEY = "minio_url"
RESPONSE_ESTIMATED_SECS_KEY = "estimated_seconds"


# Audio file settings
AUDIO_FILE_EXTENSION = ".wav"
AUDIO_FILE_PREFIX_TXT = "tts_text"
AUDIO_FILE_PREFIX_URL = "tts_url"


# Log message templates
LOG_PIPELINE_START = "TTS pipeline started — endpoint(s): {endpoints}"
LOG_FILE_HANDLER_INIT = "File logging active — {path}"
LOG_SECTION_START = (
    "--- Running: {label} | output_type: {output_type} | rows: {count} ---"
)
LOG_LOADED_ROWS = "Loaded {count} rows from {path}"
LOG_REQUESTING = "[{idx}/{total}] {label} ({lang}): {preview!r}"
LOG_REQUEST_OK_AUDIO = "[{idx}/{total}] OK — audio saved: {path}"
LOG_REQUEST_OK_URL = "[{idx}/{total}] OK — minio_key: {key}"
LOG_REQUEST_HTTP_ERROR = "[{idx}/{total}] HTTP {status} — {body}"
LOG_REQUEST_CONN_ERROR = "[{idx}/{total}] Connection error: {error}"
LOG_REQUEST_TIMEOUT = "[{idx}/{total}] Timeout after {timeout}s"
LOG_SKIPPING_ROW = "Skipping row {idx} — missing required fields"
LOG_WROTE_CSV = "Wrote {count} rows to {path}"
LOG_WROTE_AUDIO = "Saved audio: {path} ({size} bytes)"
LOG_DIR_CREATED = "Created directory: {path}"
LOG_PIPELINE_DONE = "{label} complete — {success} succeeded, {failed} failed"
LOG_NVML_UNAVAILABLE = "pynvml unavailable — CUDA profiling disabled: {error}"


# Logging setup — stream + rotating file handler
_LOG_FORMAT = "%(asctime)s  %(levelname)-8s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

_file_handler = logging.FileHandler(
    LOG_FILE_PATH, mode="a" if LOG_APPEND_MODE else "w", encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger(Path(__file__).stem)


# Helpers — directory management
TEXT_PREVIEW_LEN = 40


def _preview(text: str) -> str:
    return text[:TEXT_PREVIEW_LEN] + "…" if len(text) > TEXT_PREVIEW_LEN else text


def _ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it does not exist."""
    if not path.exists():
        path.mkdir(parents=True)
        logger.info(LOG_DIR_CREATED.format(path=path))
    return path


def _audio_path_text(idx: int) -> Path:
    """Return the target WAV path for a generate_from_text audio output."""
    subdir = _ensure_dir(Path(AUDIO_OUTPUT_DIR) / AUDIO_SUBDIR_TEXT)
    return subdir / "{prefix}_{idx:04d}{ext}".format(
        prefix=AUDIO_FILE_PREFIX_TXT, idx=idx, ext=AUDIO_FILE_EXTENSION
    )


def _audio_path_url(idx: int) -> Path:
    """Return the target WAV path for a generate_from_url audio output."""
    subdir = _ensure_dir(Path(AUDIO_OUTPUT_DIR) / AUDIO_SUBDIR_URL)
    return subdir / "{prefix}_{idx:04d}{ext}".format(
        prefix=AUDIO_FILE_PREFIX_URL, idx=idx, ext=AUDIO_FILE_EXTENSION
    )


# Helpers — MinIO key construction
def _build_full_minio_key(raw_minio_url: str) -> str:
    """
    Construct the full MinIO object key for generate_from_text URL-mode output.

    Pattern:  {bucket}/{org}/{service}/{user_id}/{filename}

    The filename is extracted from the tail of the raw minio_url returned
    by the service.
    """
    filename = Path(raw_minio_url).name
    return "{bucket}/{org}/{service}/{user}/{filename}".format(
        bucket=URL_OUTPUT_BUCKET,
        org=URL_OUTPUT_ORGANIZATION,
        service=URL_OUTPUT_SERVICE,
        user=URL_OUTPUT_USER_ID,
        filename=filename,
    )


# CSV I/O
def load_rows(csv_path: str) -> List[Dict[str, str]]:
    """Read all rows from a CSV and return as list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info(LOG_LOADED_ROWS.format(count=len(rows), path=csv_path))
    return rows


def _open_csv(csv_path: str, fieldnames: List[str]):
    """
    Return an open CSV writer context.

    Appends to existing file (skipping header) when CSV_APPEND_MODE is True.
    Overwrites the file (writing header) when CSV_APPEND_MODE is False.
    """
    file_exists = Path(csv_path).exists()
    write_header = (not CSV_APPEND_MODE) or (not file_exists)
    mode = "a" if CSV_APPEND_MODE else "w"
    fh = open(csv_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    return fh, writer


def write_text_section_output(results: List[Dict[str, str]], csv_path: str) -> None:
    """
    Write generate_from_text results.

    Columns: input_text, output_minio_url_key
    When audio mode: output_minio_url_key holds the local audio file path.
    When url mode:   output_minio_url_key holds bucket/org/service/user/filename.
    """
    fieldnames = [COL_TXT_INPUT_TEXT, COL_TXT_MINIO_KEY]
    fh, writer = _open_csv(csv_path, fieldnames)
    writer.writerows(results)
    fh.close()
    logger.info(LOG_WROTE_CSV.format(count=len(results), path=csv_path))


def write_url_section_output(results: List[Dict[str, str]], csv_path: str) -> None:
    """
    Write generate_from_url results.

    Columns: input_minio_url, output_minio_url_key, estimated_seconds, output_audio_path
    Always written regardless of output_type; unpopulated columns are empty strings.
    """
    fieldnames = [
        COL_URL_INPUT_URL,
        COL_URL_MINIO_KEY,
        COL_URL_EST_SECS,
        COL_URL_AUDIO_PATH,
    ]
    fh, writer = _open_csv(csv_path, fieldnames)
    writer.writerows(results)
    fh.close()
    logger.info(LOG_WROTE_CSV.format(count=len(results), path=csv_path))


# Payload builders
def _base_form_fields(language: str) -> Dict:
    """Common form fields shared by both endpoints."""
    base = {
        FORM_LANGUAGE: language,
        FORM_OUTPUT_TYPE: OUTPUT_TYPE,
        FORM_AUDIO_SPEED: AUDIO_SPEED,
        FORM_REFERENCE_AUDIO_ID: REFERENCE_AUDIO_ID,
    }
    return (
        {**base, FORM_TARGET_SAMPLE_RATE: TARGET_SAMPLE_RATE}
        if TARGET_SAMPLE_RATE is not None
        else base
    )


def build_text_payload(text: str, language: str) -> Dict:
    """Build form payload for generate_from_text."""
    base = {FORM_TEXT: text, **_base_form_fields(language)}
    if OUTPUT_TYPE == "url":
        return {
            **base,
            FORM_BUCKET_NAME: URL_OUTPUT_BUCKET,
            FORM_ORGANIZATION: URL_OUTPUT_ORGANIZATION,
            FORM_SERVICE: URL_OUTPUT_SERVICE,
            FORM_USER_ID: URL_OUTPUT_USER_ID,
        }
    return base


def build_url_payload(minio_url: str, language: str) -> Dict:
    """Build form payload for generate_from_url."""
    return {FORM_MINIO_URL: minio_url, **_base_form_fields(language)}


# Validation
def _is_valid_text_row(row: Dict[str, str]) -> bool:
    return all(row.get(col, "").strip() for col in (COL_INPUT_TEXT, COL_LANGUAGE))


def _is_valid_url_row(row: Dict[str, str]) -> bool:
    return all(row.get(col, "").strip() for col in (COL_INPUT_MINIO_URL, COL_LANGUAGE))


# HTTP
def _post_form(
    endpoint_url: str, data: Dict, idx: int, total: int
) -> requests.Response:
    """Execute a form-encoded POST. Raises on connection/timeout errors."""
    try:
        return requests.post(endpoint_url, data=data, timeout=REQUEST_TIMEOUT_SEC)
    except requests.ConnectionError as exc:
        logger.error(LOG_REQUEST_CONN_ERROR.format(idx=idx, total=total, error=exc))
        raise
    except requests.Timeout:
        logger.error(
            LOG_REQUEST_TIMEOUT.format(
                idx=idx, total=total, timeout=REQUEST_TIMEOUT_SEC
            )
        )
        raise


def _save_audio(content: bytes, path: Path) -> str:
    """Write WAV bytes to disk and return the absolute path string."""
    path.write_bytes(content)
    logger.info(LOG_WROTE_AUDIO.format(path=path, size=len(content)))
    return str(path.resolve())


# Empty result builders
def _empty_text_result(text: str) -> Dict[str, str]:
    return {COL_TXT_INPUT_TEXT: text, COL_TXT_MINIO_KEY: ""}


def _empty_url_result(minio_url: str) -> Dict[str, str]:
    return {
        COL_URL_INPUT_URL: minio_url,
        COL_URL_MINIO_KEY: "",
        COL_URL_EST_SECS: "",
        COL_URL_AUDIO_PATH: "",
    }


# Row processors
def _snapshot_pair(profiling_enabled: bool):
    """Return a (before, after_callable) pair for GPU profiling."""
    return snapshot(label="before", gpu_index=GPU_INDEX) if profiling_enabled else None


def _make_event(
    idx: int,
    label: str,
    before,
    after,
    profiling_enabled: bool,
) -> Optional[MemoryEvent]:
    if profiling_enabled and before is not None and after is not None:
        return record_event(idx, label, before, after)
    return None


def process_text_row(
    row: Dict[str, str],
    idx: int,
    total: int,
    profiling_enabled: bool,
) -> Tuple[Dict[str, str], Optional[MemoryEvent]]:
    """
    Process one generate_from_text row.

    Returns:
        (result_dict, Optional[MemoryEvent])
    """
    if not _is_valid_text_row(row):
        logger.warning(LOG_SKIPPING_ROW.format(idx=idx))
        return _empty_text_result(""), None

    text = row[COL_INPUT_TEXT].strip()
    language = row[COL_LANGUAGE].strip()
    payload = build_text_payload(text, language)

    logger.info(
        LOG_REQUESTING.format(
            idx=idx,
            total=total,
            label="generate_from_text",
            lang=language,
            preview=_preview(text),
        )
    )

    before = _snapshot_pair(profiling_enabled)

    try:
        response = _post_form(BASE_URL + ENDPOINT_FROM_TEXT, payload, idx, total)
    except (requests.ConnectionError, requests.Timeout):
        return _empty_text_result(text), None

    after = snapshot(label="after", gpu_index=GPU_INDEX) if profiling_enabled else None
    event = _make_event(idx, text, before, after, profiling_enabled)

    if not response.ok:
        logger.error(
            LOG_REQUEST_HTTP_ERROR.format(
                idx=idx,
                total=total,
                status=response.status_code,
                body=response.text[:200],
            )
        )
        return _empty_text_result(text), event

    if OUTPUT_TYPE == "audio":
        saved = _save_audio(response.content, _audio_path_text(idx))
        logger.info(LOG_REQUEST_OK_AUDIO.format(idx=idx, total=total, path=saved))
        return {COL_TXT_INPUT_TEXT: text, COL_TXT_MINIO_KEY: saved}, event

    raw_url = response.json().get(RESPONSE_MINIO_URL_KEY, "")
    full_key = _build_full_minio_key(raw_url) if raw_url else ""
    logger.info(LOG_REQUEST_OK_URL.format(idx=idx, total=total, key=full_key))
    return {COL_TXT_INPUT_TEXT: text, COL_TXT_MINIO_KEY: full_key}, event


def process_url_row(
    row: Dict[str, str],
    idx: int,
    total: int,
    profiling_enabled: bool,
) -> Tuple[Dict[str, str], Optional[MemoryEvent]]:
    """
    Process one generate_from_url row.

    Returns:
        (result_dict, Optional[MemoryEvent])

    All four output columns are always present; unpopulated ones are "".
    """
    if not _is_valid_url_row(row):
        logger.warning(LOG_SKIPPING_ROW.format(idx=idx))
        return _empty_url_result(""), None

    minio_url = row[COL_INPUT_MINIO_URL].strip()
    language = row[COL_LANGUAGE].strip()
    payload = build_url_payload(minio_url, language)

    logger.info(
        LOG_REQUESTING.format(
            idx=idx,
            total=total,
            label="generate_from_url",
            lang=language,
            preview=_preview(minio_url),
        )
    )

    before = _snapshot_pair(profiling_enabled)

    try:
        response = _post_form(BASE_URL + ENDPOINT_FROM_URL, payload, idx, total)
    except (requests.ConnectionError, requests.Timeout):
        return _empty_url_result(minio_url), None

    after = snapshot(label="after", gpu_index=GPU_INDEX) if profiling_enabled else None
    event = _make_event(idx, minio_url, before, after, profiling_enabled)

    if not response.ok:
        logger.error(
            LOG_REQUEST_HTTP_ERROR.format(
                idx=idx,
                total=total,
                status=response.status_code,
                body=response.text[:200],
            )
        )
        return _empty_url_result(minio_url), event

    if OUTPUT_TYPE == "audio":
        saved = _save_audio(response.content, _audio_path_url(idx))
        logger.info(LOG_REQUEST_OK_AUDIO.format(idx=idx, total=total, path=saved))
        return {
            COL_URL_INPUT_URL: minio_url,
            COL_URL_MINIO_KEY: "",
            COL_URL_EST_SECS: "",
            COL_URL_AUDIO_PATH: saved,
        }, event

    resp_json = response.json()
    raw_url = resp_json.get(RESPONSE_MINIO_URL_KEY, "")
    estimated_secs = str(resp_json.get(RESPONSE_ESTIMATED_SECS_KEY, ""))
    logger.info(LOG_REQUEST_OK_URL.format(idx=idx, total=total, key=raw_url))
    return {
        COL_URL_INPUT_URL: minio_url,
        COL_URL_MINIO_KEY: raw_url,
        COL_URL_EST_SECS: estimated_secs,
        COL_URL_AUDIO_PATH: "",
    }, event


# Section runners
def run_text_section(profiling_enabled: bool) -> Tuple[List[MemoryEvent], int, int]:
    """Run the full generate_from_text pass. Returns (events, success, failed)."""
    rows = load_rows(INPUT_TEXT_CSV)
    total = len(rows)
    logger.info(
        LOG_SECTION_START.format(
            label="generate_from_text", output_type=OUTPUT_TYPE, count=total
        )
    )

    pairs = [
        process_text_row(row, idx + 1, total, profiling_enabled)
        for idx, row in enumerate(rows)
    ]
    results = [result for result, _ in pairs]
    events = [event for _, event in pairs if event is not None]

    write_text_section_output(results, OUTPUT_CSV_TEXT)

    success = sum(1 for r in results if r[COL_TXT_MINIO_KEY])
    failed = total - success
    logger.info(
        LOG_PIPELINE_DONE.format(
            label="generate_from_text", success=success, failed=failed
        )
    )
    return events, success, failed


def run_url_section(profiling_enabled: bool) -> Tuple[List[MemoryEvent], int, int]:
    """Run the full generate_from_url pass. Returns (events, success, failed)."""
    rows = load_rows(INPUT_URL_CSV)
    total = len(rows)
    logger.info(
        LOG_SECTION_START.format(
            label="generate_from_url", output_type=OUTPUT_TYPE, count=total
        )
    )

    pairs = [
        process_url_row(row, idx + 1, total, profiling_enabled)
        for idx, row in enumerate(rows)
    ]
    results = [result for result, _ in pairs]
    events = [event for _, event in pairs if event is not None]

    write_url_section_output(results, OUTPUT_CSV_URL)

    success = sum(1 for r in results if r[COL_URL_AUDIO_PATH] or r[COL_URL_MINIO_KEY])
    failed = total - success
    logger.info(
        LOG_PIPELINE_DONE.format(
            label="generate_from_url", success=success, failed=failed
        )
    )
    return events, success, failed


# Endpoint selector
_SECTION_MAP = {
    "text": (run_text_section,),
    "url": (run_url_section,),
    "both": (run_text_section, run_url_section),
}


def _resolve_sections(active: str):
    """Return the tuple of section runner callables for the given selection."""
    normalised = active.strip().lower()
    if normalised not in _SECTION_MAP:
        raise ValueError(
            "ACTIVE_ENDPOINTS must be 'text', 'url', or 'both'. Got: {v}".format(
                v=active
            )
        )
    return _SECTION_MAP[normalised]


# Pipeline orchestration
def run_pipeline() -> None:
    """Entry point — orchestrates the selected TTS endpoint sections."""
    logger.info(LOG_FILE_HANDLER_INIT.format(path=LOG_FILE_PATH))
    logger.info(LOG_PIPELINE_START.format(endpoints=ACTIVE_ENDPOINTS))

    profiling_enabled = False
    try:
        init_nvml()
        profiling_enabled = True
    except Exception as exc:
        logger.warning(LOG_NVML_UNAVAILABLE.format(error=exc))

    sections = _resolve_sections(ACTIVE_ENDPOINTS)
    all_events = [
        event for run_section in sections for event in run_section(profiling_enabled)[0]
    ]

    if profiling_enabled:
        display_summary(all_events, gpu_index=GPU_INDEX)
        shutdown_nvml()


if __name__ == "__main__":
    run_pipeline()
