"""
ASR Pipeline
============
Calls the deployed ASR service for each row in an input CSV.

Endpoint selection  (set ACTIVE_ENDPOINTS below):
  "stream"  — only /transcribe-from-stream
  "url"     — only /transcribe-from-url
  "all"     — both endpoints in sequence

Input CSVs:
  transcribe-stream   ->  asr_input_stream.csv   (columns: audio_file_path, language)
  transcribe-url      ->  asr_input_url.csv      (columns: input_minio_url, language)

Output behaviour:
  output_type = "text"  ->  CSV with transcription column
  output_type = "url"   ->  CSV with output_minio_url column

Usage:
    python asr_pipeline.py
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

# ---------------------------------------------------------------------------
# Endpoint selection — "stream", "url", or "all"
# ---------------------------------------------------------------------------

ACTIVE_ENDPOINTS = "stream"

# ---------------------------------------------------------------------------
# Service configuration
# ---------------------------------------------------------------------------

BASE_URL             = "http://192.168.88.10:8066"
ENDPOINT_FROM_STREAM = "/transcribe-from-stream"
ENDPOINT_FROM_URL    = "/transcribe-from-url"
REQUEST_TIMEOUT_SEC  = 120
GPU_INDEX            = 0

# ---------------------------------------------------------------------------
# ASR parameters
# ---------------------------------------------------------------------------

OUTPUT_TYPE = "text"         # "text" or "url" (url only valid for stream/url endpoints)

# Required when OUTPUT_TYPE = "url" and endpoint = transcribe-from-stream
URL_OUTPUT_BUCKET       = "asr-output"
URL_OUTPUT_ORGANIZATION = "wiseyak"
URL_OUTPUT_SERVICE      = "asr"
URL_OUTPUT_USER_ID      = "pipeline"

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------

INPUT_STREAM_CSV  = "asr_input_stream.csv"
INPUT_URL_CSV     = "asr_input_url.csv"

OUTPUT_CSV_STREAM = "asr_output_stream.csv"
OUTPUT_CSV_URL    = "asr_output_url.csv"

LOG_FILE_PATH = "asr_pipeline.log"

# ---------------------------------------------------------------------------
# Write mode controls
# ---------------------------------------------------------------------------

# True  → append new runs to existing output CSVs
# False → overwrite output CSVs on each run
CSV_APPEND_MODE = True

# True  → append new runs to existing log file
# False → overwrite log file on each run
LOG_APPEND_MODE = True

# ---------------------------------------------------------------------------
# CSV column names — input
# ---------------------------------------------------------------------------

COL_AUDIO_FILE_PATH = "audio_file_path"
COL_INPUT_MINIO_URL = "input_minio_url"
COL_LANGUAGE        = "language"

# ---------------------------------------------------------------------------
# CSV column names — output: transcribe-from-stream
# ---------------------------------------------------------------------------

COL_STREAM_INPUT         = "audio_file_path"
COL_STREAM_LANGUAGE      = "language"
COL_STREAM_TRANSCRIPTION = "transcription"       # populated when output_type = "text"
COL_STREAM_MINIO_URL     = "output_minio_url"    # populated when output_type = "url"

# ---------------------------------------------------------------------------
# CSV column names — output: transcribe-from-url
# ---------------------------------------------------------------------------

COL_URL_INPUT         = "input_minio_url"
COL_URL_LANGUAGE      = "language"
COL_URL_TRANSCRIPTION = "transcription"          # populated when output_type = "text"
COL_URL_MINIO_URL     = "output_minio_url"       # populated when output_type = "url"

# ---------------------------------------------------------------------------
# Form / file field names (must match FastAPI endpoint signatures)
# ---------------------------------------------------------------------------

FORM_AUDIO          = "audio"
FORM_MINIO_URL      = "minio_url"
FORM_LANGUAGE       = "language"
FORM_OUTPUT_TYPE    = "output_type"
FORM_BUCKET_NAME    = "bucket_name"
FORM_ORGANIZATION   = "organization"
FORM_SERVICE        = "service"
FORM_USER_ID        = "user_id"

# ---------------------------------------------------------------------------
# Response keys
# ---------------------------------------------------------------------------

RESPONSE_TRANSCRIPTION_KEY = "text"
RESPONSE_MINIO_URL_KEY     = "minio_url"

# ---------------------------------------------------------------------------
# Audio MIME type for multipart upload
# ---------------------------------------------------------------------------

AUDIO_MIME_TYPE = "audio/wav"

# ---------------------------------------------------------------------------
# Log message templates — .format() only, no f-strings
# ---------------------------------------------------------------------------

LOG_PIPELINE_START     = "ASR pipeline started — endpoint(s): {endpoints}"
LOG_FILE_HANDLER_INIT  = "File logging active — {path}"
LOG_SECTION_START      = "--- Running: {label} | output_type: {output_type} | rows: {count} ---"
LOG_LOADED_ROWS        = "Loaded {count} rows from {path}"
LOG_REQUESTING         = "[{idx}/{total}] {label}: {preview!r}"
LOG_REQUEST_OK_TEXT    = "[{idx}/{total}] OK — transcription: {preview!r}"
LOG_REQUEST_OK_URL     = "[{idx}/{total}] OK — output_minio_url: {url}"
LOG_REQUEST_HTTP_ERROR = "[{idx}/{total}] HTTP {status} — {body}"
LOG_REQUEST_CONN_ERROR = "[{idx}/{total}] Connection error: {error}"
LOG_REQUEST_TIMEOUT    = "[{idx}/{total}] Timeout after {timeout}s"
LOG_AUDIO_NOT_FOUND    = "[{idx}/{total}] Audio file not found: {path}"
LOG_SKIPPING_ROW       = "Skipping row {idx} — missing required fields"
LOG_WROTE_CSV          = "Wrote {count} rows to {path}"
LOG_PIPELINE_DONE      = "{label} complete — {success} succeeded, {failed} failed"
LOG_NVML_UNAVAILABLE   = "pynvml unavailable — CUDA profiling disabled: {error}"

# ---------------------------------------------------------------------------
# Logging setup — stream + file handler
# ---------------------------------------------------------------------------

_LOG_FORMAT  = "%(asctime)s  %(levelname)-8s  %(message)s"
_LOG_DATEFMT = "%H:%M:%S"

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

_file_handler = logging.FileHandler(
    LOG_FILE_PATH, mode="a" if LOG_APPEND_MODE else "w", encoding="utf-8"
)
_file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_LOG_DATEFMT))

logging.basicConfig(level=logging.INFO, handlers=[_stream_handler, _file_handler])
logger = logging.getLogger(Path(__file__).stem)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEXT_PREVIEW_LEN = 40


def _preview(text: str) -> str:
    return text[:TEXT_PREVIEW_LEN] + "…" if len(text) > TEXT_PREVIEW_LEN else text


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


def load_rows(csv_path: str) -> List[Dict[str, str]]:
    """Read all rows from a CSV and return as list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info(LOG_LOADED_ROWS.format(count=len(rows), path=csv_path))
    return rows


def _open_csv(csv_path: str, fieldnames: List[str]):
    """
    Return an open (fh, writer) pair respecting CSV_APPEND_MODE.

    Appends to existing file (skipping header) when CSV_APPEND_MODE is True.
    Overwrites the file (writing header) when CSV_APPEND_MODE is False.
    """
    file_exists  = Path(csv_path).exists()
    write_header = (not CSV_APPEND_MODE) or (not file_exists)
    mode = "a" if CSV_APPEND_MODE else "w"
    fh   = open(csv_path, mode, newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    return fh, writer


def _write_csv(results: List[Dict[str, str]], csv_path: str, fieldnames: List[str]) -> None:
    """Write results to CSV using the configured write mode."""
    fh, writer = _open_csv(csv_path, fieldnames)
    writer.writerows(results)
    fh.close()
    logger.info(LOG_WROTE_CSV.format(count=len(results), path=csv_path))


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _is_valid_stream_row(row: Dict[str, str]) -> bool:
    return all(row.get(col, "").strip() for col in (COL_AUDIO_FILE_PATH, COL_LANGUAGE))


def _is_valid_url_row(row: Dict[str, str]) -> bool:
    return all(row.get(col, "").strip() for col in (COL_INPUT_MINIO_URL, COL_LANGUAGE))


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _post_multipart(
    endpoint_url: str,
    audio_path: str,
    data: Dict,
    idx: int,
    total: int,
) -> requests.Response:
    """
    POST multipart/form-data with an audio file upload.
    Raises on connection or timeout errors.
    """
    with open(audio_path, "rb") as audio_fh:
        files = {FORM_AUDIO: (Path(audio_path).name, audio_fh, AUDIO_MIME_TYPE)}
        try:
            return requests.post(
                endpoint_url, files=files, data=data, timeout=REQUEST_TIMEOUT_SEC
            )
        except requests.ConnectionError as exc:
            logger.error(LOG_REQUEST_CONN_ERROR.format(idx=idx, total=total, error=exc))
            raise
        except requests.Timeout:
            logger.error(LOG_REQUEST_TIMEOUT.format(idx=idx, total=total, timeout=REQUEST_TIMEOUT_SEC))
            raise


def _post_form(
    endpoint_url: str,
    data: Dict,
    idx: int,
    total: int,
) -> requests.Response:
    """POST form-encoded data (no file upload). Raises on connection or timeout errors."""
    try:
        return requests.post(endpoint_url, data=data, timeout=REQUEST_TIMEOUT_SEC)
    except requests.ConnectionError as exc:
        logger.error(LOG_REQUEST_CONN_ERROR.format(idx=idx, total=total, error=exc))
        raise
    except requests.Timeout:
        logger.error(LOG_REQUEST_TIMEOUT.format(idx=idx, total=total, timeout=REQUEST_TIMEOUT_SEC))
        raise


def _extract_transcription(resp_json: Dict) -> str:
    return resp_json.get(RESPONSE_TRANSCRIPTION_KEY, "")


def _extract_minio_url(resp_json: Dict) -> str:
    return resp_json.get(RESPONSE_MINIO_URL_KEY, "")


# ---------------------------------------------------------------------------
# GPU profiling helpers
# ---------------------------------------------------------------------------


def _before_snapshot(profiling_enabled: bool):
    return snapshot(label="before", gpu_index=GPU_INDEX) if profiling_enabled else None


def _after_snapshot(profiling_enabled: bool):
    return snapshot(label="after", gpu_index=GPU_INDEX) if profiling_enabled else None


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


# ---------------------------------------------------------------------------
# Empty result builders
# ---------------------------------------------------------------------------


def _empty_stream(audio_path: str, language: str) -> Dict[str, str]:
    return {
        COL_STREAM_INPUT:         audio_path,
        COL_STREAM_LANGUAGE:      language,
        COL_STREAM_TRANSCRIPTION: "",
        COL_STREAM_MINIO_URL:     "",
    }


def _empty_url_result(minio_url: str, language: str) -> Dict[str, str]:
    return {
        COL_URL_INPUT:         minio_url,
        COL_URL_LANGUAGE:      language,
        COL_URL_TRANSCRIPTION: "",
        COL_URL_MINIO_URL:     "",
    }


# ---------------------------------------------------------------------------
# Row processors
# ---------------------------------------------------------------------------


def process_stream_row(
    row: Dict[str, str],
    idx: int,
    total: int,
    profiling_enabled: bool,
) -> Tuple[Dict[str, str], Optional[MemoryEvent]]:
    """
    Process one transcribe-from-stream row.

    output_type = "text" → transcription column populated
    output_type = "url"  → output_minio_url column populated
    """
    if not _is_valid_stream_row(row):
        logger.warning(LOG_SKIPPING_ROW.format(idx=idx))
        return _empty_stream("", ""), None

    audio_path = row[COL_AUDIO_FILE_PATH].strip()
    language   = row[COL_LANGUAGE].strip()

    if not Path(audio_path).exists():
        logger.error(LOG_AUDIO_NOT_FOUND.format(idx=idx, total=total, path=audio_path))
        return _empty_stream(audio_path, language), None

    logger.info(LOG_REQUESTING.format(
        idx=idx, total=total, label="transcribe-from-stream ({lang})".format(lang=language),
        preview=_preview(audio_path),
    ))

    base_data = {FORM_LANGUAGE: language, FORM_OUTPUT_TYPE: OUTPUT_TYPE}
    data = (
        {
            **base_data,
            FORM_BUCKET_NAME:  URL_OUTPUT_BUCKET,
            FORM_ORGANIZATION: URL_OUTPUT_ORGANIZATION,
            FORM_SERVICE:      URL_OUTPUT_SERVICE,
            FORM_USER_ID:      URL_OUTPUT_USER_ID,
        }
        if OUTPUT_TYPE == "url"
        else base_data
    )

    before = _before_snapshot(profiling_enabled)

    try:
        response = _post_multipart(
            BASE_URL + ENDPOINT_FROM_STREAM, audio_path, data, idx, total
        )
    except (requests.ConnectionError, requests.Timeout):
        return _empty_stream(audio_path, language), None

    after = _after_snapshot(profiling_enabled)
    event = _make_event(idx, audio_path, before, after, profiling_enabled)

    if not response.ok:
        logger.error(LOG_REQUEST_HTTP_ERROR.format(
            idx=idx, total=total, status=response.status_code, body=response.text[:200]
        ))
        return _empty_stream(audio_path, language), event

    resp_json = response.json()

    if OUTPUT_TYPE == "url":
        out_url = _extract_minio_url(resp_json)
        logger.info(LOG_REQUEST_OK_URL.format(idx=idx, total=total, url=out_url))
        return {
            COL_STREAM_INPUT:         audio_path,
            COL_STREAM_LANGUAGE:      language,
            COL_STREAM_TRANSCRIPTION: "",
            COL_STREAM_MINIO_URL:     out_url,
        }, event

    transcription = _extract_transcription(resp_json)
    logger.info(LOG_REQUEST_OK_TEXT.format(idx=idx, total=total, preview=_preview(transcription)))
    return {
        COL_STREAM_INPUT:         audio_path,
        COL_STREAM_LANGUAGE:      language,
        COL_STREAM_TRANSCRIPTION: transcription,
        COL_STREAM_MINIO_URL:     "",
    }, event


def process_url_row(
    row: Dict[str, str],
    idx: int,
    total: int,
    profiling_enabled: bool,
) -> Tuple[Dict[str, str], Optional[MemoryEvent]]:
    """
    Process one transcribe-from-url row.

    output_type = "text" → transcription column populated
    output_type = "url"  → output_minio_url column populated
    """
    if not _is_valid_url_row(row):
        logger.warning(LOG_SKIPPING_ROW.format(idx=idx))
        return _empty_url_result("", ""), None

    minio_url = row[COL_INPUT_MINIO_URL].strip()
    language  = row[COL_LANGUAGE].strip()

    logger.info(LOG_REQUESTING.format(
        idx=idx, total=total, label="transcribe-from-url ({lang})".format(lang=language),
        preview=_preview(minio_url),
    ))

    data = {
        FORM_MINIO_URL:   minio_url,
        FORM_LANGUAGE:    language,
        FORM_OUTPUT_TYPE: OUTPUT_TYPE,
    }

    before = _before_snapshot(profiling_enabled)

    try:
        response = _post_form(BASE_URL + ENDPOINT_FROM_URL, data, idx, total)
    except (requests.ConnectionError, requests.Timeout):
        return _empty_url_result(minio_url, language), None

    after = _after_snapshot(profiling_enabled)
    event = _make_event(idx, minio_url, before, after, profiling_enabled)

    if not response.ok:
        logger.error(LOG_REQUEST_HTTP_ERROR.format(
            idx=idx, total=total, status=response.status_code, body=response.text[:200]
        ))
        return _empty_url_result(minio_url, language), event

    resp_json = response.json()

    if OUTPUT_TYPE == "url":
        out_url = _extract_minio_url(resp_json)
        logger.info(LOG_REQUEST_OK_URL.format(idx=idx, total=total, url=out_url))
        return {
            COL_URL_INPUT:         minio_url,
            COL_URL_LANGUAGE:      language,
            COL_URL_TRANSCRIPTION: "",
            COL_URL_MINIO_URL:     out_url,
        }, event

    transcription = _extract_transcription(resp_json)
    logger.info(LOG_REQUEST_OK_TEXT.format(idx=idx, total=total, preview=_preview(transcription)))
    return {
        COL_URL_INPUT:         minio_url,
        COL_URL_LANGUAGE:      language,
        COL_URL_TRANSCRIPTION: transcription,
        COL_URL_MINIO_URL:     "",
    }, event


# ---------------------------------------------------------------------------
# Section runners
# ---------------------------------------------------------------------------


def run_stream_section(profiling_enabled: bool) -> Tuple[List[MemoryEvent], int, int]:
    """Run the full transcribe-from-stream pass. Returns (events, success, failed)."""
    rows  = load_rows(INPUT_STREAM_CSV)
    total = len(rows)
    logger.info(LOG_SECTION_START.format(
        label="transcribe-from-stream", output_type=OUTPUT_TYPE, count=total
    ))

    pairs   = [process_stream_row(row, idx + 1, total, profiling_enabled)
               for idx, row in enumerate(rows)]
    results = [result for result, _ in pairs]
    events  = [event  for _, event  in pairs if event is not None]

    _write_csv(
        results, OUTPUT_CSV_STREAM,
        [COL_STREAM_INPUT, COL_STREAM_LANGUAGE, COL_STREAM_TRANSCRIPTION, COL_STREAM_MINIO_URL],
    )

    success = sum(
        1 for r in results
        if r[COL_STREAM_TRANSCRIPTION] or r[COL_STREAM_MINIO_URL]
    )
    failed  = total - success
    logger.info(LOG_PIPELINE_DONE.format(
        label="transcribe-from-stream", success=success, failed=failed
    ))
    return events, success, failed


def run_url_section(profiling_enabled: bool) -> Tuple[List[MemoryEvent], int, int]:
    """Run the full transcribe-from-url pass. Returns (events, success, failed)."""
    rows  = load_rows(INPUT_URL_CSV)
    total = len(rows)
    logger.info(LOG_SECTION_START.format(
        label="transcribe-from-url", output_type=OUTPUT_TYPE, count=total
    ))

    pairs   = [process_url_row(row, idx + 1, total, profiling_enabled)
               for idx, row in enumerate(rows)]
    results = [result for result, _ in pairs]
    events  = [event  for _, event  in pairs if event is not None]

    _write_csv(
        results, OUTPUT_CSV_URL,
        [COL_URL_INPUT, COL_URL_LANGUAGE, COL_URL_TRANSCRIPTION, COL_URL_MINIO_URL],
    )

    success = sum(
        1 for r in results
        if r[COL_URL_TRANSCRIPTION] or r[COL_URL_MINIO_URL]
    )
    failed  = total - success
    logger.info(LOG_PIPELINE_DONE.format(
        label="transcribe-from-url", success=success, failed=failed
    ))
    return events, success, failed


# ---------------------------------------------------------------------------
# Endpoint selector
# ---------------------------------------------------------------------------

_SECTION_MAP = {
    "stream": (run_stream_section,),
    "url":    (run_url_section,),
    "all":    (run_stream_section, run_url_section),
}


def _resolve_sections(active: str):
    """Return the tuple of section runner callables for the given selection."""
    normalised = active.strip().lower()
    if normalised not in _SECTION_MAP:
        raise ValueError(
            "ACTIVE_ENDPOINTS must be 'stream', 'url', or 'all'. Got: {v}".format(v=active)
        )
    return _SECTION_MAP[normalised]


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline() -> None:
    """Entry point — orchestrates the selected ASR endpoint sections."""
    logger.info(LOG_FILE_HANDLER_INIT.format(path=LOG_FILE_PATH))
    logger.info(LOG_PIPELINE_START.format(endpoints=ACTIVE_ENDPOINTS))

    profiling_enabled = False
    try:
        init_nvml()
        profiling_enabled = True
    except Exception as exc:
        logger.warning(LOG_NVML_UNAVAILABLE.format(error=exc))

    sections   = _resolve_sections(ACTIVE_ENDPOINTS)
    all_events = [
        event
        for run_section in sections
        for event in run_section(profiling_enabled)[0]
    ]

    if profiling_enabled:
        display_summary(all_events, gpu_index=GPU_INDEX)
        shutdown_nvml()


if __name__ == "__main__":
    run_pipeline()