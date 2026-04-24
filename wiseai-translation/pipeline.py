"""
Translation Pipeline
====================
Reads input CSV (text, input_language, output_language),
calls the deployed translation service for each row,
and writes output CSV (input_text, translated_text).

Usage:
    python translation_pipeline.py
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

# Configuration — edit these before running


BASE_URL = "http://192.168.88.10:8999"
ENDPOINT = "/translate"

INPUT_CSV_PATH = (
    "/home/oem/wiseyak/prakanda/stree-test/wiseai-translation/input_data.csv"
)
OUTPUT_CSV_PATH = (
    "/home/oem/wiseyak/prakanda/stree-test/wiseai-translation/translation_output.csv"
)

REQUEST_TIMEOUT_SECONDS = 60
GPU_INDEX = 0

# Request body field names (match your FastAPI schema)
FIELD_TEXTS = "text"
FIELD_SOURCE_LANGUAGE = "input_language"
FIELD_TARGET_LANGUAGE = "output_language"

# Response body field names (match your TranslationResponse schema)
RESPONSE_TRANSLATIONS_KEY = "translations"
RESPONSE_TRANSLATED_TEXT = "translated"

# CSV column names (input)
CSV_COL_TEXT = "text"
CSV_COL_INPUT_LANGUAGE = "input_language"
CSV_COL_OUTPUT_LANGUAGE = "output_language"

# CSV column names (output)
OUT_COL_INPUT_TEXT = "input_text"
OUT_COL_TRANSLATED_TEXT = "translated_text"


# Log message templates
LOG_PIPELINE_START = "Translation pipeline started — input: {input}, output: {output}"
LOG_LOADED_ROWS = "Loaded {count} rows from {path}"
LOG_REQUESTING = "[{idx}/{total}] Translating ({src} → {tgt}): {preview!r}"
LOG_REQUEST_SUCCESS = "[{idx}/{total}] OK — {preview!r}"
LOG_REQUEST_ERROR = "[{idx}/{total}] HTTP {status} — {text}"
LOG_CONNECTION_ERROR = "[{idx}/{total}] Connection error: {error}"
LOG_TIMEOUT_ERROR = "[{idx}/{total}] Request timed out after {timeout}s"
LOG_WROTE_OUTPUT = "Wrote {count} rows to {path}"
LOG_PIPELINE_DONE = "Pipeline complete — {success} succeeded, {failed} failed"
LOG_NVML_UNAVAILABLE = "pynvml unavailable — CUDA profiling disabled: {error}"
LOG_SKIPPING_ROW = "Skipping row {idx} — missing required fields"


# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(Path(__file__).stem)


# Pure functions
def load_rows(csv_path: str) -> List[Dict[str, str]]:
    """Read all rows from the input CSV and return as a list of dicts."""
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    logger.info(LOG_LOADED_ROWS.format(count=len(rows), path=csv_path))
    return rows


def _is_valid_row(row: Dict[str, str]) -> bool:
    """Return True if the row contains all required fields."""
    return all(
        row.get(col, "").strip()
        for col in (CSV_COL_TEXT, CSV_COL_INPUT_LANGUAGE, CSV_COL_OUTPUT_LANGUAGE)
    )


def build_payload(row: Dict[str, str]) -> Dict:
    """Construct the JSON payload for the translation endpoint."""
    return {
        FIELD_TEXTS: row[CSV_COL_TEXT].strip(),
        FIELD_SOURCE_LANGUAGE: row[CSV_COL_INPUT_LANGUAGE].strip(),
        FIELD_TARGET_LANGUAGE: row[CSV_COL_OUTPUT_LANGUAGE].strip(),
    }


def _extract_translated_text(response_json: Dict) -> str:
    """Pull the first translated text from the service response."""
    translations = response_json.get(RESPONSE_TRANSLATIONS_KEY, [])
    if not translations:
        return ""
    first = translations[0]
    # Support both dict-style and plain-string translation entries
    return (
        first.get(RESPONSE_TRANSLATED_TEXT, "")
        if isinstance(first, dict)
        else str(first)
    )


def call_service(
    payload: Dict,
    idx: int,
    total: int,
) -> Tuple[str, bool]:
    """
    POST payload to the translation service.

    Returns:
        (translated_text, success_flag)
    """
    url = BASE_URL + ENDPOINT
    preview = payload[FIELD_TEXTS][:40]

    logger.info(
        LOG_REQUESTING.format(
            idx=idx,
            total=total,
            src=payload[FIELD_SOURCE_LANGUAGE],
            tgt=payload[FIELD_TARGET_LANGUAGE],
            preview=preview,
        )
    )

    try:
        response = requests.post(url, data=payload, timeout=REQUEST_TIMEOUT_SECONDS)

        if not response.ok:
            logger.error(
                LOG_REQUEST_ERROR.format(
                    idx=idx,
                    total=total,
                    status=response.status_code,
                    text=response.text,
                )
            )

        if response.ok:
            translated = _extract_translated_text(response.json())
            logger.info(
                LOG_REQUEST_SUCCESS.format(idx=idx, total=total, preview=preview)
            )
            return translated, True

        logger.error(
            LOG_REQUEST_ERROR.format(
                idx=idx,
                total=total,
                status=response.status_code,
                text=response.text[:120],
            )
        )
        return "", False

    except requests.ConnectionError as exc:
        logger.error(LOG_CONNECTION_ERROR.format(idx=idx, total=total, error=exc))
        return "", False

    except requests.Timeout:
        logger.error(
            LOG_TIMEOUT_ERROR.format(
                idx=idx, total=total, timeout=REQUEST_TIMEOUT_SECONDS
            )
        )
        return "", False


def process_row(
    row: Dict[str, str],
    idx: int,
    total: int,
    profiling_enabled: bool,
) -> Tuple[Dict[str, str], Optional[MemoryEvent]]:
    """
    Translate a single CSV row, optionally bracketing the call with GPU snapshots.

    Returns:
        (result_dict, memory_event_or_None)
    """
    if not _is_valid_row(row):
        logger.warning(LOG_SKIPPING_ROW.format(idx=idx))
        return {
            OUT_COL_INPUT_TEXT: row.get(CSV_COL_TEXT, ""),
            OUT_COL_TRANSLATED_TEXT: "",
        }, None

    payload = build_payload(row)
    text = payload[FIELD_TEXTS]

    before = (
        snapshot(label="before", gpu_index=GPU_INDEX) if profiling_enabled else None
    )
    translated, _ = call_service(payload, idx, total)
    after = snapshot(label="after", gpu_index=GPU_INDEX) if profiling_enabled else None

    event = (
        record_event(idx, text, before, after)
        if (profiling_enabled and before is not None and after is not None)
        else None
    )

    return {OUT_COL_INPUT_TEXT: text, OUT_COL_TRANSLATED_TEXT: translated}, event


def write_output(results: List[Dict[str, str]], csv_path: str) -> None:
    """Write result rows to the output CSV."""
    fieldnames = [OUT_COL_INPUT_TEXT, OUT_COL_TRANSLATED_TEXT]
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    logger.info(LOG_WROTE_OUTPUT.format(count=len(results), path=csv_path))


# Pipeline orchestration
def run_pipeline() -> None:
    """Entry point — orchestrates the full translation pipeline."""
    logger.info(LOG_PIPELINE_START.format(input=INPUT_CSV_PATH, output=OUTPUT_CSV_PATH))

    # Attempt to enable GPU profiling
    profiling_enabled = False
    try:
        init_nvml()
        profiling_enabled = True
    except Exception as exc:
        logger.warning(LOG_NVML_UNAVAILABLE.format(error=exc))

    rows = load_rows(INPUT_CSV_PATH)
    total = len(rows)

    pairs = [
        process_row(row, idx + 1, total, profiling_enabled)
        for idx, row in enumerate(rows)
    ]

    results = [result for result, _ in pairs]
    memory_events = [event for _, event in pairs if event is not None]

    write_output(results, OUTPUT_CSV_PATH)

    success = sum(1 for r in results if r[OUT_COL_TRANSLATED_TEXT])
    failed = total - success
    logger.info(LOG_PIPELINE_DONE.format(success=success, failed=failed))

    if profiling_enabled:
        display_summary(memory_events, gpu_index=GPU_INDEX)
        shutdown_nvml()


if __name__ == "__main__":
    run_pipeline()
