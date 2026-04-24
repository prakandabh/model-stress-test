"""
MinIO Singleton Client
======================
Provides a single shared MinIO client instance loaded from .env.
Import `get_client()` wherever MinIO access is needed.
"""

import os
from typing import Optional

from dotenv import load_dotenv
from minio import Minio

load_dotenv()


# Error messages
ERR_MISSING_ENV = "Missing required MinIO env var: {key}"
ERR_CLIENT_INIT = "Failed to initialise MinIO client: {error}"
ERR_SECURE_VALUE = "MINIO_SECURE must be 'true' or 'false', got: {value}"


# Required environment variable keys
ENV_ENDPOINT = "MINIO_ENDPOINT"
ENV_ACCESS_KEY = "MINIO_ACCESS_KEY"
ENV_SECRET_KEY = "MINIO_SECRET_KEY"
ENV_SECURE = "MINIO_SECURE"

REQUIRED_ENV_KEYS = (ENV_ENDPOINT, ENV_ACCESS_KEY, ENV_SECRET_KEY, ENV_SECURE)


# Module-level singleton
_client: Optional[Minio] = None


def _resolve_secure(raw: str) -> bool:
    """Parse MINIO_SECURE env var to bool. Raises on invalid value."""
    normalised = raw.strip().lower()
    if normalised not in ("true", "false"):
        raise ValueError(ERR_SECURE_VALUE.format(value=raw))
    return normalised == "true"


def _validate_env() -> None:
    """Raise if any required env var is missing."""
    missing = [k for k in REQUIRED_ENV_KEYS if not os.getenv(k)]
    if missing:
        raise EnvironmentError(ERR_MISSING_ENV.format(key=", ".join(missing)))


def _build_client() -> Minio:
    """Construct a new Minio instance from environment variables."""
    _validate_env()
    try:
        return Minio(
            endpoint=os.getenv(ENV_ENDPOINT),
            access_key=os.getenv(ENV_ACCESS_KEY),
            secret_key=os.getenv(ENV_SECRET_KEY),
            secure=_resolve_secure(os.getenv(ENV_SECURE)),
        )
    except Exception as exc:
        raise RuntimeError(ERR_CLIENT_INIT.format(error=exc)) from exc


def get_client() -> Minio:
    """
    Return the shared MinIO client, initialising it on first call.

    Thread-safety note: acceptable for pipeline scripts where
    initialisation happens before concurrent work begins.
    """
    global _client
    if _client is None:
        _client = _build_client()
    return _client
