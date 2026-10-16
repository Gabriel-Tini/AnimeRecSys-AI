"""
backend/app/core/config.py
AnimeRecSys AI — Core Configuration & Security Base Module
Loads application settings, environment variables, isolated credentials,
strict workspace paths, and sets up security-redacted logging.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
import traceback
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Literal, Optional, Union

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# -----------------------------------------------------------------------------
# Canonical Path Anchoring
# -----------------------------------------------------------------------------
# Directory structure reference:
# <workspace_root>/backend/app/core/config.py
# parents[0] -> backend/app/core
# parents[1] -> backend/app
# parents[2] -> backend
# parents[3] -> <workspace_root> (e.g. c:\Dev\Ent ML)

CURRENT_FILE = Path(__file__).resolve()
DEFAULT_BACKEND_DIR = CURRENT_FILE.parents[2]
DEFAULT_WORKSPACE_ROOT = CURRENT_FILE.parents[3]


class Settings(BaseSettings):
    """
    Application Settings bound to environment variables and backend/.env.
    Guarantees secret isolation, path canonicalization, and strict validation.
    """

    model_config = SettingsConfigDict(
        env_file=(str(DEFAULT_BACKEND_DIR / ".env"), ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -------------------------------------------------------------------------
    # Application & Environment
    # -------------------------------------------------------------------------
    app_name: str = Field(default="AnimeRecSys AI", description="Application Name")
    environment: Literal["development", "production", "test"] = Field(
        default="development",
        alias="ENVIRONMENT",
        description="Operating environment stage"
    )
    debug: bool = Field(default=False, description="Debug mode flag")
    host: str = Field(default="127.0.0.1", alias="HOST", description="API bind host")
    port: int = Field(default=8000, alias="PORT", description="API bind port")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL", description="Log level")

    # -------------------------------------------------------------------------
    # Secret Isolation: Google Gemini API
    # -------------------------------------------------------------------------
    gemini_api_key: Optional[SecretStr] = Field(
        default=None,
        alias="GEMINI_API_KEY",
        description="Google Gemini API Key for semantic tuning (isolated in backend/.env)"
    )
    gemini_model: str = Field(
        default="gemini-2.5-flash",
        alias="GEMINI_MODEL",
        description="Target Gemini model name"
    )
    gemini_temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Low temperature for deterministic semantic extraction"
    )
    gemini_max_feedback_length: int = Field(
        default=300,
        ge=10,
        le=1000,
        description="Maximum character length for qualitative input"
    )

    # -------------------------------------------------------------------------
    # External Services: AniList GraphQL & Rate Limiting
    # -------------------------------------------------------------------------
    anilist_api_url: str = Field(
        default="https://graphql.anilist.co",
        alias="ANILIST_API_URL",
        description="AniList GraphQL API endpoint"
    )
    anilist_rate_limit_per_minute: int = Field(
        default=60,
        ge=1,
        le=90,
        alias="ANILIST_RATE_LIMIT",
        description="Strict client-side token bucket rate limit (max 60 req/min)"
    )
    anilist_request_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="Timeout for AniList GraphQL HTTP requests"
    )
    anilist_retry_after_default: float = Field(
        default=60.0,
        description="Default backoff duration in seconds when HTTP 429 is encountered"
    )

    # -------------------------------------------------------------------------
    # CORS Restrictions
    # -------------------------------------------------------------------------
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:5173", "http://127.0.0.1:5173"],
        alias="CORS_ORIGINS",
        description="Allowed origins for CORS (wildcards '*' strictly prohibited)"
    )

    # -------------------------------------------------------------------------
    # Recommendation System Hyperparameters
    # -------------------------------------------------------------------------
    rec_eta: float = Field(
        default=0.15,
        ge=0.01,
        le=1.0,
        alias="REC_ETA",
        description="Learning rate per rating update (default 0.15)"
    )
    rec_alpha: float = Field(
        default=0.25,
        ge=0.01,
        le=1.0,
        alias="REC_ALPHA",
        description="Semantic boost multiplier for LLM qualitative insights (default 0.25)"
    )
    rec_gamma: float = Field(
        default=0.98,
        ge=0.50,
        le=0.999,
        alias="REC_GAMMA",
        description="Temporal weight decay factor (default 0.98)"
    )
    rec_lambda_mmr: float = Field(
        default=0.50,
        ge=0.0,
        le=1.0,
        alias="REC_LAMBDA_MMR",
        description="Maximal Marginal Relevance diversity balance parameter (default 0.50)"
    )

    # -------------------------------------------------------------------------
    # Concurrency & Persistence Settings
    # -------------------------------------------------------------------------
    filelock_timeout_seconds: float = Field(
        default=10.0,
        ge=1.0,
        le=60.0,
        description="FileLock acquisition timeout for atomic data file operations"
    )
    catalog_cache_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="In-memory cache TTL for catalog search queries"
    )

    # -------------------------------------------------------------------------
    # Path Resolution & Anchors
    # -------------------------------------------------------------------------
    custom_workspace_root: Optional[Path] = Field(
        default=None,
        alias="WORKSPACE_ROOT",
        description="Optional override for workspace root directory"
    )

    # -------------------------------------------------------------------------
    # Field Validators
    # -------------------------------------------------------------------------
    @field_validator("cors_origins", mode="before")
    @classmethod
    def validate_cors_origins(cls, value: Any) -> List[str]:
        """
        Parses CORS origins from JSON string, comma-separated string, or list,
        and enforces strict security rejection of wildcards ('*').
        """
        origins: List[str] = []
        if isinstance(value, str):
            value = value.strip()
            if value.startswith("[") and value.endswith("]"):
                try:
                    parsed = json.loads(value)
                    if isinstance(parsed, list):
                        origins = [str(item).strip() for item in parsed]
                except json.JSONDecodeError as err:
                    raise ValueError(f"Invalid JSON format for CORS_ORIGINS: {err}") from err
            else:
                origins = [item.strip() for item in value.split(",") if item.strip()]
        elif isinstance(value, (list, tuple, set)):
            origins = [str(item).strip() for item in value]
        else:
            raise ValueError(f"Unsupported type for CORS_ORIGINS: {type(value)}")

        if not origins:
            raise ValueError("CORS_ORIGINS cannot be empty.")

        cleaned: List[str] = []
        for origin in origins:
            if origin == "*" or "*" in origin:
                raise ValueError(
                    f"CORS wildcard origin '{origin}' is strictly prohibited by SECURITY.md"
                )
            # Remove trailing slashes for standard CORS origin matching
            cleaned.append(origin.rstrip("/"))

        return cleaned

    # -------------------------------------------------------------------------
    # Derived Path Properties
    # -------------------------------------------------------------------------
    @property
    def workspace_root(self) -> Path:
        """Returns the canonical absolute workspace root path."""
        if self.custom_workspace_root is not None:
            return self.custom_workspace_root.resolve()
        return DEFAULT_WORKSPACE_ROOT

    @property
    def backend_dir(self) -> Path:
        """Returns the absolute path to the backend directory."""
        return self.workspace_root / "backend"

    @property
    def data_dir(self) -> Path:
        """Returns the absolute path to the data persistence directory."""
        return self.workspace_root / "data"

    @property
    def locks_dir(self) -> Path:
        """Returns the absolute path to the data locks directory."""
        return self.data_dir / ".locks"

    @property
    def catalog_cache_path(self) -> Path:
        """Returns the absolute path to cache_catalog.json."""
        return self.data_dir / "cache_catalog.json"

    @property
    def user_profile_path(self) -> Path:
        """Returns the absolute path to user_profile.json."""
        return self.data_dir / "user_profile.json"

    @property
    def watch_history_path(self) -> Path:
        """Returns the absolute path to watch_history.json."""
        return self.data_dir / "watch_history.json"

    @property
    def fixtures_path(self) -> Path:
        """Returns the absolute path to .antigravity/FIXTURES.json."""
        return self.workspace_root / ".antigravity" / "FIXTURES.json"

    @property
    def is_gemini_configured(self) -> bool:
        """
        Verifies if GEMINI_API_KEY is present and not set to a placeholder.
        Enables transparent fallback to fixtures when False.
        """
        if self.gemini_api_key is None:
            return False
        val = self.gemini_api_key.get_secret_value().strip()
        if not val or val == "your_gemini_api_key_here":
            return False
        return True

    def resolve_path(self, path: Union[str, Path]) -> Path:
        """
        Resolves a path strictly against workspace_root if relative,
        or returns resolved absolute path if already absolute.
        """
        p = Path(path)
        if p.is_absolute():
            return p.resolve()
        return (self.workspace_root / p).resolve()

    def ensure_directories(self) -> None:
        """Ensures that required data and lock directories exist on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Security Redaction Logging Filter
# -----------------------------------------------------------------------------
class SensitiveDataFilter(logging.Filter):
    """
    Logging filter that intercepts and sanitizes sensitive data (API keys,
    Bearer tokens, secret query parameters) from log messages, args, and tracebacks.
    """

    # Matches Google/Gemini API keys (e.g. AIzaSyA1234567890abcdefghijklmnopqrstuv)
    # Includes (?<=m) so it does not fail when preceded by ANSI color escape codes (e.g. \x1b[31m)
    GEMINI_KEY_PATTERN = re.compile(
        r"(?:(?<=[^0-9A-Za-z\-_])|(?<=m)|^)(?:AIza[0-9A-Za-z\-_]{28,45})(?=[^0-9A-Za-z\-_]|$)"
    )
    # Matches Bearer authorization headers
    BEARER_PATTERN = re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*")
    # Matches sensitive query parameters in URLs
    QUERY_PARAM_PATTERN = re.compile(
        r"(?i)(?:key|apikey|secret|token)=([^&\s]+)"
    )
    # Matches explicit header dumps
    HEADER_PATTERN = re.compile(
        r"(?i)(?:x-goog-api-key|authorization):\s*([^\s,]+)"
    )

    # Standard LogRecord attributes that should not be treated as custom extra fields
    STANDARD_RECORD_ATTRS = frozenset({
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    })

    def __init__(self, exact_secrets: Optional[List[str]] = None) -> None:
        super().__init__()
        self.exact_secrets = [s for s in (exact_secrets or []) if s and len(s) > 4]

    def redact_text(self, text: str) -> str:
        """Scrubs all sensitive patterns and exact secrets from a string."""
        if not text:
            return text

        # 1. Redact Google API key regex (e.g. AIza...)
        text = self.GEMINI_KEY_PATTERN.sub("[REDACTED_GEMINI_KEY]", text)

        # 2. Redact exact configured secrets
        for secret in self.exact_secrets:
            text = text.replace(secret, "[REDACTED_SECRET]")

        # 3. Redact Bearer tokens
        text = self.BEARER_PATTERN.sub("Bearer [REDACTED_TOKEN]", text)

        # 4. Redact query params
        text = self.QUERY_PARAM_PATTERN.sub("key=[REDACTED]", text)

        # 5. Redact headers
        text = self.HEADER_PATTERN.sub(lambda m: f"{m.group(0).split(':')[0]}: [REDACTED]", text)

        return text

    def _safe_format_exception(self, exc_info: Any) -> str:
        """Safely formats exc_info into a traceback string without raising exceptions."""
        if isinstance(exc_info, tuple) and len(exc_info) == 3:
            try:
                return logging.Formatter().formatException(exc_info)
            except Exception:
                try:
                    return "".join(traceback.format_exception(*exc_info)).rstrip("\n")
                except Exception:
                    return str(exc_info)
        elif isinstance(exc_info, BaseException):
            try:
                return "".join(traceback.format_exception(exc_info)).rstrip("\n")
            except Exception:
                return str(exc_info)
        elif exc_info is True:
            cur_ei = sys.exc_info()
            if cur_ei and cur_ei[0] is not None:
                return self._safe_format_exception(cur_ei)
            return ""
        return str(exc_info)

    def _clean_record_arg(
        self, val: Any, visited: Optional[set[int]] = None
    ) -> Any:
        """
        Recursively cleans an argument or custom extra attribute:
        1. Preserves primitives (int, float, bool, bytes, None) without converting to str.
        2. Redacts strings via self.redact_text(val).
        3. Recursively handles dicts: redacts string keys, recursively cleans values.
        4. Recursively handles lists and tuples: recursively cleans elements, preserving type.
        5. Handles sets and frozensets cleanly.
        6. Safely stringifies and redacts any other custom objects with exception guard.
        7. Guards against circular reference loops via visited set tracking.
        """
        if val is None or isinstance(val, (bool, int, float, bytes)):
            return val

        if isinstance(val, str):
            return self.redact_text(val)

        if visited is None:
            visited = set()
        val_id = id(val)
        if val_id in visited:
            return "<circular_ref>"
        visited.add(val_id)

        try:
            if isinstance(val, dict):
                return {
                    (self.redact_text(k) if isinstance(k, str) else k): self._clean_record_arg(v, visited)
                    for k, v in val.items()
                }
            if isinstance(val, (list, tuple)):
                cleaned = [self._clean_record_arg(item, visited) for item in val]
                return tuple(cleaned) if isinstance(val, tuple) else cleaned
            if isinstance(val, (set, frozenset)):
                cleaned_set = {self._clean_record_arg(item, visited) for item in val}
                return frozenset(cleaned_set) if isinstance(val, frozenset) else cleaned_set
            try:
                return self.redact_text(str(val))
            except Exception:
                return "<unprintable_object>"
        finally:
            visited.remove(val_id)

    def _sanitize_exception_attributes(
        self, exc_val: Any, visited: Optional[set[int]] = None
    ) -> None:
        """
        Recursively scrubs sensitive data from exception args, notes,
        nested exception groups (BaseExceptionGroup.exceptions),
        and chained exceptions (__cause__, __context__).
        Maintains cycle protection via visited set of object IDs.
        """
        if exc_val is None:
            return

        if visited is None:
            visited = set()

        stack = [exc_val]

        while stack:
            curr = stack.pop()
            if curr is None or id(curr) in visited:
                continue
            visited.add(id(curr))

            # 1. Sanitize exception args
            if hasattr(curr, "args") and curr.args:
                try:
                    # If an argument is a wrapped exception, queue it for traversal
                    for a in curr.args:
                        if isinstance(a, BaseException) and id(a) not in visited:
                            stack.append(a)

                    curr.args = tuple(
                        a if isinstance(a, BaseException)
                        else (self._clean_record_arg(a) if hasattr(self, "_clean_record_arg")
                        else (self.redact_text(a) if isinstance(a, str) else a))
                        for a in curr.args
                    )
                except Exception:
                    pass

            # 2. Sanitize standard error attributes
            for attr in ("strerror", "filename", "filename2", "msg", "message"):
                if hasattr(curr, attr):
                    try:
                        val = getattr(curr, attr, None)
                        if isinstance(val, str):
                            setattr(curr, attr, self.redact_text(val))
                    except Exception:
                        pass

            # 3. Sanitize notes (Python 3.11+)
            if hasattr(curr, "__notes__") and curr.__notes__:
                try:
                    curr.__notes__ = [self.redact_text(str(n)) for n in curr.__notes__]
                except Exception:
                    pass

            # 4. Traverse sub-exceptions in ExceptionGroup / BaseExceptionGroup
            if hasattr(curr, "exceptions") and curr.exceptions:
                try:
                    for child in curr.exceptions:
                        if child is not None and id(child) not in visited:
                            stack.append(child)
                except Exception:
                    pass

            # 5. Traverse chained exceptions (__cause__ and __context__)
            for chain_attr in ("__cause__", "__context__"):
                try:
                    chained = getattr(curr, chain_attr, None)
                    if chained is not None and id(chained) not in visited:
                        stack.append(chained)
                except Exception:
                    pass

    def filter(self, record: logging.LogRecord) -> bool:
        """Applies comprehensive sanitization to the log record."""
        # 1. Sanitize message (handles strings, exceptions, and arbitrary objects)
        if record.msg is not None:
            if isinstance(record.msg, str):
                record.msg = self.redact_text(record.msg)
            else:
                record.msg = self.redact_text(str(record.msg))

        # 2. Sanitize record arguments
        if record.args:
            record.args = self._clean_record_arg(record.args)

        # 3. Sanitize exc_info and exc_text
        if record.exc_info:
            # In-place sanitization of exception representation and attributes
            try:
                exc_val = None
                if isinstance(record.exc_info, tuple) and len(record.exc_info) >= 2:
                    exc_val = record.exc_info[1]
                elif isinstance(record.exc_info, BaseException):
                    exc_val = record.exc_info

                if exc_val is not None:
                    self._sanitize_exception_attributes(exc_val)
            except Exception:
                pass

            if not record.exc_text:
                raw_tb = self._safe_format_exception(record.exc_info)
                record.exc_text = self.redact_text(raw_tb)
            else:
                record.exc_text = self.redact_text(record.exc_text)
        elif record.exc_text:
            record.exc_text = self.redact_text(record.exc_text)

        # 4. Sanitize stack_info if present
        if getattr(record, "stack_info", None):
            record.stack_info = self.redact_text(str(record.stack_info))

        # 5. Sanitize custom record attributes added via extra={...}
        for attr, val in list(record.__dict__.items()):
            if attr not in self.STANDARD_RECORD_ATTRS:
                setattr(record, attr, self._clean_record_arg(val))

        return True


# -----------------------------------------------------------------------------
# Logging Configuration Setup
# -----------------------------------------------------------------------------
def setup_logging(settings: Settings) -> None:
    """
    Initializes root, uvicorn, and application loggers with security redaction.
    """
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    exact_secrets: List[str] = []
    if settings.gemini_api_key:
        secret_val = settings.gemini_api_key.get_secret_value().strip()
        if secret_val and secret_val != "your_gemini_api_key_here":
            exact_secrets.append(secret_val)

    redaction_filter = SensitiveDataFilter(exact_secrets=exact_secrets)

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ"
    )

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Attach filter to existing handlers or configure StreamHandler
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        handler.addFilter(redaction_filter)
        root_logger.addHandler(handler)
    else:
        for handler in root_logger.handlers:
            handler.setFormatter(formatter)
            handler.addFilter(redaction_filter)

    # Ensure uvicorn and app loggers inherit the filter
    for logger_name in ("uvicorn", "uvicorn.access", "uvicorn.error", "fastapi", "app"):
        logger = logging.getLogger(logger_name)
        logger.addFilter(redaction_filter)


# -----------------------------------------------------------------------------
# Singleton Accessor
# -----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Returns the cached singleton instance of application Settings.
    Ensures directories are created and logging is initialized on first call.
    """
    settings = Settings()
    settings.ensure_directories()
    setup_logging(settings)
    return settings
