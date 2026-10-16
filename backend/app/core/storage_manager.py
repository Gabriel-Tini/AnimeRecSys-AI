"""
backend/app/core/storage_manager.py
Thread-safe and process-safe atomic JSON persistence engine for AnimeRecSys AI.
Integrates filelock.FileLock, tempfile.NamedTemporaryFile, os.replace,
Pydantic v2 schema validation, and automatic corruption recovery.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, TypeVar, Union

import filelock
from pydantic import TypeAdapter, ValidationError

try:
    from app.models.anime import AnimeEntity
    from app.models.user import UserProfileData, WatchHistoryEntry
except ImportError:
    try:
        from backend.app.models.anime import AnimeEntity
        from backend.app.models.user import UserProfileData, WatchHistoryEntry
    except ImportError:
        from ..models.anime import AnimeEntity
        from ..models.user import UserProfileData, WatchHistoryEntry

logger = logging.getLogger("animerecsys.storage")

T = TypeVar("T")


# ============================================================================
# Storage Exceptions
# ============================================================================

class StorageError(Exception):
    """Base exception for all storage manager operations."""
    pass


class StorageLockTimeoutError(StorageError):
    """Raised when acquiring a filelock times out."""
    def __init__(self, target_path: Path, lock_path: Path, timeout: float):
        self.target_path = target_path
        self.lock_path = lock_path
        self.timeout = timeout
        super().__init__(
            f"Timeout of {timeout:.1f}s exceeded attempting to acquire lock on '{lock_path}' "
            f"for target file '{target_path}'"
        )


class StorageCorruptedError(StorageError):
    """Raised when persistent data cannot be parsed, validated, or recovered."""
    pass


# ============================================================================
# Storage Manager Implementation
# ============================================================================

class StorageManager:
    """
    Manages persistent JSON state with strict concurrency guarantees:
    - Thread-safe and process-safe FileLock synchronization.
    - Atomic writes via tempfile.NamedTemporaryFile + os.replace on Windows.
    - Pydantic v2 model validation on ingestion and serialization.
    - Automatic corruption detection, forensic backup, and self-healing recovery.
    """

    DEFAULT_TIMEOUT: float = 10.0

    def __init__(
        self,
        workspace_root: Optional[Union[str, Path]] = None,
        data_dir: Optional[Union[str, Path]] = None,
        locks_dir: Optional[Union[str, Path]] = None,
        fixtures_path: Optional[Union[str, Path]] = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        """
        Initializes storage paths. Resolves relative paths to workspace root.
        """
        if workspace_root is None:
            # Default to canonical project root c:\Dev\Ent ML
            self.workspace_root = Path(__file__).resolve().parents[3]
        else:
            self.workspace_root = Path(workspace_root).resolve()

        self.data_dir = Path(data_dir).resolve() if data_dir else self.workspace_root / "data"
        self.locks_dir = Path(locks_dir).resolve() if locks_dir else self.data_dir / ".locks"
        self.fixtures_path = (
            Path(fixtures_path).resolve()
            if fixtures_path
            else self.workspace_root / ".antigravity" / "FIXTURES.json"
        )
        self.timeout = float(timeout)

        # Standard file targets
        self.catalog_path = self.data_dir / "cache_catalog.json"
        self.user_profile_path = self.data_dir / "user_profile.json"
        self.watch_history_path = self.data_dir / "watch_history.json"

        # Pydantic v2 TypeAdapters for collection validation
        self._catalog_adapter = TypeAdapter(List[AnimeEntity])
        self._history_adapter = TypeAdapter(List[WatchHistoryEntry])

        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensures data and lock directories exist on disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.locks_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------------
    # Lock Mechanics
    # ------------------------------------------------------------------------

    def get_lock_path(self, target_path: Path) -> Path:
        """
        Computes the deterministic lockfile path in data/.locks/.
        Example: data/user_profile.json -> data/.locks/user_profile.json.lock
        """
        target_path = Path(target_path).resolve()
        try:
            rel = target_path.relative_to(self.data_dir.resolve())
            lock_file = self.locks_dir / f"{rel}.lock"
        except ValueError:
            # File is outside data_dir (e.g. testing temporary paths)
            lock_file = self.locks_dir / f"{target_path.name}.lock"
        return lock_file

    @contextmanager
    def file_lock(self, target_path: Path, timeout: Optional[float] = None) -> Iterator[filelock.FileLock]:
        """
        Context manager for acquiring a FileLock on a target data file.
        Always creates parent directory for lock file if absent.
        """
        lock_path = self.get_lock_path(target_path)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        t = self.timeout if timeout is None else float(timeout)
        lock = filelock.FileLock(str(lock_path), timeout=t)
        try:
            with lock:
                yield lock
        except filelock.Timeout as exc:
            logger.error(f"Failed to acquire file lock within {t:.1f}s: {lock_path}")
            raise StorageLockTimeoutError(target_path=target_path, lock_path=lock_path, timeout=t) from exc

    # ------------------------------------------------------------------------
    # Low-Level Atomic Operations
    # ------------------------------------------------------------------------

    def _unlocked_atomic_write(self, target_path: Path, content: str) -> None:
        """
        Writes content to target_path using tempfile and os.replace.
        Caller MUST hold the corresponding file_lock.
        
        Windows Safety Guarantees:
        1. delete=False allows manual closure and subsequent replacement.
        2. dir=target_path.parent ensures same filesystem volume.
        3. os.fsync flushes dirty pages to non-volatile disk.
        4. tf handle is fully closed upon block exit BEFORE os.replace.
        5. Retry loop handles transient NTFS file-system locking delays.
        """
        target_path = Path(target_path).resolve()
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path: Optional[Path] = None

        try:
            prefix = f".{target_path.stem}_tmp_"
            with tempfile.NamedTemporaryFile(
                mode="w",
                dir=target_path.parent,
                delete=False,
                encoding="utf-8",
                prefix=prefix,
                suffix=".tmp",
            ) as tf:
                temp_path = Path(tf.name)
                tf.write(content)
                tf.flush()
                os.fsync(tf.fileno())

            # File handle is closed. Perform atomic swap with transient retry for Windows NTFS.
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    os.replace(temp_path, target_path)
                    temp_path = None
                    break
                except PermissionError:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(0.02)
        except Exception as exc:
            if temp_path is not None and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            logger.error(f"Atomic replacement failed on '{target_path}': {exc}")
            raise StorageError(f"Atomic replacement failed on '{target_path}': {exc}") from exc

    def _unlocked_read_raw(self, target_path: Path) -> Any:
        """
        Reads raw JSON from target_path.
        Caller MUST hold the corresponding file_lock.
        """
        target_path = Path(target_path).resolve()
        if not target_path.exists():
            raise FileNotFoundError(f"Target file does not exist: {target_path}")
        if target_path.stat().st_size == 0:
            raise ValueError(f"Target file is 0 bytes: {target_path}")
        text = target_path.read_text(encoding="utf-8")
        return json.loads(text)

    def _backup_corrupted_file(self, target_path: Path, reason: str) -> Optional[Path]:
        """
        Creates a timestamped backup of a corrupted file before self-healing.
        """
        target_path = Path(target_path).resolve()
        if not target_path.exists():
            return None
        try:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            backup_path = target_path.parent / f"corrupted_{target_path.stem}_{ts}{target_path.suffix}.bak"
            shutil.copy2(target_path, backup_path)
            logger.warning(
                f"Preserved corrupted file '{target_path}' to '{backup_path}'. Reason: {reason}"
            )
            return backup_path
        except Exception as exc:
            logger.error(f"Failed to create forensic backup for '{target_path}': {exc}")
            return None

    # ------------------------------------------------------------------------
    # Public Generic Atomic JSON Helpers
    # ------------------------------------------------------------------------

    def atomic_read_json(self, target_path: Path) -> Any:
        """Reads JSON data with FileLock synchronization."""
        with self.file_lock(target_path):
            return self._unlocked_read_raw(target_path)

    def atomic_write_json(self, target_path: Path, data: Any) -> None:
        """Serializes and writes JSON data with FileLock synchronization."""
        content = json.dumps(data, indent=2, ensure_ascii=False)
        with self.file_lock(target_path):
            self._unlocked_atomic_write(target_path, content)

    # ------------------------------------------------------------------------
    # Catalog Storage Operations (cache_catalog.json)
    # ------------------------------------------------------------------------

    def _unlocked_load_catalog(self) -> List[AnimeEntity]:
        """Loads and validates cache_catalog.json with fallback to FIXTURES.json."""
        try:
            raw_data = self._unlocked_read_raw(self.catalog_path)
            return self._catalog_adapter.validate_python(raw_data)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Catalog invalid or absent ({reason}). Initiating recovery.")
            if self.catalog_path.exists():
                self._backup_corrupted_file(self.catalog_path, reason)
            return self._recover_catalog_from_fixtures()

    def _recover_catalog_from_fixtures(self) -> List[AnimeEntity]:
        """Recovers catalog from .antigravity/FIXTURES.json sample_animes_full."""
        if not self.fixtures_path.exists():
            logger.error(f"Fixtures file missing at '{self.fixtures_path}'. Catalog recovery empty.")
            empty_catalog: List[AnimeEntity] = []
            content = json.dumps(
                self._catalog_adapter.dump_python(empty_catalog, mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            self._unlocked_atomic_write(self.catalog_path, content)
            return empty_catalog

        try:
            fixtures_text = self.fixtures_path.read_text(encoding="utf-8")
            fixtures_json = json.loads(fixtures_text)
            sample_animes = fixtures_json.get("sample_animes_full", [])
            catalog = self._catalog_adapter.validate_python(sample_animes)
            # Self-heal catalog file on disk
            content = json.dumps(
                self._catalog_adapter.dump_python(catalog, mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            self._unlocked_atomic_write(self.catalog_path, content)
            logger.info(f"Self-healed catalog with {len(catalog)} entities from fixtures.")
            return catalog
        except Exception as exc:
            logger.error(f"Fatal error recovering catalog from fixtures: {exc}")
            raise StorageCorruptedError(f"Cannot recover catalog: {exc}") from exc

    def load_catalog(self) -> List[AnimeEntity]:
        """Loads catalog with FileLock synchronization and corruption self-healing."""
        with self.file_lock(self.catalog_path):
            return self._unlocked_load_catalog()

    def save_catalog(self, catalog: List[AnimeEntity]) -> None:
        """Atomically saves catalog with FileLock synchronization."""
        content = json.dumps(
            self._catalog_adapter.dump_python(catalog, mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        with self.file_lock(self.catalog_path):
            self._unlocked_atomic_write(self.catalog_path, content)

    def upsert_catalog_items(self, new_items: List[AnimeEntity]) -> int:
        """
        Thread/process-safe upsert of new anime items into catalog cache.
        Returns the number of new unique items added.
        """
        with self.file_lock(self.catalog_path):
            current = self._unlocked_load_catalog()
            existing_ids = {a.id for a in current}
            added = 0
            for item in new_items:
                if item.id not in existing_ids:
                    current.append(item)
                    existing_ids.add(item.id)
                    added += 1
            if added > 0:
                content = json.dumps(
                    self._catalog_adapter.dump_python(current, mode="json"),
                    indent=2,
                    ensure_ascii=False,
                )
                self._unlocked_atomic_write(self.catalog_path, content)
            return added

    # ------------------------------------------------------------------------
    # User Profile Storage Operations (user_profile.json)
    # ------------------------------------------------------------------------

    def _unlocked_load_user_profile(self) -> UserProfileData:
        """Loads and validates user_profile.json with fallback to default profile."""
        try:
            raw_data = self._unlocked_read_raw(self.user_profile_path)
            return UserProfileData.model_validate(raw_data)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning(f"User profile invalid or absent ({reason}). Initiating recovery.")
            if self.user_profile_path.exists():
                self._backup_corrupted_file(self.user_profile_path, reason)
            return self._recover_default_user_profile()

    def _recover_default_user_profile(self) -> UserProfileData:
        """Creates default user profile and self-heals user_profile.json."""
        default_profile = UserProfileData(
            user_id="default_user",
            updated_at=datetime.now(timezone.utc),
            total_ratings=0,
            feature_weights={},
        )
        content = json.dumps(
            default_profile.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        self._unlocked_atomic_write(self.user_profile_path, content)
        logger.info("Self-healed user profile with default state.")
        return default_profile

    def load_user_profile(self) -> UserProfileData:
        """Loads user profile with FileLock synchronization."""
        with self.file_lock(self.user_profile_path):
            return self._unlocked_load_user_profile()

    def save_user_profile(self, profile: UserProfileData) -> None:
        """Atomically saves user profile with FileLock synchronization."""
        content = json.dumps(
            profile.model_dump(mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        with self.file_lock(self.user_profile_path):
            self._unlocked_atomic_write(self.user_profile_path, content)

    def update_user_profile(
        self,
        modifier_fn: Callable[[UserProfileData], UserProfileData]
    ) -> UserProfileData:
        """
        Thread/process-safe read-modify-write on user profile.
        Guarantees ZERO lost updates under high concurrency.
        """
        with self.file_lock(self.user_profile_path):
            current = self._unlocked_load_user_profile()
            updated = modifier_fn(current)
            content = json.dumps(
                updated.model_dump(mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            self._unlocked_atomic_write(self.user_profile_path, content)
            return updated

    def reset_user_profile(self) -> UserProfileData:
        """Resets user profile to initial default state atomically."""
        with self.file_lock(self.user_profile_path):
            return self._recover_default_user_profile()

    # ------------------------------------------------------------------------
    # Watch History Storage Operations (watch_history.json)
    # ------------------------------------------------------------------------

    def _unlocked_load_watch_history(self) -> List[WatchHistoryEntry]:
        """Loads and validates watch_history.json with fallback to empty list."""
        try:
            raw_data = self._unlocked_read_raw(self.watch_history_path)
            return self._history_adapter.validate_python(raw_data)
        except (FileNotFoundError, ValueError, json.JSONDecodeError, ValidationError) as exc:
            reason = f"{type(exc).__name__}: {exc}"
            logger.warning(f"Watch history invalid or absent ({reason}). Initiating recovery.")
            if self.watch_history_path.exists():
                self._backup_corrupted_file(self.watch_history_path, reason)
            return self._recover_empty_watch_history()

    def _recover_empty_watch_history(self) -> List[WatchHistoryEntry]:
        """Creates empty watch history and self-heals watch_history.json."""
        empty_history: List[WatchHistoryEntry] = []
        content = json.dumps(
            self._history_adapter.dump_python(empty_history, mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        self._unlocked_atomic_write(self.watch_history_path, content)
        logger.info("Self-healed watch history with empty list.")
        return empty_history

    def load_watch_history(self) -> List[WatchHistoryEntry]:
        """Loads watch history with FileLock synchronization."""
        with self.file_lock(self.watch_history_path):
            return self._unlocked_load_watch_history()

    def save_watch_history(self, history: List[WatchHistoryEntry]) -> None:
        """Atomically saves watch history with FileLock synchronization."""
        content = json.dumps(
            self._history_adapter.dump_python(history, mode="json"),
            indent=2,
            ensure_ascii=False,
        )
        with self.file_lock(self.watch_history_path):
            self._unlocked_atomic_write(self.watch_history_path, content)

    def append_watch_history(self, entry: WatchHistoryEntry) -> None:
        """
        Thread/process-safe atomic upsert to watch history.
        Acquires FileLock once for the entire read-modify-write cycle.
        If an entry for entry.anime_id already exists, updates it in place to prevent duplicates.
        Otherwise appends the new entry.
        Guarantees ZERO lost updates under 20+ concurrent operations and ZERO duplicates.
        """
        with self.file_lock(self.watch_history_path):
            current = self._unlocked_load_watch_history()
            found_idx = -1
            for idx, existing in enumerate(current):
                if existing.anime_id == entry.anime_id:
                    found_idx = idx
                    break
            if found_idx >= 0:
                current[found_idx] = entry
            else:
                current.append(entry)

            content = json.dumps(
                self._history_adapter.dump_python(current, mode="json"),
                indent=2,
                ensure_ascii=False,
            )
            self._unlocked_atomic_write(self.watch_history_path, content)

    def enrich_latest_watch_history(
        self,
        anime_id: Optional[int] = None,
        user_review_raw: Optional[str] = None,
        qualitative_tuning: Optional[Any] = None,
        user_review: Optional[str] = None,
        tuning: Optional[Any] = None,
    ) -> Optional[WatchHistoryEntry]:
        """
        Thread-safe and process-safe atomic enrichment of the latest watch history entry
        (optionally filtered by anime_id) with user review commentary and qualitative tuning data.
        Uses FileLock and atomic NamedTemporaryFile replacement.
        """
        review = user_review_raw if user_review_raw is not None else user_review
        tuning_val = qualitative_tuning if qualitative_tuning is not None else tuning

        with self.file_lock(self.watch_history_path):
            history = self._unlocked_load_watch_history()
            for i in range(len(history) - 1, -1, -1):
                if anime_id is None or history[i].anime_id == anime_id:
                    old_entry = history[i]
                    updated_entry = old_entry.model_copy(
                        update={
                            "user_review_raw": review[:300] if review else old_entry.user_review_raw,
                            "qualitative_tuning": tuning_val or old_entry.qualitative_tuning,
                        }
                    )
                    history[i] = updated_entry
                    content = json.dumps(
                        self._history_adapter.dump_python(history, mode="json"),
                        indent=2,
                        ensure_ascii=False,
                    )
                    self._unlocked_atomic_write(self.watch_history_path, content)
                    return updated_entry
            return None

    def update_watch_history_entry(
        self,
        entry_id: str,
        modifier_fn: Callable[[WatchHistoryEntry], WatchHistoryEntry],
    ) -> Optional[WatchHistoryEntry]:
        """
        Atomically updates a specific watch history entry identified by entry_id.
        Uses FileLock and atomic NamedTemporaryFile replacement.
        """
        with self.file_lock(self.watch_history_path):
            history = self._unlocked_load_watch_history()
            for i, entry in enumerate(history):
                if entry.id == entry_id:
                    updated = modifier_fn(entry)
                    history[i] = updated
                    content = json.dumps(
                        self._history_adapter.dump_python(history, mode="json"),
                        indent=2,
                        ensure_ascii=False,
                    )
                    self._unlocked_atomic_write(self.watch_history_path, content)
                    return updated
            return None

    def reset_watch_history(self) -> List[WatchHistoryEntry]:
        """Resets watch history to empty list atomically."""
        with self.file_lock(self.watch_history_path):
            return self._recover_empty_watch_history()

    # ------------------------------------------------------------------------
    # Global Reset
    # ------------------------------------------------------------------------

    def reset_all_user_data(self) -> tuple[UserProfileData, List[WatchHistoryEntry]]:
        """
        Resets both user profile and watch history atomically.
        Preserves connected Google account if the user was authenticated.
        Locks both files in deterministic order to prevent deadlocks.
        """
        with self.file_lock(self.user_profile_path):
            with self.file_lock(self.watch_history_path):
                existing_profile = self._unlocked_load_user_profile()
                saved_google = existing_profile.google_account

                profile = self._recover_default_user_profile()
                if saved_google:
                    profile.google_account = saved_google
                    content = json.dumps(
                        profile.model_dump(mode="json"),
                        indent=2,
                        ensure_ascii=False,
                    )
                    self._unlocked_atomic_write(self.user_profile_path, content)

                history = self._recover_empty_watch_history()
                return profile, history

