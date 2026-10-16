"""
tests/unit/test_storage_manager.py
Automated test suite verifying FileLock, atomic writes, Windows safety semantics,
corruption recovery, and 20-worker thread/process concurrency.
"""

import json
import logging
import multiprocessing
import os
import shutil
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import pytest

from app.core.storage_manager import (
    StorageCorruptedError,
    StorageError,
    StorageLockTimeoutError,
    StorageManager,
)
from app.models.anime import AnimeEntity, AnimeTag
from app.models.user import UserProfileData, WatchHistoryEntry


# ----------------------------------------------------------------------------
# Top-level functions for multiprocessing (required on Windows 'spawn')
# ----------------------------------------------------------------------------
def _mp_append_worker(args: Tuple[str, str, int]) -> int:
    """Multiprocessing worker function to append watch history entry."""
    data_dir, locks_dir, worker_id = args
    sm = StorageManager(data_dir=data_dir, locks_dir=locks_dir, timeout=15.0)
    entry = WatchHistoryEntry(
        id=f"mp_entry_{worker_id}",
        anime_id=1000 + worker_id,
        anime_title=f"MP Anime {worker_id}",
        rating=7.5,
        origin_track="discovery",
    )
    sm.append_watch_history(entry)
    return worker_id


def _mp_profile_worker(args: Tuple[str, str, int]) -> int:
    """Multiprocessing worker function to update user profile."""
    data_dir, locks_dir, worker_id = args
    sm = StorageManager(data_dir=data_dir, locks_dir=locks_dir, timeout=15.0)

    def _modifier(profile: UserProfileData) -> UserProfileData:
        profile.total_ratings += 1
        profile.feature_weights[f"mp_tag_{worker_id}"] = float(worker_id)
        return profile

    sm.update_user_profile(_modifier)
    return worker_id


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------
@pytest.fixture
def storage_env(tmp_path: Path):
    """Provides an isolated test environment with fresh directories and fixtures."""
    data_dir = tmp_path / "data"
    locks_dir = data_dir / ".locks"
    fixtures_file = tmp_path / "FIXTURES.json"

    # Create minimal valid fixture
    fixture_payload = {
        "sample_animes_full": [
            {
                "id": 9253,
                "title_romaji": "Steins;Gate",
                "title_english": "Steins;Gate",
                "title_native": "シュタインズ・ゲート",
                "cover_image": "https://example.com/steins.png",
                "format": "TV",
                "status": "FINISHED",
                "popularity": 320000,
                "genres": ["Sci-Fi", "Drama"],
                "tags": [
                    {"id": 6, "name": "Time Manipulation", "rank": 98, "is_media_spoiler": False},
                    {"id": 11, "name": "Plot Twist", "rank": 96, "is_media_spoiler": True},
                ],
            }
        ]
    }
    fixtures_file.write_text(json.dumps(fixture_payload), encoding="utf-8")

    sm = StorageManager(
        workspace_root=tmp_path,
        data_dir=data_dir,
        locks_dir=locks_dir,
        fixtures_path=fixtures_file,
        timeout=5.0,
    )
    return sm, tmp_path


# ----------------------------------------------------------------------------
# Suite 1: Basic Operations & Initial Recovery
# ----------------------------------------------------------------------------
def test_initial_catalog_fallback_to_fixtures(storage_env):
    sm, _ = storage_env
    # cache_catalog.json does not exist yet
    catalog = sm.load_catalog()
    assert len(catalog) == 1
    assert catalog[0].id == 9253
    assert catalog[0].title_romaji == "Steins;Gate"
    # Ensure file was self-healed to disk
    assert sm.catalog_path.exists()


def test_initial_user_profile_default(storage_env):
    sm, _ = storage_env
    profile = sm.load_user_profile()
    assert profile.user_id == "default_user"
    assert profile.total_ratings == 0
    assert profile.feature_weights == {}
    assert sm.user_profile_path.exists()


def test_initial_watch_history_default(storage_env):
    sm, _ = storage_env
    history = sm.load_watch_history()
    assert history == []
    assert sm.watch_history_path.exists()


def test_catalog_save_and_upsert(storage_env):
    sm, _ = storage_env
    initial = sm.load_catalog()
    assert len(initial) == 1

    new_item = AnimeEntity(
        id=16498,
        title_romaji="Attack on Titan",
        cover_image="https://example.com/aot.png",
        genres=["Action"],
        tags=[AnimeTag(id=1, name="Action", rank=95, is_media_spoiler=False)],
    )

    # Upsert new item
    added = sm.upsert_catalog_items([new_item])
    assert added == 1

    # Upsert same item again (deduplication check)
    added_again = sm.upsert_catalog_items([new_item])
    assert added_again == 0

    catalog = sm.load_catalog()
    assert len(catalog) == 2
    assert {c.id for c in catalog} == {9253, 16498}


def test_reset_operations(storage_env):
    sm, _ = storage_env
    # Mutate profile and history
    sm.update_user_profile(lambda p: setattr(p, "total_ratings", 5) or p)
    sm.append_watch_history(
        WatchHistoryEntry(
            anime_id=1, anime_title="A", rating=8.0, origin_track="affinity"
        )
    )

    assert sm.load_user_profile().total_ratings == 5
    assert len(sm.load_watch_history()) == 1

    # Reset all atomically
    profile, history = sm.reset_all_user_data()
    assert profile.total_ratings == 0
    assert history == []
    assert sm.load_user_profile().total_ratings == 0
    assert sm.load_watch_history() == []


# ----------------------------------------------------------------------------
# Suite 2: Corruption Detection & Forensic Recovery
# ----------------------------------------------------------------------------
def test_corruption_recovery_and_backup_syntax_error(storage_env):
    sm, _ = storage_env
    # Corrupt user_profile.json with broken syntax
    sm.user_profile_path.write_text("{'user_id': broken json...", encoding="utf-8")

    # Load should self-heal and not crash
    profile = sm.load_user_profile()
    assert profile.user_id == "default_user"
    assert profile.total_ratings == 0

    # Verify backup was created
    backups = list(sm.data_dir.glob("corrupted_user_profile_*.bak"))
    assert len(backups) >= 1
    assert "broken json" in backups[0].read_text(encoding="utf-8")


def test_corruption_recovery_zero_byte_file(storage_env):
    sm, _ = storage_env
    # Write 0-byte file
    sm.catalog_path.write_text("", encoding="utf-8")

    # Load should self-heal from fixtures
    catalog = sm.load_catalog()
    assert len(catalog) == 1
    assert catalog[0].id == 9253

    # Verify backup was created
    backups = list(sm.data_dir.glob("corrupted_cache_catalog_*.bak"))
    assert len(backups) >= 1


def test_corruption_recovery_invalid_schema(storage_env):
    sm, _ = storage_env
    # Write JSON with invalid schema (total_ratings is not an int)
    sm.user_profile_path.write_text(
        json.dumps({"user_id": "default_user", "total_ratings": "not-an-int"}),
        encoding="utf-8",
    )

    profile = sm.load_user_profile()
    assert profile.user_id == "default_user"
    assert profile.total_ratings == 0


# ----------------------------------------------------------------------------
# Suite 3: Lock Timeout & Domain Exception
# ----------------------------------------------------------------------------
def test_lock_timeout_error(storage_env):
    sm, _ = storage_env

    # Acquire lock in thread and hold it
    def _hold_lock():
        with sm.file_lock(sm.user_profile_path):
            time.sleep(1.0)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_hold_lock)
    time.sleep(0.1)  # Allow thread to acquire lock

    # Main thread tries to acquire with short timeout
    with pytest.raises(StorageLockTimeoutError) as exc_info:
        with sm.file_lock(sm.user_profile_path, timeout=0.2):
            pass

    assert "Timeout of 0.2s exceeded" in str(exc_info.value)
    future.result()
    executor.shutdown(wait=True)


# ----------------------------------------------------------------------------
# Suite 4: 20-Worker Thread Concurrency
# (Acceptance Criteria: 20 concurrent write operations produce 0 lost updates
# and 0 JSONDecodeError)
# ----------------------------------------------------------------------------
def test_20_worker_thread_concurrency_watch_history(storage_env):
    """
    R4 Acceptance Criteria Verification:
    20 concurrent threads appending to watch_history.json produce zero JSONDecodeError
    and zero lost updates.
    """
    sm, _ = storage_env
    num_workers = 20

    def _worker(worker_id: int):
        entry = WatchHistoryEntry(
            id=f"entry_{worker_id}",
            anime_id=100 + worker_id,
            anime_title=f"Anime {worker_id}",
            rating=8.0,
            origin_track="affinity",
        )
        sm.append_watch_history(entry)
        return worker_id

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_workers)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == num_workers

    # Verify no lost updates
    history = sm.load_watch_history()
    assert len(history) == num_workers
    entry_ids = {h.id for h in history}
    expected_ids = {f"entry_{i}" for i in range(num_workers)}
    assert entry_ids == expected_ids

    # Verify JSON on disk parses cleanly
    raw_text = sm.watch_history_path.read_text(encoding="utf-8")
    parsed = json.loads(raw_text)
    assert len(parsed) == num_workers


def test_20_worker_thread_concurrency_user_profile(storage_env):
    """
    20 concurrent threads updating user_profile.json produce zero lost updates.
    """
    sm, _ = storage_env
    num_workers = 20

    def _worker(worker_id: int):
        def _modifier(p: UserProfileData) -> UserProfileData:
            p.total_ratings += 1
            p.feature_weights[f"tag_{worker_id}"] = float(worker_id)
            return p

        sm.update_user_profile(_modifier)
        return worker_id

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_worker, i) for i in range(num_workers)]
        results = [f.result() for f in as_completed(futures)]

    assert len(results) == num_workers

    profile = sm.load_user_profile()
    assert profile.total_ratings == num_workers
    assert len(profile.feature_weights) == num_workers
    for i in range(num_workers):
        assert profile.feature_weights[f"tag_{i}"] == float(i)


# ----------------------------------------------------------------------------
# Suite 5: Multi-Process Concurrency (20 Parallel Processes / Tasks)
# ----------------------------------------------------------------------------
def test_multiprocess_concurrency_append(storage_env):
    """
    Process-safe concurrency across separate OS processes on Windows.
    """
    sm, tmp_path = storage_env
    data_dir = str(sm.data_dir)
    locks_dir = str(sm.locks_dir)
    num_tasks = 20

    tasks = [(data_dir, locks_dir, i) for i in range(num_tasks)]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_mp_append_worker, tasks)

    assert len(results) == num_tasks

    # Verify final state
    history = sm.load_watch_history()
    assert len(history) == num_tasks
    mp_ids = {h.id for h in history}
    expected = {f"mp_entry_{i}" for i in range(num_tasks)}
    assert mp_ids == expected


def test_multiprocess_concurrency_profile_update(storage_env):
    """
    Process-safe read-modify-write on user profile across multiple processes.
    """
    sm, tmp_path = storage_env
    data_dir = str(sm.data_dir)
    locks_dir = str(sm.locks_dir)
    num_tasks = 20

    tasks = [(data_dir, locks_dir, i) for i in range(num_tasks)]

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_mp_profile_worker, tasks)

    assert len(results) == num_tasks

    profile = sm.load_user_profile()
    assert profile.total_ratings == num_tasks
    assert len(profile.feature_weights) == num_tasks


# ----------------------------------------------------------------------------
# Suite 6: Interleaved Readers and Writers Stress Test
# ----------------------------------------------------------------------------
def test_interleaved_readers_and_writers(storage_env):
    """
    High contention: 10 reader threads and 10 writer threads executing concurrently.
    Guarantees no Windows PermissionError or partial JSON reads.
    """
    sm, _ = storage_env
    num_readers = 10
    num_writers = 10
    read_iterations = 30
    write_iterations = 10

    reader_errors: List[Exception] = []
    writer_errors: List[Exception] = []

    def _reader():
        try:
            for _ in range(read_iterations):
                profile = sm.load_user_profile()
                assert profile.user_id == "default_user"
                history = sm.load_watch_history()
                assert isinstance(history, list)
                time.sleep(0.005)
        except Exception as exc:
            reader_errors.append(exc)

    def _writer(w_id: int):
        try:
            for i in range(write_iterations):
                entry = WatchHistoryEntry(
                    id=f"stress_{w_id}_{i}",
                    anime_id=1000 + w_id * 100 + i,
                    anime_title=f"Stress Anime {w_id}",
                    rating=8.0,
                    origin_track="affinity",
                )
                sm.append_watch_history(entry)
                sm.update_user_profile(
                    lambda p: setattr(p, "total_ratings", p.total_ratings + 1) or p
                )
                time.sleep(0.005)
        except Exception as exc:
            writer_errors.append(exc)

    with ThreadPoolExecutor(max_workers=num_readers + num_writers) as executor:
        read_futures = [executor.submit(_reader) for _ in range(num_readers)]
        write_futures = [executor.submit(_writer, w) for w in range(num_writers)]

        for f in read_futures + write_futures:
            f.result()

    assert not reader_errors, f"Reader encountered errors: {reader_errors}"
    assert not writer_errors, f"Writer encountered errors: {writer_errors}"

    final_history = sm.load_watch_history()
    expected_total_writes = num_writers * write_iterations
    assert len(final_history) == expected_total_writes
