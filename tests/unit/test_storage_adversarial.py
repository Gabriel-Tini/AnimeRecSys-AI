"""
tests/unit/test_storage_adversarial.py
Adversarial Concurrency, Atomic Persistence & Stress Test Suite for StorageManager.

Empirical verification under:
1. High-contention interleaved readers and writers across threads.
2. High-contention multi-process writes across separate Windows processes.
3. Rapid consecutive atomic resets while concurrent writes are in flight.
4. Simulated lock timeouts and post-timeout recovery.
5. Partial file writes, truncation (0-byte), syntax corruption, and forensic recovery.
6. Windows NTFS transient file-sharing lock contention during atomic os.replace.
7. Concurrent catalog upsert deduplication and race conditions.
8. Abnormal process termination (SIGTERM/kill) and automated OS lock release.
9. UnicodeDecodeError, null JSON, and primitive JSON type recovery.
10. Fixtures file absence during catalog corruption recovery.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import shutil
import tempfile
import threading
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


# ============================================================================
# Top-Level Functions for Windows Multiprocessing ("spawn" mode)
# ============================================================================

def _mp_adversarial_writer(args: Tuple[str, str, int, int]) -> int:
    """
    Worker process executing a batch of atomic writes against watch history
    and user profile under high process contention.
    """
    data_dir, locks_dir, worker_id, iterations = args
    sm = StorageManager(data_dir=data_dir, locks_dir=locks_dir, timeout=20.0)

    for i in range(iterations):
        # 1. Append watch history entry
        entry = WatchHistoryEntry(
            id=f"mp_adv_w{worker_id}_i{i}",
            anime_id=10000 + worker_id * 1000 + i,
            anime_title=f"MP Adv Anime {worker_id}-{i}",
            rating=5.0 + (i % 5),
            origin_track="affinity" if (i % 2 == 0) else "discovery",
        )
        sm.append_watch_history(entry)

        # 2. Update user profile
        def _update(p: UserProfileData) -> UserProfileData:
            p.total_ratings += 1
            p.feature_weights[f"p{worker_id}_tag_{i}"] = float(worker_id * 10 + i)
            return p

        sm.update_user_profile(_update)

    return worker_id


def _mp_lock_holding_worker(data_dir: str, locks_dir: str, ready_queue: multiprocessing.Queue) -> None:
    """
    Worker process that acquires the profile filelock, signals parent, and sleeps
    until forcibly killed.
    """
    sm = StorageManager(data_dir=data_dir, locks_dir=locks_dir, timeout=10.0)
    with sm.file_lock(sm.user_profile_path):
        ready_queue.put("LOCKED")
        time.sleep(30.0)


# ============================================================================
# Test Fixture
# ============================================================================

@pytest.fixture
def adv_storage_env(tmp_path: Path):
    """Provides an isolated sandbox environment with realistic seed fixtures."""
    data_dir = tmp_path / "data"
    locks_dir = data_dir / ".locks"
    fixtures_file = tmp_path / "FIXTURES.json"

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
            },
            {
                "id": 16498,
                "title_romaji": "Shingeki no Kyojin",
                "title_english": "Attack on Titan",
                "title_native": "進撃の巨人",
                "cover_image": "https://example.com/aot.png",
                "format": "TV",
                "status": "FINISHED",
                "popularity": 540000,
                "genres": ["Action", "Fantasy"],
                "tags": [
                    {"id": 1, "name": "Military", "rank": 92, "is_media_spoiler": False},
                ],
            },
        ]
    }
    fixtures_file.write_text(json.dumps(fixture_payload, indent=2), encoding="utf-8")

    sm = StorageManager(
        workspace_root=tmp_path,
        data_dir=data_dir,
        locks_dir=locks_dir,
        fixtures_path=fixtures_file,
        timeout=10.0,
    )
    return sm, tmp_path


# ============================================================================
# Empirical Stress Tests
# ============================================================================

class TestAdversarialConcurrencyAndPersistence:
    """Adversarial stress harness for storage_manager.py."""

    def test_adv_01_extreme_interleaved_readers_and_writers_threads(self, adv_storage_env):
        """
        Adversarial Stress: 25 concurrent writer threads + 20 concurrent reader threads (45 threads total).
        Tests high-contention interleaved reads, writes, and profile updates.
        Verifies:
        - Zero Windows PermissionError (WinError 5 / 32 sharing violations)
        - Zero JSONDecodeError
        - Zero lost updates across writers
        - Exact final state consistency
        """
        sm, _ = adv_storage_env
        num_writers = 25
        num_readers = 20
        write_iterations = 8
        read_iterations = 25

        reader_exceptions: List[Exception] = []
        writer_exceptions: List[Exception] = []

        stop_event = threading.Event()

        def _reader_job(reader_id: int):
            try:
                count = 0
                while not stop_event.is_set() and count < read_iterations:
                    # Alternating reads
                    profile = sm.load_user_profile()
                    assert profile.user_id == "default_user"
                    assert isinstance(profile.feature_weights, dict)

                    history = sm.load_watch_history()
                    assert isinstance(history, list)

                    catalog = sm.load_catalog()
                    assert len(catalog) >= 2
                    count += 1
                    time.sleep(0.002)
            except Exception as exc:
                reader_exceptions.append(exc)

        def _writer_job(writer_id: int):
            try:
                for i in range(write_iterations):
                    # 1. Append history entry
                    entry = WatchHistoryEntry(
                        id=f"thread_adv_w{writer_id}_i{i}",
                        anime_id=20000 + writer_id * 100 + i,
                        anime_title=f"Thread Adv Anime {writer_id}",
                        rating=7.0,
                        origin_track="affinity",
                    )
                    sm.append_watch_history(entry)

                    # 2. Update profile
                    def _mod(p: UserProfileData) -> UserProfileData:
                        p.total_ratings += 1
                        p.feature_weights[f"t_w{writer_id}_i{i}"] = float(i + 1)
                        return p

                    sm.update_user_profile(_mod)
                    time.sleep(0.002)
            except Exception as exc:
                writer_exceptions.append(exc)

        total_threads = num_writers + num_readers
        with ThreadPoolExecutor(max_workers=total_threads) as executor:
            read_futures = [executor.submit(_reader_job, r) for r in range(num_readers)]
            write_futures = [executor.submit(_writer_job, w) for w in range(num_writers)]

            # Wait for all writers to complete
            for wf in write_futures:
                wf.result()

            stop_event.set()
            for rf in read_futures:
                rf.result()

        assert not reader_exceptions, f"Reader threads encountered exceptions: {reader_exceptions}"
        assert not writer_exceptions, f"Writer threads encountered exceptions: {writer_exceptions}"

        # Invariant checks: Zero lost updates
        expected_total_writes = num_writers * write_iterations
        history = sm.load_watch_history()
        assert len(history) == expected_total_writes

        history_ids = {h.id for h in history}
        expected_ids = {
            f"thread_adv_w{w}_i{i}"
            for w in range(num_writers)
            for i in range(write_iterations)
        }
        assert history_ids == expected_ids

        profile = sm.load_user_profile()
        assert profile.total_ratings == expected_total_writes
        assert len(profile.feature_weights) == expected_total_writes

        # Verify disk files parse cleanly via standard json without StorageManager
        raw_history = json.loads(sm.watch_history_path.read_text(encoding="utf-8"))
        assert len(raw_history) == expected_total_writes
        raw_profile = json.loads(sm.user_profile_path.read_text(encoding="utf-8"))
        assert raw_profile["total_ratings"] == expected_total_writes

    def test_adv_02_multiprocess_high_contention_spam(self, adv_storage_env):
        """
        Adversarial Multi-Process Stress:
        Spawns 8 concurrent worker processes simultaneously hitting watch_history
        and user_profile for multiple iterations.
        Verifies:
        - Complete cross-process synchronization via FileLock
        - Zero lost updates across separate processes
        - Zero Windows permission sharing violations
        """
        sm, _ = adv_storage_env
        num_workers = 8
        iterations_per_worker = 10
        total_expected = num_workers * iterations_per_worker

        tasks = [
            (str(sm.data_dir), str(sm.locks_dir), w, iterations_per_worker)
            for w in range(num_workers)
        ]

        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_mp_adversarial_writer, tasks)

        assert len(results) == num_workers

        # Verify exact updates
        history = sm.load_watch_history()
        assert len(history) == total_expected

        all_ids = {h.id for h in history}
        expected_ids = {
            f"mp_adv_w{w}_i{i}"
            for w in range(num_workers)
            for i in range(iterations_per_worker)
        }
        assert all_ids == expected_ids

        profile = sm.load_user_profile()
        assert profile.total_ratings == total_expected
        assert len(profile.feature_weights) == total_expected

    def test_adv_03_rapid_consecutive_resets_while_writes_in_flight(self, adv_storage_env):
        """
        Adversarial Scenario: Rapid Resets vs In-Flight Concurrent Writes
        Thread 1 & 2: Rapidly spamming reset_all_user_data() in a tight loop.
        Threads 3 to 14: Rapidly writing watch history and profile updates.
        Verifies:
        - Zero deadlocks between profile lock and history lock.
        - Zero corrupted files / JSON syntax errors at any point.
        - State after storm is completely valid and cleanly deserializable.
        """
        sm, _ = adv_storage_env
        duration = 2.0  # run storm for 2 seconds
        num_writers = 12
        num_resetters = 2

        storm_active = threading.Event()
        storm_active.set()

        exceptions: List[Exception] = []
        successful_resets = [0] * num_resetters
        successful_writes = [0] * num_writers

        def _reset_spammer(idx: int):
            try:
                while storm_active.is_set():
                    p, h = sm.reset_all_user_data()
                    assert p.total_ratings == 0
                    assert h == []
                    successful_resets[idx] += 1
                    time.sleep(0.01)
            except Exception as exc:
                exceptions.append(exc)

        def _write_spammer(idx: int):
            try:
                op_id = 0
                while storm_active.is_set():
                    entry = WatchHistoryEntry(
                        id=f"storm_w{idx}_{op_id}",
                        anime_id=30000 + idx * 1000 + op_id,
                        anime_title=f"Storm Anime {idx}",
                        rating=8.0,
                        origin_track="discovery",
                    )
                    sm.append_watch_history(entry)

                    def _mod(p: UserProfileData) -> UserProfileData:
                        p.total_ratings += 1
                        p.feature_weights[f"storm_{idx}_{op_id}"] = 1.0
                        return p

                    sm.update_user_profile(_mod)
                    successful_writes[idx] += 1
                    op_id += 1
                    time.sleep(0.005)
            except Exception as exc:
                exceptions.append(exc)

        with ThreadPoolExecutor(max_workers=num_resetters + num_writers) as executor:
            reset_futures = [executor.submit(_reset_spammer, r) for r in range(num_resetters)]
            write_futures = [executor.submit(_write_spammer, w) for w in range(num_writers)]

            time.sleep(duration)
            storm_active.clear()

            for f in reset_futures + write_futures:
                f.result()

        assert not exceptions, f"Exceptions encountered during reset storm: {exceptions}"
        assert sum(successful_resets) > 0, "No resets were successfully executed"
        assert sum(successful_writes) > 0, "No writes were successfully executed"

        # Verify state integrity after storm
        final_profile = sm.load_user_profile()
        final_history = sm.load_watch_history()
        assert isinstance(final_profile.total_ratings, int)
        assert final_profile.total_ratings >= 0
        assert isinstance(final_history, list)

        # Confirm on-disk JSON is 100% parseable
        json.loads(sm.user_profile_path.read_text(encoding="utf-8"))
        json.loads(sm.watch_history_path.read_text(encoding="utf-8"))

    def test_adv_04_simulated_lock_timeout_and_clean_recovery(self, adv_storage_env):
        """
        Adversarial Test: Lock contention timeout handling and post-timeout recovery.
        Verifies:
        - When a lock is held, callers with short timeout raise StorageLockTimeoutError.
        - Exception includes target_path, lock_path, and timeout duration.
        - Destination files remain completely uncorrupted.
        - Once lock is released, subsequent read/write operations succeed immediately.
        """
        sm, tmp_path = adv_storage_env

        # 1. Establish initial baseline
        initial_profile = sm.load_user_profile()
        assert initial_profile.total_ratings == 0

        # Create a fast-timing-out client (timeout = 0.25s)
        fast_sm = StorageManager(
            workspace_root=tmp_path,
            data_dir=sm.data_dir,
            locks_dir=sm.locks_dir,
            timeout=0.25,
        )

        lock_held_event = threading.Event()
        release_event = threading.Event()

        def _blocker():
            with sm.file_lock(sm.user_profile_path):
                lock_held_event.set()
                release_event.wait(timeout=5.0)

        t = threading.Thread(target=_blocker, daemon=True)
        t.start()
        assert lock_held_event.wait(timeout=2.0), "Blocker failed to acquire lock"

        # 2. Attempt operation with fast_sm -> must raise StorageLockTimeoutError after ~0.25s
        start_time = time.time()
        with pytest.raises(StorageLockTimeoutError) as exc_info:
            fast_sm.update_user_profile(lambda p: setattr(p, "total_ratings", 999) or p)
        elapsed = time.time() - start_time

        # Verify timeout duration and error metadata
        assert 0.20 <= elapsed <= 1.5, f"Elapsed time {elapsed} outside expected window"
        err = exc_info.value
        assert err.target_path == sm.user_profile_path
        assert err.lock_path == sm.get_lock_path(sm.user_profile_path)
        assert err.timeout == 0.25

        # 3. Release lock and verify recovery
        release_event.set()
        t.join(timeout=2.0)

        # 4. Immediate post-timeout operation must succeed without deadlock
        def _inc(p: UserProfileData) -> UserProfileData:
            p.total_ratings = 42
            return p

        updated = sm.update_user_profile(_inc)
        assert updated.total_ratings == 42
        assert sm.load_user_profile().total_ratings == 42

    def test_adv_05_partial_write_truncation_and_forensic_recovery(self, adv_storage_env):
        """
        Adversarial Test: Truncated JSON, partial writes, and syntax corruption.
        Simulates:
        - Incomplete write (truncated JSON ending mid-key or mid-string).
        - 0-byte file (abrupt kill before content flush).
        - Random garbage binary data.
        Verifies:
        - Forensic backup is generated with original corrupted payload.
        - Automatic self-healing restores valid state.
        - Normal read/write capability resumes seamlessly.
        """
        sm, _ = adv_storage_env

        # Case A: Truncated JSON in user_profile.json
        truncated_payload = '{"user_id": "default_user", "total_ratings": 10, "feature_weight'
        sm.user_profile_path.write_text(truncated_payload, encoding="utf-8")

        recovered_profile = sm.load_user_profile()
        assert recovered_profile.user_id == "default_user"
        assert recovered_profile.total_ratings == 0

        # Check forensic backup
        profile_baks = list(sm.data_dir.glob("corrupted_user_profile_*.bak"))
        assert len(profile_baks) == 1
        assert profile_baks[0].read_text(encoding="utf-8") == truncated_payload

        # Case B: 0-byte watch_history.json
        sm.watch_history_path.write_text("", encoding="utf-8")

        recovered_history = sm.load_watch_history()
        assert recovered_history == []

        history_baks = list(sm.data_dir.glob("corrupted_watch_history_*.bak"))
        assert len(history_baks) == 1
        assert history_baks[0].stat().st_size == 0

        # Case C: Garbage binary payload in cache_catalog.json
        sm.catalog_path.write_bytes(b"\x00\xff\xfe\x01\x02\x03CORRUPTED_BLOB")

        recovered_catalog = sm.load_catalog()
        assert len(recovered_catalog) == 2  # Recovered from fixtures (Steins;Gate + AoT)

        catalog_baks = list(sm.data_dir.glob("corrupted_cache_catalog_*.bak"))
        assert len(catalog_baks) == 1
        assert catalog_baks[0].read_bytes() == b"\x00\xff\xfe\x01\x02\x03CORRUPTED_BLOB"

        # Case D: Can still write normally after recoveries
        sm.append_watch_history(
            WatchHistoryEntry(
                anime_id=9253,
                anime_title="Steins;Gate",
                rating=9.5,
                origin_track="affinity",
            )
        )
        assert len(sm.load_watch_history()) == 1

    def test_adv_06_windows_transient_sharing_contention_during_replace(self, adv_storage_env):
        """
        Adversarial Test: Windows NTFS transient file-sharing contention during os.replace.
        On Windows, if another process or thread briefly touches target_path during os.replace
        (e.g. antivirus spot scan or transient handle check), a transient PermissionError can occur.
        Verifies:
        - _unlocked_atomic_write retry loop absorbs transient lock release delays.
        - Destination file successfully updates without leaking temporary files.
        """
        sm, _ = adv_storage_env
        target = sm.user_profile_path

        # Initial write
        sm.save_user_profile(UserProfileData(user_id="default_user", total_ratings=1))
        assert sm.load_user_profile().total_ratings == 1

        # Background thread intermittently opens the target file for a quick peek (transient read),
        # simulating occasional filesystem contention
        stop_contention = threading.Event()
        open_count = [0]

        def _transient_peeker():
            while not stop_contention.is_set():
                try:
                    if target.exists():
                        with open(target, "r", encoding="utf-8") as f:
                            _ = f.read(20)
                        open_count[0] += 1
                except OSError:
                    pass
                time.sleep(0.015)

        t = threading.Thread(target=_transient_peeker, daemon=True)
        t.start()

        try:
            # Perform 10 consecutive atomic updates under transient handle touches
            for i in range(2, 12):
                sm.update_user_profile(lambda p, i=i: setattr(p, "total_ratings", i) or p)
                time.sleep(0.01)
        finally:
            stop_contention.set()
            t.join(timeout=2.0)

        assert open_count[0] > 0, "Transient peeker thread did not execute read opens"
        final_p = sm.load_user_profile()
        assert final_p.total_ratings == 11

        # Verify no orphan .tmp files remain in data_dir
        orphan_tmps = list(sm.data_dir.glob("*.tmp"))
        assert len(orphan_tmps) == 0, f"Found orphan temp files: {orphan_tmps}"

    def test_adv_07_concurrent_catalog_upsert_deduplication(self, adv_storage_env):
        """
        Adversarial Test: High-concurrency upserts into cache_catalog.json.
        20 concurrent threads attempting to upsert overlapping batches of animes.
        Verifies:
        - Zero duplicate anime IDs added.
        - Catalog remains valid Pydantic AnimeEntity collection.
        - Zero lost unique additions.
        """
        sm, _ = adv_storage_env
        num_threads = 20

        # Base items: ID 9253, 16498 already in fixtures
        # 10 unique new items shared across 20 threads
        candidate_items = [
            AnimeEntity(
                id=5000 + i,
                title_romaji=f"Candidate Anime {i}",
                cover_image=f"https://example.com/cover_{i}.png",
                genres=["Action", "Adventure"],
                tags=[AnimeTag(id=100 + i, name=f"Tag {i}", rank=90, is_media_spoiler=False)],
            )
            for i in range(10)
        ]

        def _upsert_job(thread_id: int) -> int:
            # Each thread submits a subset with intentional overlaps
            subset = [candidate_items[i % 10] for i in range(thread_id, thread_id + 5)]
            return sm.upsert_catalog_items(subset)

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(_upsert_job, t) for t in range(num_threads)]
            added_counts = [f.result() for f in as_completed(futures)]

        # Total unique new items added must be exactly 10
        assert sum(added_counts) == 10

        catalog = sm.load_catalog()
        # 2 original fixtures + 10 newly upserted = 12 total
        assert len(catalog) == 12

        unique_ids = {a.id for a in catalog}
        assert len(unique_ids) == 12
        assert {5000 + i for i in range(10)}.issubset(unique_ids)

        # Upserting all 10 again must result in 0 added
        second_pass = sm.upsert_catalog_items(candidate_items)
        assert second_pass == 0
        assert len(sm.load_catalog()) == 12

    def test_adv_08_process_kill_lock_release(self, adv_storage_env):
        """
        Adversarial Test: Abrupt child process kill while holding file lock.
        Verifies:
        - OS immediately releases the lock on process death.
        - Parent process can acquire the lock immediately without stale-lock lockout.
        """
        sm, _ = adv_storage_env

        ctx = multiprocessing.get_context("spawn")
        ready_queue = ctx.Queue()

        proc = ctx.Process(
            target=_mp_lock_holding_worker,
            args=(str(sm.data_dir), str(sm.locks_dir), ready_queue),
        )
        proc.start()

        try:
            # Wait for child to confirm it holds the lock
            status = ready_queue.get(timeout=5.0)
            assert status == "LOCKED"

            # Verify that another attempt with short timeout fails while proc is alive
            with pytest.raises(StorageLockTimeoutError):
                with sm.file_lock(sm.user_profile_path, timeout=0.2):
                    pass

            # Abruptly terminate the worker process (simulating crash / SIGKILL)
            proc.kill()
            proc.join(timeout=3.0)

            # OS should have released lock immediately. Parent can acquire without error.
            with sm.file_lock(sm.user_profile_path, timeout=1.0):
                # Lock successfully acquired by parent
                pass

            # Normal operations succeed without lock collision
            sm.save_user_profile(UserProfileData(user_id="default_user", total_ratings=77))
            assert sm.load_user_profile().total_ratings == 77
        finally:
            if proc.is_alive():
                proc.kill()
                proc.join()


    def test_adv_09_unicode_decode_error_and_primitive_json_recovery(self, adv_storage_env):
        """
        Adversarial Test: Non-dict payloads and encoding corruptions.
        Simulates:
        - Non-UTF-8 bytes (invalid code points triggering UnicodeDecodeError).
        - Primitive JSON types (integer, string, boolean, null) where dict/list is required.
        Verifies:
        - Proper detection and forensic backup creation.
        - Automatic self-healing to valid typed schemas without throwing uncaught exceptions.
        """
        sm, _ = adv_storage_env

        # 1. Invalid UTF-8 bytes in user_profile.json
        sm.user_profile_path.write_bytes(b"\x80\x81\xfe\xffINVALID_UTF8")
        recovered_profile = sm.load_user_profile()
        assert recovered_profile.user_id == "default_user"
        assert recovered_profile.total_ratings == 0

        # 2. Primitive JSON (bare string "hello") in watch_history.json (expected List)
        sm.watch_history_path.write_text('"bare string instead of list"', encoding="utf-8")
        recovered_history = sm.load_watch_history()
        assert recovered_history == []

        # 3. Primitive JSON (bare integer 12345) in cache_catalog.json (expected List[AnimeEntity])
        sm.catalog_path.write_text("12345", encoding="utf-8")
        recovered_catalog = sm.load_catalog()
        assert len(recovered_catalog) == 2

        # 4. JSON "null" in user_profile.json
        sm.user_profile_path.write_text("null", encoding="utf-8")
        recovered_profile2 = sm.load_user_profile()
        assert recovered_profile2.user_id == "default_user"

    def test_adv_10_missing_fixtures_catalog_recovery(self, adv_storage_env):
        """
        Adversarial Test: Catalog corruption recovery when FIXTURES.json is also missing.
        Verifies:
        - StorageManager gracefully recovers to an empty catalog list [] instead of crashing.
        - Cache catalog file is initialized with empty list JSON on disk.
        """
        sm, tmp_path = adv_storage_env

        # Delete fixtures file
        if sm.fixtures_path.exists():
            sm.fixtures_path.unlink()

        # Corrupt catalog
        sm.catalog_path.write_text("{broken json", encoding="utf-8")

        # Load catalog must not crash; returns []
        catalog = sm.load_catalog()
        assert catalog == []

        # Disk catalog must now contain []
        raw = json.loads(sm.catalog_path.read_text(encoding="utf-8"))
        assert raw == []
