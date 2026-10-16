"""
tests/test_challenger_m3_2_empirical_probes.py
Comprehensive Empirical Regression & Integration Challenge Test Suite.
Milestone 3 - AnimeRecSys AI (Challenger M3.2).

Validates:
1. SensitiveDataFilter logging secret redaction across formats, chained exceptions, and circular refs.
2. StorageManager atomic persistence and thread concurrency under interleaved writes and enrichment.
3. VectorEngine VSM representation and spherical norm invariant | ||v||_2 - 1.0 | < 1e-6.
4. LearningEngine rating gradient delta, temporal decay, taste inversion, and 100-step plasticity.
5. RecommendationEngine 80/20 partitioning, watch history exclusion, and MMR diversity.
"""

from __future__ import annotations

import io
import json
import logging
import math
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import pytest

from backend.app.core.config import SensitiveDataFilter
from backend.app.core.storage_manager import StorageManager
from backend.app.models.anime import AnimeEntity, AnimeTag
from backend.app.models.recommendation import (
    RecommendationResponse,
    SemanticTagDelta,
    SemanticTuningResponse,
)
from backend.app.models.user import Hyperparameters, UserProfileData, WatchHistoryEntry
from backend.app.services.learning_engine import LearningEngine
from backend.app.services.recommendation_engine import RecommendationEngine
from backend.app.services.spoiler_service import CANONICAL_SPOILER_TAGS, SpoilerService
from backend.app.services.vector_engine import (
    CANONICAL_FIXTURE_TAGS,
    VectorEngine,
    cosine_similarity,
    l2_normalize,
    normalize_vector,
)


# =============================================================================
# Logging Helpers
# =============================================================================
def make_probe_logger(
    name: str,
    filter_obj: SensitiveDataFilter,
) -> Tuple[logging.Logger, io.StringIO]:
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    handler.setFormatter(formatter)
    handler.addFilter(filter_obj)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger, stream


# =============================================================================
# Suite 1: SensitiveDataFilter Logging Defense & Regressions
# =============================================================================
class TestSensitiveDataFilterLoggingProbes:
    """Empirical probe suite verifying SensitiveDataFilter logging redaction."""

    def test_gemini_key_redaction_in_messages_and_args(self):
        gemini_key = "AIzaSyB1234567890abcdefghijklmnopqrstuvw"
        filter_obj = SensitiveDataFilter(exact_secrets=[gemini_key])
        logger, stream = make_probe_logger("probe_gemini_key", filter_obj)

        logger.info("Connecting with key: %s", gemini_key)
        logger.warning(f"Direct interpolation: {gemini_key}")
        logger.error(f"ANSI formatted \x1b[31m{gemini_key}\x1b[0m")
        logger.info(f"URL: https://generativelanguage.googleapis.com/v1beta/models?key={gemini_key}")
        logger.info(f"Header: x-goog-api-key: {gemini_key}")
        logger.info("Bearer token: Bearer ya29.a0AfH6SMDI88941234567890==")

        output = stream.getvalue()
        assert gemini_key not in output, "Gemini API key leaked into log output!"
        assert "[REDACTED_GEMINI_KEY]" in output or "[REDACTED_SECRET]" in output
        assert "[REDACTED_TOKEN]" in output

    def test_chained_exceptions_secret_scrubbing(self):
        secret = "AIzaSyDChainedSecret12345678901234567890"
        filter_obj = SensitiveDataFilter(exact_secrets=[secret])
        logger, stream = make_probe_logger("probe_chained_exc", filter_obj)

        try:
            try:
                raise ValueError(f"Root cause with secret {secret}")
            except ValueError as root_err:
                raise RuntimeError(f"Chained wrapper holding {secret}") from root_err
        except RuntimeError as wrapper_err:
            logger.error("Caught chained error", exc_info=wrapper_err)

        output = stream.getvalue()
        assert secret not in output, "Secret leaked in chained exception traceback!"
        assert "[REDACTED_GEMINI_KEY]" in output or "[REDACTED_SECRET]" in output

    def test_circular_and_complex_extra_records_defense(self):
        secret = "AIzaSyECircularSecret1234567890123456789"
        filter_obj = SensitiveDataFilter(exact_secrets=[secret])
        logger, stream = make_probe_logger("probe_circular_extra", filter_obj)

        circular_dict: Dict[str, Any] = {"token": secret}
        circular_dict["self"] = circular_dict

        class CustomUnprintable:
            def __str__(self):
                raise RuntimeError("Intentional __str__ failure")

        # Must not crash with RecursionError or custom unprintable exception
        logger.info(
            "Logging circular payload",
            extra={"circular": circular_dict, "bad_obj": CustomUnprintable()},
        )
        output = stream.getvalue()
        assert secret not in output, "Secret leaked from circular payload!"


# =============================================================================
# Suite 2: StorageManager Concurrency & Persistence Under Interleaving
# =============================================================================
class TestStorageManagerConcurrencyProbes:
    """Empirical probe suite verifying StorageManager atomic concurrency."""

    def test_interleaved_appends_and_enrichments_under_thread_concurrency(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "data"
            locks_dir = data_dir / ".locks"
            fixtures_path = Path(__file__).resolve().parents[1] / ".antigravity" / "FIXTURES.json"

            sm = StorageManager(
                workspace_root=tmp_path,
                data_dir=data_dir,
                locks_dir=locks_dir,
                fixtures_path=fixtures_path,
                timeout=15.0,
            )

            # Pre-seed one watch history entry to allow immediate enrichments
            base_entry = WatchHistoryEntry(
                id="entry_seed_0",
                anime_id=101,
                anime_title="Seed Anime",
                rating=8.0,
                origin_track="affinity",
            )
            sm.append_watch_history(base_entry)

            num_appends = 12
            num_enriches = 8
            num_profile_updates = 5
            total_tasks = num_appends + num_enriches + num_profile_updates

            def do_append(idx: int) -> int:
                entry = WatchHistoryEntry(
                    id=f"entry_thread_{idx}",
                    anime_id=200 + idx,
                    anime_title=f"Thread Anime {idx}",
                    rating=7.0 + (idx % 3),
                    origin_track="discovery" if idx % 2 == 0 else "affinity",
                )
                sm.append_watch_history(entry)
                return idx

            def do_enrich(idx: int) -> bool:
                review_text = f"Review from thread {idx}: Amazing pacing and character arcs."
                tuning_mock = SemanticTuningResponse(
                    identified_tags=[
                        SemanticTagDelta(
                            tag_name="Cyberpunk",
                            impact_multiplier=0.25,
                            confidence=0.8,
                            rationale=f"Thread {idx} extraction",
                        )
                    ],
                    public_safe_reason=f"Safe recommendation reason for thread {idx}.",
                )
                res = sm.enrich_latest_watch_history(
                    user_review_raw=review_text,
                    qualitative_tuning=tuning_mock,
                )
                return res is not None

            def do_profile_update(idx: int) -> int:
                def modifier(prof: UserProfileData) -> UserProfileData:
                    prof.feature_weights[f"tag_thread_{idx}"] = 0.5
                    prof.total_ratings += 1
                    return prof

                sm.update_user_profile(modifier)
                return idx

            futures = []
            with ThreadPoolExecutor(max_workers=25) as executor:
                for i in range(num_appends):
                    futures.append(executor.submit(do_append, i))
                for j in range(num_enriches):
                    futures.append(executor.submit(do_enrich, j))
                for k in range(num_profile_updates):
                    futures.append(executor.submit(do_profile_update, k))

                results = [f.result() for f in as_completed(futures)]

            assert len(results) == total_tasks

            # Post-concurrency verification:
            # 1. Watch history valid JSON, no JSONDecodeError
            history = sm.load_watch_history()
            assert len(history) == 1 + num_appends, (
                f"Expected {1 + num_appends} entries, got {len(history)}! Lost updates detected!"
            )

            # 2. At least one entry should be enriched
            assert any(e.user_review_raw is not None for e in history)
            assert any(e.qualitative_tuning is not None for e in history)

            # 3. User profile valid JSON, total_ratings incremented
            profile = sm.load_user_profile()
            assert profile.total_ratings == num_profile_updates
            assert len(profile.feature_weights) == num_profile_updates

    def test_corrupted_json_recovery_fallback_empirically(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            data_dir = tmp_path / "data"
            locks_dir = data_dir / ".locks"
            fixtures_path = Path(__file__).resolve().parents[1] / ".antigravity" / "FIXTURES.json"

            sm = StorageManager(
                workspace_root=tmp_path,
                data_dir=data_dir,
                locks_dir=locks_dir,
                fixtures_path=fixtures_path,
            )

            # Write broken JSON to watch history
            watch_hist_file = data_dir / "watch_history.json"
            watch_hist_file.write_text("{ broken_corrupt_json: [", encoding="utf-8")

            # Must recover gracefully without raising StorageCorruptedError
            recovered_hist = sm.load_watch_history()
            assert isinstance(recovered_hist, list)


# =============================================================================
# Suite 3: VectorEngine VSM and Spherical Norm Invariant
# =============================================================================
class TestVectorEngineMathematicalInvariants:
    """Empirical probe suite verifying VectorEngine VSM mappings and unit norms."""

    def test_genre_and_community_tag_weighting_scaling(self):
        engine = VectorEngine()
        anime = AnimeEntity(
            id=1,
            title_romaji="Cowboy Bebop",
            cover_image="https://example.com/cover.jpg",
            genres=["Action", "Sci-Fi"],
            tags=[
                AnimeTag(id=1, name="Space", rank=90, is_media_spoiler=False),
                AnimeTag(id=2, name="Tragedy", rank=80, is_media_spoiler=True),
            ],
        )

        vec = engine.build_feature_vector(anime)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-6, f"Unit norm invariant violated: {norm}"

        # In internal vector, spoiler tag Tragedy must participate fully
        tragedy_idx = engine.term_index("Tragedy")
        assert tragedy_idx is not None
        assert vec[tragedy_idx] > 0.0, "Spoiler tag Tragedy failed to participate in vector!"

    def test_extreme_and_degenerate_vector_magnitudes(self):
        # Zero vector projection
        zero_vec = np.zeros(100, dtype=np.float64)
        norm_zero = normalize_vector(zero_vec)
        assert np.all(norm_zero == 0.0)
        assert not np.isnan(norm_zero).any()

        # Extremely huge magnitudes (1e150) without overflow to Inf
        huge_vec = np.full(50, 1e150, dtype=np.float64)
        norm_huge = normalize_vector(huge_vec)
        huge_norm = float(np.linalg.norm(norm_huge))
        assert abs(huge_norm - 1.0) < 1e-6, f"Huge vector norm failed: {huge_norm}"
        assert not np.isinf(norm_huge).any()

        # Subnormal tiny floats (1e-300) without underflow to NaN
        tiny_vec = np.full(50, 1e-300, dtype=np.float64)
        norm_tiny = normalize_vector(tiny_vec)
        assert not np.isnan(norm_tiny).any()


# =============================================================================
# Suite 4: LearningEngine Gradient Delta, Decay, and Taste Inversion
# =============================================================================
class TestLearningEngineMathematicalDynamics:
    """Empirical probe suite verifying LearningEngine dynamics and stability."""

    def test_neutral_rating_produces_zero_delta(self):
        dim = 30
        v = np.zeros(dim, dtype=np.float64)
        v[0] = 1.0

        # R = 5.5 is exact neutral pivot: Delta u = eta * (5.5 - 5.5) * v = 0
        delta = LearningEngine.rating_gradient_delta(rating=5.5, item_vec=v, eta=0.15)
        assert np.allclose(delta, 0.0), f"Neutral rating produced non-zero delta: {delta}"

    def test_temporal_weight_decay_exponential_diminishing(self):
        dim = 10
        weights_vec = np.ones(dim, dtype=np.float64)

        # Apply decay factor gamma = 0.98 across 10 steps
        decayed = weights_vec.copy()
        for _ in range(10):
            decayed = LearningEngine.apply_temporal_decay(decayed, gamma=0.98)

        expected = 1.0 * (0.98**10)
        assert np.allclose(decayed, expected, atol=1e-6), (
            f"Decay mismatch: expected {expected}, got {decayed[0]}"
        )

    def test_taste_inversion_repulsion_dynamics_and_norm_stability(self):
        """
        Verify that successive low ratings (R=1.0) actively repel preferences
        and invert cosine similarity while preserving | ||u||_2 - 1.0 | < 1e-6
        at every single step across 100 consecutive updates.
        """
        dim = 25
        # Initial isotropic cold-start vector across dimensions
        u = np.ones(dim, dtype=np.float64) / math.sqrt(dim)

        target_anime_vec = np.zeros(dim, dtype=np.float64)
        target_anime_vec[0] = 1.0

        initial_sim = LearningEngine.cosine_similarity(u, target_anime_vec)
        assert initial_sim > 0.0

        for step in range(100):
            u = LearningEngine.update_vector(u, target_anime_vec, rating=1.0, eta=0.15, gamma=0.98)
            norm = float(np.linalg.norm(u))
            assert abs(norm - 1.0) < 1e-6, (
                f"Norm invariant violated at step {step}: {norm}"
            )

        final_sim = LearningEngine.cosine_similarity(u, target_anime_vec)
        assert final_sim < -0.90, f"Taste inversion failed! Final similarity: {final_sim}"

    def test_duck_typing_semantic_boost_integration(self):
        engine = LearningEngine()
        prof = UserProfileData(
            feature_weights={"Sci-Fi": 0.5, "Action": 0.5},
            total_ratings=2,
            last_updated="2026-10-16T00:00:00Z",
            hyperparameters=Hyperparameters(),
        )

        tuning = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Cyberpunk", impact_multiplier=0.8, confidence=0.9),
                SemanticTagDelta(tag_name="Romance", impact_multiplier=-0.5, confidence=0.7),
            ],
            public_safe_reason="User enjoys futuristic aesthetic.",
        )

        updated_prof = engine.apply_semantic_boost(prof, tuning)
        assert "Cyberpunk" in updated_prof.feature_weights
        assert updated_prof.feature_weights["Cyberpunk"] > 0.0
        assert "Romance" in updated_prof.feature_weights
        assert updated_prof.feature_weights["Romance"] < 0.0


# =============================================================================
# Suite 5: RecommendationEngine Invariants & Diversity
# =============================================================================
class TestRecommendationEngineProbes:
    """Empirical probe suite verifying 80/20 partitioning, history exclusion, and MMR."""

    @pytest.fixture
    def synthetic_catalog(self) -> List[AnimeEntity]:
        catalog = []
        genres_pool = [
            ["Action", "Sci-Fi"],
            ["Drama", "Romance"],
            ["Comedy", "Slice of Life"],
            ["Fantasy", "Adventure"],
            ["Horror", "Mystery"],
        ]
        for i in range(1, 51):
            genre_pair = genres_pool[i % len(genres_pool)]
            catalog.append(
                AnimeEntity(
                    id=i,
                    title_romaji=f"Catalog Anime {i}",
                    cover_image=f"https://example.com/cover_{i}.jpg",
                    genres=genre_pair,
                    tags=[
                        AnimeTag(id=100 + i, name=f"Tag_{genre_pair[0]}", rank=80, is_media_spoiler=False),
                        AnimeTag(id=200 + i, name="Plot Twist" if i % 5 == 0 else "Safe Tag", rank=90, is_media_spoiler=(i % 5 == 0)),
                    ],
                )
            )
        return catalog

    def test_80_20_partitioning_and_disjointness_guarantee(self, synthetic_catalog):
        rec_engine = RecommendationEngine()
        user_prof = UserProfileData(
            feature_weights={"Action": 0.8, "Sci-Fi": 0.6},
            total_ratings=5,
            last_updated="2026-10-16T00:00:00Z",
            hyperparameters=Hyperparameters(),
        )

        response = rec_engine.generate_recommendations(
            profile=user_prof,
            catalog=synthetic_catalog,
            history=[],
            top_k=10,
        )

        assert len(response.affinity_track) == 8, f"Expected 8 affinity items, got {len(response.affinity_track)}"
        assert len(response.discovery_track) == 2, f"Expected 2 discovery items, got {len(response.discovery_track)}"

        affinity_ids = {item.id for item in response.affinity_track}
        discovery_ids = {item.id for item in response.discovery_track}
        intersection = affinity_ids & discovery_ids
        assert len(intersection) == 0, f"Partition intersection non-empty! Leaked IDs: {intersection}"

    def test_watch_history_strict_exclusion(self, synthetic_catalog):
        rec_engine = RecommendationEngine()
        user_prof = UserProfileData(
            feature_weights={"Action": 1.0, "Sci-Fi": 1.0},
            total_ratings=10,
            last_updated="2026-10-16T00:00:00Z",
            hyperparameters=Hyperparameters(),
        )

        # Mark top Action/Sci-Fi anime (IDs 5, 10, 15, 20) as watched
        history = [
            WatchHistoryEntry(
                id=f"h_{aid}",
                anime_id=aid,
                anime_title=f"Watched {aid}",
                rating=9.0,
                origin_track="affinity",
            )
            for aid in [5, 10, 15, 20]
        ]
        watched_ids = {5, 10, 15, 20}

        response = rec_engine.generate_recommendations(
            profile=user_prof,
            catalog=synthetic_catalog,
            history=history,
            top_k=10,
        )

        recommended_ids = {item.id for item in response.affinity_track} | {item.id for item in response.discovery_track}
        overlap = recommended_ids & watched_ids
        assert len(overlap) == 0, f"Watched anime appeared in recommendations! Leaked IDs: {overlap}"

    def test_mmr_discovery_intra_list_diversity_superiority(self, synthetic_catalog):
        rec_engine = RecommendationEngine()
        user_prof = UserProfileData(
            feature_weights={"Action": 0.9, "Sci-Fi": 0.8},
            total_ratings=8,
            last_updated="2026-10-16T00:00:00Z",
            hyperparameters=Hyperparameters(),
        )

        response = rec_engine.generate_recommendations(
            profile=user_prof,
            catalog=synthetic_catalog,
            history=[],
            top_k=10,
        )

        # Helper to compute Intra-List Diversity (ILD)
        def compute_ild(items: List[Any], vector_engine: VectorEngine) -> float:
            if len(items) < 2:
                return 1.0
            vecs = [vector_engine.build_feature_vector(it) for it in items]
            sims = []
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sims.append(float(np.dot(vecs[i], vecs[j])))
            avg_sim = sum(sims) / len(sims)
            return 1.0 - avg_sim

        v_engine = VectorEngine()
        ild_affinity = compute_ild(response.affinity_track, v_engine)
        ild_discovery = compute_ild(response.discovery_track, v_engine)

        # Discovery track (MMR with lambda=0.50) must promote diversity
        assert ild_discovery >= ild_affinity - 1e-6, (
            f"Discovery ILD ({ild_discovery}) should not be significantly lower than Affinity ILD ({ild_affinity})"
        )

    def test_zero_spoiler_leakage_in_public_recommendation_dto(self, synthetic_catalog):
        rec_engine = RecommendationEngine()
        user_prof = UserProfileData(
            feature_weights={"Action": 0.8},
            total_ratings=3,
            last_updated="2026-10-16T00:00:00Z",
            hyperparameters=Hyperparameters(),
        )

        response = rec_engine.generate_recommendations(
            profile=user_prof,
            catalog=synthetic_catalog,
            history=[],
            top_k=10,
        )

        all_items = list(response.affinity_track) + list(response.discovery_track)
        for item in all_items:
            for tag in item.tags:
                assert tag not in CANONICAL_SPOILER_TAGS, (
                    f"Spoiler tag '{tag}' leaked into public AnimePublicResponse.tags for anime {item.id}!"
                )
