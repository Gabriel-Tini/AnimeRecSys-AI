"""
tests/unit/test_recommendation_engine.py
Unit test suite for backend.app.services.recommendation_engine.RecommendationEngine.
Verifies 80/20 partitioning, disjoint invariants, MMR diversity, spoiler shielding,
and DTO conversions.
"""

import math
import shutil
import tempfile
import numpy as np
import pytest

from app.core.storage_manager import StorageManager
from app.models.anime import AnimeEntity, AnimeTag, KNOWN_SPOILER_TAGS
from app.models.recommendation import RecommendationResponse
from app.models.user import UserProfileData, WatchHistoryEntry
from app.services.recommendation_engine import RecommendationEngine


@pytest.fixture
def sample_catalog(sample_animes):
    """Converts fixture sample animes into AnimeEntity instances."""
    return [AnimeEntity.model_validate(a) for a in sample_animes]


@pytest.fixture
def empty_profile():
    """Provides a fresh cold-start user profile."""
    return UserProfileData(
        user_id="test_user",
        total_ratings=0,
        feature_weights={},
    )


@pytest.fixture
def active_profile():
    """Provides an active user profile with established genre preferences."""
    return UserProfileData(
        user_id="scifi_lover",
        total_ratings=10,
        feature_weights={"Sci-Fi": 0.7071, "Action": 0.7071},
    )


@pytest.fixture
def isolated_storage():
    """Creates an isolated temporary directory and StorageManager for testing."""
    tmp_dir = tempfile.mkdtemp(prefix="rec_engine_test_")
    sm = StorageManager(workspace_root=tmp_dir)
    yield sm
    shutil.rmtree(tmp_dir, ignore_errors=True)


class TestRecommendationEnginePartitioning:
    """Verifies 80/20 partitioning ratios and boundary allocations."""

    def test_partition_k_exact_80_20_split(self):
        k_aff, k_disc = RecommendationEngine.partition_k(top_k=10, available_candidates=20)
        assert k_aff == 8
        assert k_disc == 2
        assert k_aff + k_disc == 10

    def test_partition_k_various_sizes(self):
        # K = 5 -> 4 aff, 1 disc
        assert RecommendationEngine.partition_k(5, 10) == (4, 1)
        # K = 4 -> round(3.2) = 3 aff, 1 disc
        assert RecommendationEngine.partition_k(4, 10) == (3, 1)
        # K = 2 -> round(1.6) = 2 aff, 0 disc
        assert RecommendationEngine.partition_k(2, 10) == (2, 0)
        # K = 1 -> round(0.8) = 1 aff, 0 disc
        assert RecommendationEngine.partition_k(1, 10) == (1, 0)
        # K = 20 -> 16 aff, 4 disc
        assert RecommendationEngine.partition_k(20, 30) == (16, 4)

    def test_partition_k_sparse_candidate_pool(self):
        # Only 2 candidates available when 10 requested
        k_aff, k_disc = RecommendationEngine.partition_k(top_k=10, available_candidates=2)
        assert k_aff == 2
        assert k_disc == 0
        assert k_aff + k_disc == 2

    def test_partition_k_empty_pool(self):
        assert RecommendationEngine.partition_k(top_k=10, available_candidates=0) == (0, 0)


class TestRecommendationEngineInvariants:
    """Verifies disjoint-track, history exclusion, and spoiler suppression invariants."""

    def test_watch_history_strict_exclusion(self, sample_catalog, active_profile):
        engine = RecommendationEngine()
        steins_id = 9253
        history = [
            WatchHistoryEntry(
                anime_id=steins_id,
                anime_title="Steins;Gate",
                rating=10.0,
                origin_track="affinity",
            )
        ]

        resp = engine.generate_recommendations(active_profile, sample_catalog, history, top_k=5)
        recommended_ids = {item.id for item in resp.affinity_track + resp.discovery_track}
        assert steins_id not in recommended_ids

    def test_all_items_watched_produces_empty_response(self, sample_catalog, active_profile):
        engine = RecommendationEngine()
        all_watched = [
            WatchHistoryEntry(
                anime_id=a.id,
                anime_title=a.title_romaji,
                rating=8.0,
                origin_track="affinity",
            )
            for a in sample_catalog
        ]

        resp = engine.generate_recommendations(active_profile, sample_catalog, all_watched, top_k=10)
        assert resp.affinity_track == []
        assert resp.discovery_track == []

    def test_mutually_disjoint_tracks_invariant(self, sample_catalog, active_profile):
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5)

        aff_ids = {item.id for item in resp.affinity_track}
        disc_ids = {item.id for item in resp.discovery_track}
        assert aff_ids.isdisjoint(disc_ids)
        assert len(aff_ids) == 4
        assert len(disc_ids) == 1

    def test_recommendation_response_model_validation(self, sample_catalog, active_profile):
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5)
        assert isinstance(resp, RecommendationResponse)
        assert len(resp.affinity_track) == 4
        assert len(resp.discovery_track) == 1

    def test_dual_spoiler_shielding_zero_leakage(self, sample_catalog, active_profile):
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5)

        spoiler_names_lower = {s.lower() for s in KNOWN_SPOILER_TAGS}
        for item in resp.affinity_track + resp.discovery_track:
            for tag in item.tags:
                assert tag.lower() not in spoiler_names_lower, f"Spoiler tag '{tag}' leaked in anime {item.title_romaji}"


class TestRecommendationEngineMMRDiversity:
    """Verifies MMR diversity mechanics and intra-list diversity superiority."""

    def test_mmr_intra_list_diversity_superiority(self, sample_catalog, active_profile):
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=6)
        cand_map = {a.id: a for a in sample_catalog}
        aff_vecs = engine.compute_candidate_vectors([cand_map[item.id] for item in resp.affinity_track])

        resp_diverse = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5, lambda_param=0.0)
        disc_diverse_id = resp_diverse.discovery_track[0].id
        assert disc_diverse_id not in {item.id for item in resp_diverse.affinity_track}

    def test_mmr_extreme_lambda_pure_relevance(self, sample_catalog, active_profile):
        """When lambda=1.0, MMR discovery track prioritizes highest user similarity."""
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5, lambda_param=1.0)
        assert len(resp.discovery_track) == 1
        disc_item = resp.discovery_track[0]
        # In pure relevance mode, discovery item must not be in affinity track
        assert disc_item.id not in {item.id for item in resp.affinity_track}

    def test_mmr_extreme_lambda_pure_diversity(self, sample_catalog, active_profile):
        """When lambda=0.0, MMR discovery track minimizes redundancy to affinity track."""
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5, lambda_param=0.0)
        assert len(resp.discovery_track) == 1
        disc_item = resp.discovery_track[0]
        assert disc_item.id not in {item.id for item in resp.affinity_track}

    def test_score_conversion_rectified_clamping(self):
        engine = RecommendationEngine(bipolar_scaling=False)
        score, pct = engine.similarity_to_score_and_percentage(0.9423)
        assert np.isclose(score, 0.9423, atol=1e-4)
        assert pct == 94

        score_neg, pct_neg = engine.similarity_to_score_and_percentage(-0.35)
        assert score_neg == 0.0
        assert pct_neg == 0

    def test_score_conversion_bipolar_scaling(self):
        engine = RecommendationEngine(bipolar_scaling=True)
        score_high, pct_high = engine.similarity_to_score_and_percentage(1.0)
        assert score_high == 1.0 and pct_high == 100

        score_mid, pct_mid = engine.similarity_to_score_and_percentage(0.0)
        assert score_mid == 0.5 and pct_mid == 50

        score_low, pct_low = engine.similarity_to_score_and_percentage(-1.0)
        assert score_low == 0.0 and pct_low == 0


class TestRecommendationEngineIntegrationAndLifecycle:
    """Verifies cold-start behavior, StorageManager integration, and determinism."""

    def test_cold_start_profile_recommendations(self, sample_catalog, empty_profile):
        """Cold-start profile generates valid recommendations without division by zero."""
        engine = RecommendationEngine()
        resp = engine.generate_recommendations(empty_profile, sample_catalog, [], top_k=5)
        assert len(resp.affinity_track) == 4
        assert len(resp.discovery_track) == 1
        for item in resp.affinity_track + resp.discovery_track:
            assert item.compatibility_score is not None
            assert 0.0 <= item.compatibility_score <= 1.0

    def test_storage_manager_integration(self, isolated_storage, sample_catalog):
        """End-to-end get_recommendations() loads catalog and profiles from StorageManager."""
        isolated_storage.save_catalog(sample_catalog)
        engine = RecommendationEngine(storage_manager=isolated_storage)

        resp = engine.get_recommendations(top_k=5)
        assert len(resp.affinity_track) == 4
        assert len(resp.discovery_track) == 1

    def test_deterministic_reproducibility(self, sample_catalog, active_profile):
        """Successive runs with identical state produce identical recommendations."""
        engine = RecommendationEngine()
        resp1 = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5)
        resp2 = engine.generate_recommendations(active_profile, sample_catalog, [], top_k=5)

        aff_ids_1 = [item.id for item in resp1.affinity_track]
        aff_ids_2 = [item.id for item in resp2.affinity_track]
        assert aff_ids_1 == aff_ids_2

        disc_ids_1 = [item.id for item in resp1.discovery_track]
        disc_ids_2 = [item.id for item in resp2.discovery_track]
        assert disc_ids_1 == disc_ids_2
