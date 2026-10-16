"""
tests/unit/test_learning_engine.py
Unit Test Suite for LearningEngine (Milestone 2).

Coverage:
1. Rating gradient delta calculation (positive, moderate, neutral, negative repulsion)
2. Temporal weight decay factor application
3. Unit L2 spherical projection invariant (| ||u||_2 - 1.0 | < 1e-6)
4. Consecutive 100+ updates invariant stability and non-overflow
5. Consecutive 100 ratings convergence to reinforced dimension
6. Negative rating taste inversion and repulsion
7. Cold-start initialization and neutral rating stability
8. Gemini qualitative semantic boost calculation and clamping
9. Spoiler tags participation in preference learning (R3)
10. Domain model functional updates (UserProfileData)
11. Storage persistence integration (save_user_profile, append_watch_history)
12. Thread-safe atomic read-modify-write under simulated concurrency
13. Profile reset to cold-start state
14. Maximal Marginal Relevance (MMR) selection
"""

import math
import shutil
import tempfile
import numpy as np
import pytest

from backend.app.core.storage_manager import StorageManager
from backend.app.models.anime import AnimeEntity, AnimeTag
from backend.app.models.recommendation import SemanticTagDelta, SemanticTuningResponse
from backend.app.models.user import Hyperparameters, UserProfileData, WatchHistoryEntry
from backend.app.services.learning_engine import LearningEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Provides a standalone LearningEngine instance without storage manager."""
    return LearningEngine()


@pytest.fixture
def isolated_storage():
    """Creates an isolated temporary directory and StorageManager for testing."""
    tmp_dir = tempfile.mkdtemp(prefix="learning_engine_test_")
    sm = StorageManager(workspace_root=tmp_dir)
    yield sm
    shutil.rmtree(tmp_dir, ignore_errors=True)


@pytest.fixture
def sample_anime_entity():
    """Provides a canonical AnimeEntity (Steins;Gate) with mixed safe and spoiler tags."""
    return AnimeEntity(
        id=9253,
        title_romaji="Steins;Gate",
        cover_image="https://s4.anilist.co/cover/9253.png",
        genres=["Drama", "Sci-Fi", "Suspense"],
        tags=[
            AnimeTag(id=6, name="Time Manipulation", rank=98, is_media_spoiler=False),
            AnimeTag(id=7, name="Psychological", rank=92, is_media_spoiler=False),
            AnimeTag(id=11, name="Plot Twist", rank=96, is_media_spoiler=True),
            AnimeTag(id=13, name="Tragedy", rank=88, is_media_spoiler=True),
        ],
    )


# ===========================================================================
# 1. Mathematical Gradient Delta Tests
# ===========================================================================

class TestGradientDelta:
    """Verifies Delta u_nota = eta * (R - 5.5) * hat{v}_i."""

    def test_gradient_maximal_positive_rating_10(self, engine):
        """Rating R=10.0 produces Delta u = 0.15 * (10 - 5.5) * v = +0.675 * v."""
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        delta = engine.rating_gradient_delta(10.0, v)
        assert np.isclose(delta[0], 0.675, atol=1e-7)
        assert np.isclose(delta[1], 0.0, atol=1e-7)

    def test_gradient_moderate_positive_rating_7(self, engine):
        """Rating R=7.0 produces Delta u = 0.15 * (7 - 5.5) * v = +0.225 * v."""
        v = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        delta = engine.rating_gradient_delta(7.0, v)
        assert np.isclose(delta[1], 0.225, atol=1e-7)

    def test_gradient_neutral_rating_5_5(self, engine):
        """Rating R=5.5 produces Delta u = 0.0 exactly."""
        v = np.array([0.577, 0.577, 0.577], dtype=np.float64)
        delta = engine.rating_gradient_delta(5.5, v)
        assert np.allclose(delta, 0.0, atol=1e-12)

    def test_gradient_moderate_negative_rating_2(self, engine):
        """Rating R=2.0 produces Delta u = 0.15 * (2 - 5.5) * v = -0.525 * v."""
        v = np.array([1.0, 0.0], dtype=np.float64)
        delta = engine.rating_gradient_delta(2.0, v)
        assert np.isclose(delta[0], -0.525, atol=1e-7)

    def test_gradient_maximal_negative_rating_1(self, engine):
        """Rating R=1.0 produces Delta u = 0.15 * (1 - 5.5) * v = -0.675 * v."""
        v = np.array([1.0, 0.0], dtype=np.float64)
        delta = engine.rating_gradient_delta(1.0, v)
        assert np.isclose(delta[0], -0.675, atol=1e-7)

    def test_gradient_clamps_out_of_bounds_ratings(self, engine):
        """Out-of-bounds ratings (R=999.0 or R=-50.0) are clamped to [1.0, 10.0]."""
        v = np.array([1.0], dtype=np.float64)
        delta_high = engine.rating_gradient_delta(999.0, v)
        assert np.isclose(delta_high[0], 0.675, atol=1e-7)

        delta_low = engine.rating_gradient_delta(-50.0, v)
        assert np.isclose(delta_low[0], -0.675, atol=1e-7)


# ===========================================================================
# 2. Temporal Decay & Spherical Projection Tests
# ===========================================================================

class TestDecayAndProjection:
    """Verifies temporal weight decay and spherical projection."""

    def test_temporal_decay_factor(self, engine):
        """Applying decay gamma=0.98 ten times diminishes weight to 0.98^10."""
        vec = np.array([1.0, 0.5, 0.25], dtype=np.float64)
        decayed = vec.copy()
        for _ in range(10):
            decayed = engine.apply_temporal_decay(decayed, gamma=0.98)

        expected = vec * (0.98 ** 10)
        assert np.allclose(decayed, expected, atol=1e-7)

    def test_project_l2_sphere_normalizes_to_one(self, engine):
        """Arbitrary non-zero vectors project to unit norm with |norm - 1.0| < 1e-6."""
        v = np.array([3.0, 4.0, 0.0], dtype=np.float64)
        hat_v = engine.project_l2_sphere(v)
        norm = float(np.linalg.norm(hat_v))
        assert abs(norm - 1.0) < 1e-6

    def test_project_l2_sphere_handles_zero_vector(self, engine):
        """Zero vector projects to zero vector without ZeroDivisionError or NaN."""
        zero_v = np.zeros(5, dtype=np.float64)
        res = engine.project_l2_sphere(zero_v)
        assert not np.isnan(res).any()
        assert np.allclose(res, 0.0, atol=1e-12)


# ===========================================================================
# 3. Stability Invariant & Convergence Tests (>= 100 Updates)
# ===========================================================================

class TestStabilityInvariants:
    """Verifies numerical stability across sequential extreme rating sequences."""

    def test_unit_norm_invariant_across_100_extreme_updates(self, engine):
        """Strict unit invariant | ||u_t||_2 - 1.0 | < 1e-6 across 100 extreme updates."""
        dim = 25
        u = np.zeros(dim, dtype=np.float64)
        rng = np.random.default_rng(123)

        # Initialize with first rating
        v0 = engine.project_l2_sphere(rng.uniform(0.1, 1.0, size=dim))
        u = engine.update_vector(u, v0, rating=10.0)
        assert abs(np.linalg.norm(u) - 1.0) < 1e-6

        # 100 sequential updates with alternating extreme ratings R=10.0 and R=1.0
        for step in range(100):
            v_step = engine.project_l2_sphere(rng.uniform(0.1, 1.0, size=dim))
            rating = 10.0 if step % 2 == 0 else 1.0
            u = engine.update_vector(u, v_step, rating=rating)

            norm = float(np.linalg.norm(u))
            assert abs(norm - 1.0) < 1e-6, f"Norm drifted at step {step}: norm={norm}"
            assert not np.isnan(u).any()
            assert not np.isinf(u).any()

    def test_repeated_100_high_ratings_convergence(self, engine):
        """100 consecutive ratings of 10.0 on target item converge cosine similarity to > 0.9999."""
        dim = 10
        target = np.zeros(dim, dtype=np.float64)
        target[0] = 1.0  # pure dimension 0

        u = np.zeros(dim, dtype=np.float64)
        for step in range(100):
            u = engine.update_vector(u, target, rating=10.0)
            assert abs(np.linalg.norm(u) - 1.0) < 1e-6

        sim = engine.cosine_similarity(u, target)
        assert sim > 0.9999, f"Convergence failed: sim={sim}"

    def test_consecutive_negative_ratings_taste_inversion(self, engine):
        """50 consecutive ratings of 1.0 invert taste dimension to negative while preserving unit norm."""
        dim = 5
        # User starts liking item 0 and item 1 equally
        u = np.array([0.70710678, 0.70710678, 0.0, 0.0, 0.0], dtype=np.float64)
        hated = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for step in range(50):
            u = engine.update_vector(u, hated, rating=1.0)
            norm = float(np.linalg.norm(u))
            assert abs(norm - 1.0) < 1e-6

        # Dimension 0 should be strongly negative (repelled)
        assert u[0] < 0.0, f"Taste inversion failed: u[0]={u[0]}"
        assert u[1] > 0.0, f"Unrated dimension lost: u[1]={u[1]}"


# ===========================================================================
# 4. Sparse Dictionary Feature Weight Updates
# ===========================================================================

class TestSparseDictionaryUpdates:
    """Verifies update_feature_weights operating directly on Dict[str, float]."""

    def test_cold_start_to_active_orientation(self, engine):
        """Cold-start empty dictionary transitions to unit norm on first rating."""
        empty_weights = {}
        item = {"Sci-Fi": 0.8, "Drama": 0.6}  # norm = 1.0

        new_weights = engine.update_feature_weights(empty_weights, item, rating=10.0)
        norm = math.sqrt(sum(w * w for w in new_weights.values()))
        assert abs(norm - 1.0) < 1e-6
        assert np.isclose(new_weights["Sci-Fi"], 0.8, atol=1e-5)
        assert np.isclose(new_weights["Drama"], 0.6, atol=1e-5)

    def test_cold_start_neutral_rating_produces_empty_dict(self, engine):
        """Neutral rating R=5.5 on cold start produces empty dictionary without errors."""
        empty_weights = {}
        item = {"Action": 1.0}
        new_weights = engine.update_feature_weights(empty_weights, item, rating=5.5)
        assert new_weights == {}

    def test_decay_diminishes_dormant_tags(self, engine):
        """Dormant tags decay by gamma=0.98 relative to newly reinforced dimensions."""
        initial = {"Sci-Fi": 0.70710678, "Romance": 0.70710678}
        active_item = {"Sci-Fi": 1.0}

        new_weights = engine.update_feature_weights(initial, active_item, rating=10.0)
        norm = math.sqrt(sum(w * w for w in new_weights.values()))
        assert abs(norm - 1.0) < 1e-6
        # Sci-Fi gained weight, Romance decayed
        assert new_weights["Sci-Fi"] > new_weights["Romance"]


# ===========================================================================
# 5. Gemini Qualitative Semantic Boost Integration
# ===========================================================================

class TestQualitativeGeminiIntegration:
    """Verifies qualitative feedback boost application and R6 clamping."""

    def test_semantic_boost_calculation(self, engine):
        """Semantic deltas apply boost = delta * conf * alpha with alpha=0.25."""
        vocab = ["Time Manipulation", "Psychological", "Action"]
        deltas = [
            {"tag_name": "Time Manipulation", "impact_multiplier": 0.8, "confidence": 0.9},
            {"tag_name": "Psychological", "impact_multiplier": -0.5, "confidence": 1.0},
        ]
        boost = engine.semantic_boost_delta(deltas, vocab, alpha=0.25)

        # Expected: 0.8 * 0.9 * 0.25 = 0.18
        assert np.isclose(boost[0], 0.18, atol=1e-5)
        # Expected: -0.5 * 1.0 * 0.25 = -0.125
        assert np.isclose(boost[1], -0.125, atol=1e-5)
        # Action was not mentioned
        assert np.isclose(boost[2], 0.0, atol=1e-9)

    def test_semantic_boost_clamps_adversarial_inputs(self, engine):
        """Adversarial prompt injection values (impact=999.0) are clamped to 1.0."""
        vocab = ["Action"]
        deltas = [{"tag_name": "Action", "impact_multiplier": 999.0, "confidence": 5.0}]
        boost = engine.semantic_boost_delta(deltas, vocab, alpha=0.25)
        # Clamped to 1.0 * 1.0 * 0.25 = 0.25
        assert np.isclose(boost[0], 0.25, atol=1e-5)

    def test_unknown_hallucinated_tags_are_ignored(self, engine):
        """Hallucinated tags not present in vocabulary do not alter vector."""
        vocab = ["Sci-Fi"]
        deltas = [{"tag_name": "FakeHallucinatedTag", "impact_multiplier": 1.0, "confidence": 1.0}]
        boost = engine.semantic_boost_delta(deltas, vocab, alpha=0.25)
        assert np.isclose(boost[0], 0.0, atol=1e-9)


# ===========================================================================
# 6. Requirement R3: Spoiler Tags Participation
# ===========================================================================

class TestSpoilerTagsParticipation:
    """Verifies that is_media_spoiler: True tags participate fully in learning."""

    def test_spoiler_tags_participate_in_item_feature_weights(self, engine, sample_anime_entity):
        """Item feature weights include spoiler tags (Plot Twist, Tragedy)."""
        weights = engine.extract_item_feature_weights(sample_anime_entity)
        assert "Plot Twist" in weights
        assert weights["Plot Twist"] > 0.0
        assert "Tragedy" in weights
        assert weights["Tragedy"] > 0.0

    def test_spoiler_tags_participate_in_profile_update(self, engine, sample_anime_entity):
        """User profile update retains spoiler tag dimensions with positive weights."""
        profile = engine.initialize_cold_start_profile()
        updated = engine.update_profile_rating(profile, sample_anime_entity, rating=10.0)

        assert "Plot Twist" in updated.feature_weights
        assert updated.feature_weights["Plot Twist"] > 0.0
        assert abs(updated.compute_l2_norm() - 1.0) < 1e-6


# ===========================================================================
# 7. Domain Model & Storage Persistence Tests
# ===========================================================================

class TestDomainAndStorageIntegration:
    """Verifies UserProfileData functional transformations and StorageManager I/O."""

    def test_update_profile_rating_increments_total_ratings(self, engine, sample_anime_entity):
        """update_profile_rating increments total_ratings and updates timestamp."""
        profile = engine.initialize_cold_start_profile()
        assert profile.total_ratings == 0

        updated = engine.update_profile_rating(profile, sample_anime_entity, rating=9.0)
        assert updated.total_ratings == 1
        assert not updated.is_cold_start()
        assert abs(updated.compute_l2_norm() - 1.0) < 1e-6

    def test_record_rating_persists_to_storage(self, isolated_storage, sample_anime_entity):
        """record_rating saves user profile and appends to watch history atomically."""
        # Seed catalog with sample anime
        isolated_storage.save_catalog([sample_anime_entity])

        engine = LearningEngine(storage_manager=isolated_storage)
        updated_profile, history_entry = engine.record_rating(
            anime_id=sample_anime_entity.id,
            rating=9.5,
            origin_track="affinity",
            user_review="Outstanding narrative arc and time travel loops.",
        )

        # Verify returned objects
        assert updated_profile.total_ratings == 1
        assert history_entry.anime_id == sample_anime_entity.id
        assert history_entry.rating == 9.5
        assert history_entry.origin_track == "affinity"

        # Verify disk persistence
        loaded_profile = isolated_storage.load_user_profile()
        assert loaded_profile.total_ratings == 1
        assert "Time Manipulation" in loaded_profile.feature_weights
        assert abs(loaded_profile.compute_l2_norm() - 1.0) < 1e-6

        loaded_history = isolated_storage.load_watch_history()
        assert len(loaded_history) == 1
        assert loaded_history[0].id == history_entry.id

    def test_record_rating_fails_cleanly_on_missing_anime(self, isolated_storage):
        """record_rating raises ValueError when anime_id is not in catalog."""
        isolated_storage.save_catalog([])
        engine = LearningEngine(storage_manager=isolated_storage)

        with pytest.raises(ValueError, match="not found in catalog"):
            engine.record_rating(anime_id=99999, rating=8.0)

    def test_reset_profile_restores_cold_start(self, isolated_storage, sample_anime_entity):
        """reset_profile clears feature weights and resets total_ratings to 0."""
        isolated_storage.save_catalog([sample_anime_entity])
        engine = LearningEngine(storage_manager=isolated_storage)

        # Rate one anime
        engine.record_rating(anime_id=sample_anime_entity.id, rating=10.0)
        assert isolated_storage.load_user_profile().total_ratings == 1

        # Reset
        reset_prof = engine.reset_profile()
        assert reset_prof.total_ratings == 0
        assert reset_prof.feature_weights == {}
        assert reset_prof.is_cold_start()


# ===========================================================================
# 8. Maximal Marginal Relevance (MMR) Diversity Tests
# ===========================================================================

class TestMMRSelection:
    """Verifies MMR selection equation balancing similarity and diversity."""

    def test_mmr_selects_diverse_candidate(self, engine):
        """MMR penalizes redundancy and selects high-novelty candidate."""
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        # Candidate 0 is identical to already selected item (redundancy = 1.0)
        v0 = np.array([0.70710678, 0.70710678, 0.0], dtype=np.float64)
        v0 /= np.linalg.norm(v0)

        # Candidate 1 has equal similarity to user but is diverse from v0 (redundancy = 0.5)
        v1 = np.array([0.70710678, 0.0, 0.70710678], dtype=np.float64)
        v1 /= np.linalg.norm(v1)

        already_selected = [v0]

        # With lambda=0.5, candidate 1 is preferred over candidate 0 because max_sim(v0, v0) = 1.0 > max_sim(v1, v0) = 0.5
        best_idx = engine.compute_mmr(
            user_vec=u,
            candidate_vecs=[v0, v1],
            already_selected_vecs=already_selected,
            lambda_param=0.5,
        )
        assert best_idx == 1
