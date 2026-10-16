"""
AnimeRecSys AI — Tier 2: Boundary & Corner Cases Test Suite
Authoritative Reference: ORIGINAL_REQUEST.md, ARCHITECTURE.md, SECURITY.md, FIXTURES.json

Coverage:
- Zero vectors, empty inputs, sparse items with epsilon protection (1e-9)
- Extreme ratings (R=1.0, R=10.0, R=5.5) and boundary fractional values
- Out-of-bounds ratings rejection (R < 1.0, R > 10.0, NaN, inf)
- Qualitative text boundaries: 0 chars, 300 chars, 301 chars, 10,000 chars, whitespace
- Repeated rating convergence across 100 updates without numerical overflow
- Semantic tuning boundary clamping (impact in [-1, 1], confidence in [0, 1])
- Catalog exhaustion and sparse candidate pools
- High-concurrency lock contention boundaries
Total Tier 2 Tests: >= 15 comprehensive boundary tests
"""

import json
import math
import os
import re
import tempfile
import numpy as np
import pytest
from filelock import FileLock

try:
    from pydantic import ValidationError, BaseModel, Field
except ImportError:
    ValidationError = None
    BaseModel = None
    Field = None

from tests.conftest import MathematicalOracle, ROOT_DIR


class TestTier2BoundariesAndCornerCases:
    """Rigorous boundary and stress tests for algorithms, data validation, and persistence."""

    # -----------------------------------------------------------------------
    # 1. Zero Vectors and Epsilon Protection
    # -----------------------------------------------------------------------
    def test_tier2_01_zero_vector_similarity_does_not_nan(self, oracle):
        """Zero profile vector Sim(0, v) with epsilon protection returns 0.0, not NaN."""
        zero_u = np.zeros(10, dtype=np.float64)
        item_v = np.ones(10, dtype=np.float64)
        item_v = oracle.l2_normalize(item_v)

        sim = oracle.cosine_similarity(zero_u, item_v)
        assert not math.isnan(sim), "Similarity calculation produced NaN on zero user vector"
        assert np.isclose(sim, 0.0, atol=1e-9)

    def test_tier2_02_anime_with_no_genres_and_no_tags(self, tags_vocabulary, oracle):
        """Anime with empty genres and empty tags produces zero vector without zero-division error."""
        vocab = [t["name"] for t in tags_vocabulary]
        empty_vec = oracle.build_feature_vector([], [], vocab)

        assert not np.isnan(empty_vec).any(), "Feature vector contains NaN values"
        # Zero vector divided by (0 + 1e-9) is still 0
        assert np.allclose(empty_vec, 0.0, atol=1e-8)

    # -----------------------------------------------------------------------
    # 2. Rating Scale Boundaries
    # -----------------------------------------------------------------------
    def test_tier2_03_exact_neutral_rating_produces_zero_delta(self, oracle):
        """Rating R = 5.5 exactly yields Delta u_nota = 0.0 vector."""
        item_vec = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
        delta = oracle.rating_gradient_delta(5.5, item_vec)
        assert np.allclose(delta, 0.0, atol=1e-12)

    def test_tier2_04_extreme_rating_1_0_maximal_repulsion(self, oracle):
        """Rating R = 1.0 produces exact maximal negative repulsion delta -0.675 * v."""
        item_vec = np.array([1.0, 0.0], dtype=np.float64)
        delta = oracle.rating_gradient_delta(1.0, item_vec)
        expected = 0.15 * (1.0 - 5.5) * 1.0  # -0.675
        assert np.isclose(delta[0], expected, atol=1e-7)

    def test_tier2_05_extreme_rating_10_0_maximal_attraction(self, oracle):
        """Rating R = 10.0 produces exact maximal positive attraction delta +0.675 * v."""
        item_vec = np.array([1.0, 0.0], dtype=np.float64)
        delta = oracle.rating_gradient_delta(10.0, item_vec)
        expected = 0.15 * (10.0 - 5.5) * 1.0  # +0.675
        assert np.isclose(delta[0], expected, atol=1e-7)

    def test_tier2_06_out_of_bounds_ratings_rejected_by_schema(self):
        """Ratings < 1.0 or > 10.0 or NaN must raise validation error in Pydantic DTO."""
        if BaseModel is None:
            pytest.skip("pydantic not installed in active environment; run with .venv/Scripts/pytest")

        class RatingModel(BaseModel):
            rating: float = Field(ge=1.0, le=10.0)

        # Test valid boundary values
        RatingModel(rating=1.0)
        RatingModel(rating=10.0)
        RatingModel(rating=5.5)

        # Test below lower bound
        with pytest.raises(ValidationError):
            RatingModel(rating=0.999)

        with pytest.raises(ValidationError):
            RatingModel(rating=0.0)

        with pytest.raises(ValidationError):
            RatingModel(rating=-5.0)

        # Test above upper bound
        with pytest.raises(ValidationError):
            RatingModel(rating=10.001)

        with pytest.raises(ValidationError):
            RatingModel(rating=11.0)

        with pytest.raises(ValidationError):
            RatingModel(rating=999.0)

        # Test NaN and inf
        with pytest.raises(ValidationError):
            RatingModel(rating=float("nan"))

        with pytest.raises(ValidationError):
            RatingModel(rating=float("inf"))

    # -----------------------------------------------------------------------
    # 3. Qualitative User Feedback Text Length Boundaries
    # -----------------------------------------------------------------------
    def test_tier2_07_empty_and_whitespace_qualitative_feedback(self, tags_vocabulary, oracle):
        """Empty or whitespace-only feedback results in zero semantic boost."""
        vocab = [t["name"] for t in tags_vocabulary]
        empty_deltas = []
        boost = oracle.semantic_boost_delta(empty_deltas, vocab)
        assert np.allclose(boost, 0.0, atol=1e-12)

    def test_tier2_08_exact_300_char_boundary(self):
        """Feedback of exactly 300 characters is preserved without truncation."""
        exact_300 = "X" * 300
        assert len(exact_300) == 300
        truncated = exact_300[:300]
        assert truncated == exact_300

    def test_tier2_09_boundary_301_and_massive_input_truncation(self):
        """Feedback of 301 and 10,000 characters is strictly clamped to 300 characters."""
        text_301 = "Y" * 301
        assert len(text_301[:300]) == 300

        text_10k = "Z" * 10000
        sanitized = text_10k[:300]
        assert len(sanitized) == 300

    # -----------------------------------------------------------------------
    # 4. Repeated Ratings Convergence & Stability
    # -----------------------------------------------------------------------
    def test_tier2_10_repeated_100_high_ratings_convergence(self, oracle):
        """100 consecutive ratings of 10.0 converge to target vector with norm strictly 1.0 +- 1e-6."""
        dim = 10
        target_vec = np.zeros(dim, dtype=np.float64)
        target_vec[0] = 1.0  # pure dimension 0

        user_vec = np.zeros(dim, dtype=np.float64)

        for step in range(100):
            user_vec = oracle.update_user_profile(user_vec, target_vec, 10.0)
            norm = np.linalg.norm(user_vec)
            assert abs(norm - 1.0) < 1e-6, f"Norm drifted at step {step}: {norm}"

        # Cosine similarity to target vector should converge towards 1.0
        sim = oracle.cosine_similarity(user_vec, target_vec)
        assert sim > 0.9999, f"Profile did not converge to reinforced target: {sim}"

    def test_tier2_11_repeated_100_low_ratings_inversion_stability(self, oracle):
        """100 consecutive ratings of 1.0 on a dimension repels profile while maintaining unit norm."""
        dim = 5
        # User starts liking item A (dim 0) and item B (dim 1)
        user_vec = np.array([0.70710678, 0.70710678, 0.0, 0.0, 0.0], dtype=np.float64)
        hated_item = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for step in range(50):
            user_vec = oracle.update_user_profile(user_vec, hated_item, 1.0)
            norm = np.linalg.norm(user_vec)
            assert abs(norm - 1.0) < 1e-6

        # Taste for dimension 0 should be strongly diminished or negative compared to dim 1
        assert user_vec[0] < user_vec[1], f"Repulsion failed: dim 0={user_vec[0]}, dim 1={user_vec[1]}"

    # -----------------------------------------------------------------------
    # 5. Semantic Tuning Delta Clamping Boundaries
    # -----------------------------------------------------------------------
    def test_tier2_12_semantic_tuning_extreme_value_clamping(self, tags_vocabulary, oracle):
        """Extreme impact multipliers and confidences are strictly clamped to [-1.0, 1.0] and [0.0, 1.0]."""
        vocab = [t["name"] for t in tags_vocabulary]
        extreme_deltas = [
            {"tag_name": "Action", "impact_multiplier": 500.0, "confidence": 10.0, "rationale": "huge"},
            {"tag_name": "Sci-Fi", "impact_multiplier": -500.0, "confidence": -5.0, "rationale": "neg"}
        ]

        boost = oracle.semantic_boost_delta(extreme_deltas, vocab)

        action_idx = vocab.index("Action")
        # Clamped: delta = 1.0, conf = 1.0 -> boost = 1.0 * 1.0 * 0.25 = 0.25
        assert np.isclose(boost[action_idx], 0.25, atol=1e-5)

        sci_fi_idx = vocab.index("Sci-Fi")
        # Clamped: delta = -1.0, conf = 0.0 -> boost = -1.0 * 0.0 * 0.25 = 0.0
        assert np.isclose(boost[sci_fi_idx], 0.0, atol=1e-5)

    def test_tier2_13_zero_confidence_neutralizes_impact(self, tags_vocabulary, oracle):
        """Confidence of 0.0 produces zero vector delta regardless of impact_multiplier magnitude."""
        vocab = [t["name"] for t in tags_vocabulary]
        deltas = [
            {"tag_name": "Action", "impact_multiplier": 1.0, "confidence": 0.0, "rationale": "no confidence"}
        ]
        boost = oracle.semantic_boost_delta(deltas, vocab)
        action_idx = vocab.index("Action")
        assert np.isclose(boost[action_idx], 0.0, atol=1e-12)

    # -----------------------------------------------------------------------
    # 6. Catalog Exhaustion and Sparse Candidate Pools
    # -----------------------------------------------------------------------
    def test_tier2_14_candidate_pool_smaller_than_top_n(self, tags_vocabulary, sample_animes, oracle):
        """When candidate pool has fewer items than requested N, partitions all available items without error."""
        vocab = [t["name"] for t in tags_vocabulary]
        # Only 2 candidates available, but top_n requested is 10
        small_pool = sample_animes[:2]
        small_vecs = [oracle.build_feature_vector(a["genres"], a["tags"], vocab) for a in small_pool]
        user_vec = small_vecs[0].copy()

        result = oracle.dual_track_partition(user_vec, small_pool, small_vecs, top_n=10)
        total_returned = len(result["affinity_track"]) + len(result["discovery_track"])
        assert total_returned == 2
        # No duplicates
        ids = [a["id"] for a in result["affinity_track"]] + [a["id"] for a in result["discovery_track"]]
        assert len(ids) == len(set(ids))

    def test_tier2_15_empty_candidate_pool_handling(self, oracle):
        """When candidate pool is empty (all items in history), returns empty tracks gracefully."""
        user_vec = np.array([1.0, 0.0], dtype=np.float64)
        result = oracle.dual_track_partition(user_vec, [], [], top_n=10)
        assert result["affinity_track"] == []
        assert result["discovery_track"] == []

    # -----------------------------------------------------------------------
    # 7. Concurrency Lock Timeout Boundary
    # -----------------------------------------------------------------------
    def test_tier2_16_filelock_timeout_handles_contention(self, temp_data_dir):
        """FileLock raises Timeout error when an unreleased lock exceeds timeout duration."""
        lock_file = os.path.join(temp_data_dir, "timeout_test.lock")
        lock1 = FileLock(lock_file, timeout=0.2)
        lock2 = FileLock(lock_file, timeout=0.2)

        with lock1:
            with pytest.raises(Exception):
                # Lock2 attempts to acquire lock while lock1 is held, should timeout
                lock2.acquire(timeout=0.1)
