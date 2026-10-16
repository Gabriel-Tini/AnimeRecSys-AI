"""
tests/test_challenger_m2_math_stress.py
Adversarial Empirical Stress Test Suite for Milestone 2 (Challenger M2.1).

Validates mathematical stability, numerical invariants, edge cases, and algorithmic properties:
1. Consecutive updates: 100, 200, 500 consecutive extreme ratings (R=10.0 and R=1.0),
   verifying | ||u_t||_2 - 1.0 | < 1e-6 at EVERY step across dense, sparse, and domain models.
2. Taste inversion: repeated R=1.0 ratings invert preference vector, driving cosine similarity below -0.90.
3. MMR diversity: empirical Intra-List Diversity (ILD) on discovery track vs affinity track,
   verifying Discovery ILD is strictly higher.
4. Empty/Zero vectors: empty genres/tags produce valid zero vectors without NaNs,
   and cosine similarity with zero vectors returns 0.0.
5. Partitioning edge cases: K in [1, 2, 3, 5, 10, 50, 100] and pool sizes smaller than K,
   verifying disjointness, partition sums, and zero crashes.
6. Adversarial boundaries: non-finite ratings, extreme magnitudes (1e150), and neutral stability.
"""

import math
import numpy as np
import pytest

try:
    from app.models.anime import AnimeEntity, AnimeTag
    from app.models.recommendation import RecommendationResponse
    from app.models.user import Hyperparameters, UserProfileData
    from app.services.learning_engine import LearningEngine
    from app.services.recommendation_engine import RecommendationEngine
    from app.services.vector_engine import VectorEngine
except ImportError:
    from backend.app.models.anime import AnimeEntity, AnimeTag
    from backend.app.models.recommendation import RecommendationResponse
    from backend.app.models.user import Hyperparameters, UserProfileData
    from backend.app.services.learning_engine import LearningEngine
    from backend.app.services.recommendation_engine import RecommendationEngine
    from backend.app.services.vector_engine import VectorEngine


# ===========================================================================
# 1. Consecutive Updates Spherical Norm Invariant Stress Tests
# ===========================================================================

class TestConsecutiveUpdatesSphericalNormInvariant:
    """
    Stress-tests the spherical normalization invariant:
    | ||u_t||_2 - 1.0 | < 1e-6 at EVERY step across 100, 200, and 500 extreme updates.
    """

    def test_consecutive_100_extreme_positive_updates(self):
        """100 consecutive R=10.0 updates on random unit vectors maintain unit norm at every step."""
        engine = LearningEngine()
        rng = np.random.default_rng(1001)
        dim = 20
        u = np.ones(dim, dtype=np.float64) / math.sqrt(dim)

        for step in range(100):
            raw = rng.uniform(0.1, 1.0, size=dim)
            v = raw / np.linalg.norm(raw)
            u = engine.update_vector(u, v, rating=10.0)

            norm = float(np.linalg.norm(u))
            assert abs(norm - 1.0) < 1e-6, f"Norm drifted at step {step}: norm={norm}"
            assert not np.isnan(u).any(), f"NaN detected at step {step}"
            assert not np.isinf(u).any(), f"Inf detected at step {step}"

    def test_consecutive_200_extreme_negative_updates(self):
        """200 consecutive R=1.0 updates on random unit vectors maintain unit norm at every step."""
        engine = LearningEngine()
        rng = np.random.default_rng(2002)
        dim = 20
        u = np.ones(dim, dtype=np.float64) / math.sqrt(dim)

        for step in range(200):
            raw = rng.uniform(0.1, 1.0, size=dim)
            v = raw / np.linalg.norm(raw)
            u = engine.update_vector(u, v, rating=1.0)

            norm = float(np.linalg.norm(u))
            assert abs(norm - 1.0) < 1e-6, f"Norm drifted at step {step}: norm={norm}"
            assert not np.isnan(u).any(), f"NaN detected at step {step}"
            assert not np.isinf(u).any(), f"Inf detected at step {step}"

    def test_consecutive_500_alternating_extreme_updates(self):
        """500 consecutive alternating extreme updates (R=10.0, 1.0, 10.0...) maintain unit norm at every step."""
        engine = LearningEngine()
        rng = np.random.default_rng(5005)
        dim = 20
        u = np.ones(dim, dtype=np.float64) / math.sqrt(dim)

        for step in range(500):
            raw = rng.uniform(0.1, 1.0, size=dim)
            v = raw / np.linalg.norm(raw)
            rating = 10.0 if (step % 2 == 0) else 1.0
            u = engine.update_vector(u, v, rating=rating)

            norm = float(np.linalg.norm(u))
            assert abs(norm - 1.0) < 1e-6, f"Norm drifted at step {step}: norm={norm}"
            assert not np.isnan(u).any(), f"NaN detected at step {step}"
            assert not np.isinf(u).any(), f"Inf detected at step {step}"

    def test_consecutive_500_sparse_dictionary_updates(self):
        """500 consecutive extreme updates on sparse feature weights maintain unit norm at every step."""
        engine = LearningEngine()
        rng = np.random.default_rng(777)
        feature_names = [f"Tag_{i}" for i in range(25)]
        current_weights: dict[str, float] = {}

        for step in range(500):
            n_tags = int(rng.integers(2, 6))
            chosen = list(rng.choice(feature_names, size=n_tags, replace=False))
            raw = {t: float(rng.uniform(0.2, 1.0)) for t in chosen}
            raw_norm = math.sqrt(sum(w * w for w in raw.values()))
            item_weights = {t: w / raw_norm for t, w in raw.items()}

            rating = 10.0 if (step % 3 == 0) else 1.0
            current_weights = engine.update_feature_weights(
                current_weights=current_weights,
                item_weights=item_weights,
                rating=rating,
            )

            if current_weights:
                norm = math.sqrt(sum(w * w for w in current_weights.values()))
                assert abs(norm - 1.0) < 1e-6, f"Sparse norm drifted at step {step}: norm={norm}"
                for k, v in current_weights.items():
                    assert not math.isnan(v), f"NaN in sparse weight {k} at step {step}"
                    assert not math.isinf(v), f"Inf in sparse weight {k} at step {step}"

    def test_consecutive_500_domain_model_updates(self):
        """500 consecutive extreme ratings through update_profile_rating increment total_ratings and maintain norm."""
        engine = LearningEngine()
        rng = np.random.default_rng(888)
        profile = UserProfileData(user_id="stress_tester", feature_weights={}, total_ratings=0)

        all_genres = ["Action", "Sci-Fi", "Drama", "Fantasy", "Comedy", "Supernatural", "Mystery"]

        for step in range(500):
            g = str(rng.choice(all_genres))
            anime = AnimeEntity(
                id=step + 1,
                title_romaji=f"Anime {step + 1}",
                cover_image="http://example.com/cover.jpg",
                genres=[g],
                tags=[AnimeTag(id=1, name="Plot Twist", rank=90, is_media_spoiler=True)],
            )
            rating = 10.0 if (step % 2 == 0) else 1.0
            profile = engine.update_profile_rating(profile, anime, rating=rating)

            assert profile.total_ratings == step + 1
            if profile.feature_weights:
                norm = math.sqrt(sum(w * w for w in profile.feature_weights.values()))
                assert abs(norm - 1.0) < 1e-6, f"Domain profile norm drifted at step {step}: norm={norm}"


# ===========================================================================
# 2. Taste Inversion and Repulsion Dynamics
# ===========================================================================

class TestTasteInversionAndRepulsionDynamics:
    """
    Empirically verifies taste inversion: repeated R=1.0 ratings drive
    the user preference vector in the opposite direction (cosine similarity drops below -0.90).
    """

    def test_taste_inversion_from_isotropic_cold_start(self):
        """Repeated R=1.0 ratings on an anime drive cosine similarity below -0.90 within 5 steps."""
        engine = LearningEngine()
        dim = 10
        u = np.ones(dim, dtype=np.float64) / math.sqrt(dim)
        target = np.zeros(dim, dtype=np.float64)
        target[0] = 1.0  # Targeted feature

        initial_sim = engine.cosine_similarity(u, target)
        assert initial_sim > 0.0

        sim_trajectory = []
        for step in range(15):
            u = engine.update_vector(u, target, rating=1.0)
            sim = engine.cosine_similarity(u, target)
            sim_trajectory.append(sim)

        # Cosine similarity must drop below -0.90
        assert sim_trajectory[-1] < -0.90, f"Taste inversion failed to drop below -0.90: final_sim={sim_trajectory[-1]}"
        # In fact, after 10+ steps it converges to -1.0
        assert sim_trajectory[-1] <= -0.999, f"Expected near -1.0 convergence: final_sim={sim_trajectory[-1]}"
        # Ensure monotonic decrease until saturation
        for i in range(len(sim_trajectory) - 1):
            assert sim_trajectory[i] >= sim_trajectory[i + 1] - 1e-9, f"Trajectory not monotonically decreasing at step {i}"

    def test_taste_inversion_from_multi_genre_active_profile(self):
        """Active profile with Action=0.7071, Sci-Fi=0.7071 inverts Action affinity below -0.90 on repeated R=1.0."""
        engine = LearningEngine()
        dim = 5
        # u starts liking dim 0 (Action) and dim 1 (Sci-Fi)
        u = np.array([0.70710678, 0.70710678, 0.0, 0.0, 0.0], dtype=np.float64)
        hated = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        for _ in range(25):
            u = engine.update_vector(u, hated, rating=1.0)

        sim = engine.cosine_similarity(u, hated)
        assert sim < -0.90, f"Cosine similarity did not drop below -0.90: sim={sim}"
        assert u[0] < -0.90, f"Target dimension weight did not drop below -0.90: u[0]={u[0]}"
        # Unit norm preserved
        assert abs(np.linalg.norm(u) - 1.0) < 1e-6

    def test_taste_inversion_sparse_dictionary_preserves_negative_weights(self):
        """update_feature_weights retains negative weights without clamping to 0.0."""
        engine = LearningEngine()
        initial = {"Action": 0.70710678, "Romance": 0.70710678}
        hated_item = {"Action": 1.0}

        weights = initial
        for _ in range(10):
            weights = engine.update_feature_weights(weights, hated_item, rating=1.0)

        assert "Action" in weights
        assert weights["Action"] < 0.0, f"Action weight was clamped: {weights['Action']}"
        norm = math.sqrt(sum(w * w for w in weights.values()))
        assert abs(norm - 1.0) < 1e-6


# ===========================================================================
# 3. MMR Intra-List Diversity (ILD) Empirical Superiority
# ===========================================================================

class TestMMRIntraListDiversityEmpirical:
    """
    Empirically calculates Intra-List Diversity (ILD) on discovery track vs affinity track
    and verifies that Discovery ILD is strictly higher.
    """

    def test_discovery_ild_strictly_higher_than_affinity_ild_top_10(self):
        """Top-10 recommendation (K_aff=8, K_disc=2) proves Discovery ILD > Affinity ILD."""
        vocab = ["Action", "Sci-Fi", "Fantasy", "Romance", "Comedy", "Horror", "Drama", "Mystery"]
        ve = VectorEngine(vocabulary=vocab)
        rec = RecommendationEngine(vector_engine=ve)

        catalog: list[AnimeEntity] = []
        # 10 Action/Sci-Fi items (homogeneous cluster)
        for i in range(10):
            catalog.append(
                AnimeEntity(
                    id=i + 1,
                    title_romaji=f"Action SciFi {i + 1}",
                    cover_image="http://example.com/img.png",
                    genres=["Action", "Sci-Fi"],
                    tags=[],
                    mean_score=80 + i,
                    popularity=1000 - i,
                )
            )

        # 10 Diverse items (distinct genres)
        diverse_genres = ["Fantasy", "Romance", "Comedy", "Horror", "Drama", "Mystery", "Fantasy", "Romance", "Comedy", "Horror"]
        for i, g in enumerate(diverse_genres):
            catalog.append(
                AnimeEntity(
                    id=100 + i,
                    title_romaji=f"{g} Anime {i + 1}",
                    cover_image="http://example.com/img.png",
                    genres=[g],
                    tags=[],
                    mean_score=75,
                    popularity=500,
                )
            )

        profile = UserProfileData(
            user_id="user_ild_test",
            feature_weights={"Action": 0.7071, "Sci-Fi": 0.7071},
        )

        res = rec.generate_recommendations(profile=profile, catalog=catalog, history=[], top_k=10)

        assert len(res.affinity_track) == 8
        assert len(res.discovery_track) == 2

        aff_vecs = [ve.build_feature_vector(item) for item in res.affinity_track]
        disc_vecs = [ve.build_feature_vector(item) for item in res.discovery_track]

        aff_ild = RecommendationEngine.compute_intra_list_diversity(aff_vecs)
        disc_ild = RecommendationEngine.compute_intra_list_diversity(disc_vecs)

        # The affinity items are all Action/Sci-Fi (pairwise similarity = 1.0 -> ILD = 0.0)
        # The discovery items are distinct diverse genres (pairwise similarity = 0.0 -> ILD = 1.0)
        assert disc_ild > aff_ild, f"Discovery ILD ({disc_ild}) must be strictly higher than Affinity ILD ({aff_ild})"
        assert np.isclose(aff_ild, 0.0, atol=1e-5)
        assert np.isclose(disc_ild, 1.0, atol=1e-5)

    def test_discovery_ild_strictly_higher_than_affinity_ild_top_20(self):
        """Top-20 recommendation (K_aff=16, K_disc=4) proves Discovery ILD > Affinity ILD."""
        genres = ["Action", "Sci-Fi", "Fantasy", "Romance", "Comedy", "Horror", "Drama", "Mystery"]
        ve = VectorEngine(vocabulary=genres)
        rec = RecommendationEngine(vector_engine=ve)

        catalog: list[AnimeEntity] = []
        # 20 Action/Sci-Fi items
        for i in range(20):
            catalog.append(
                AnimeEntity(
                    id=i + 1,
                    title_romaji=f"Action SciFi {i + 1}",
                    cover_image="http://example.com/img.png",
                    genres=["Action", "Sci-Fi"],
                    tags=[],
                    mean_score=80 + i,
                    popularity=1000 - i,
                )
            )

        diverse_genres = ["Fantasy", "Romance", "Comedy", "Horror", "Drama", "Mystery"]
        for i in range(30):
            g1 = diverse_genres[i % len(diverse_genres)]
            g2 = diverse_genres[(i + 1) % len(diverse_genres)]
            catalog.append(
                AnimeEntity(
                    id=100 + i,
                    title_romaji=f"{g1} {g2} {i + 1}",
                    cover_image="http://example.com/img.png",
                    genres=[g1, g2],
                    tags=[],
                    mean_score=75,
                    popularity=500,
                )
            )

        profile = UserProfileData(
            user_id="user_top20",
            feature_weights={"Action": 0.7071, "Sci-Fi": 0.7071},
        )

        res = rec.generate_recommendations(profile=profile, catalog=catalog, history=[], top_k=20)

        assert len(res.affinity_track) == 16
        assert len(res.discovery_track) == 4

        aff_vecs = [ve.build_feature_vector(item) for item in res.affinity_track]
        disc_vecs = [ve.build_feature_vector(item) for item in res.discovery_track]

        aff_ild = RecommendationEngine.compute_intra_list_diversity(aff_vecs)
        disc_ild = RecommendationEngine.compute_intra_list_diversity(disc_vecs)

        assert disc_ild > aff_ild, f"Discovery ILD ({disc_ild}) must be strictly higher than Affinity ILD ({aff_ild})"
        assert disc_ild > 0.70, f"Expected high Discovery ILD, got {disc_ild}"

    def test_mmr_extreme_lambda_diversity_maximization(self):
        """MMR with lambda=0.0 selects candidate with minimum similarity to selected set."""
        engine = LearningEngine()
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        cands = [
            np.array([0.9, 0.1, 0.0], dtype=np.float64),
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        ]
        # Item 0 is already in selected set
        selected = [cands[0]]

        # lambda=0.0: pure diversity -> should NOT pick cands[0]
        chosen_idx = engine.compute_mmr(u, cands, selected, lambda_param=0.0)
        assert chosen_idx in [1, 2], f"Pure diversity selected redundant item: {chosen_idx}"


# ===========================================================================
# 4. Empty and Zero Vector Edge Cases
# ===========================================================================

class TestEmptyAndZeroVectorHandling:
    """
    Verifies that empty genres/tags produce valid zero vectors without NaNs,
    and cosine similarity with zero vectors returns 0.0 without errors or NaNs.
    """

    def test_empty_genres_and_tags_produces_clean_zero_vector(self):
        """AnimeEntity with empty genres and empty tags produces clean zero vector."""
        ve = VectorEngine()
        anime = AnimeEntity(
            id=9999,
            title_romaji="Empty Meta Anime",
            cover_image="http://example.com/cover.jpg",
            genres=[],
            tags=[],
        )
        vec = ve.build_feature_vector(anime)
        assert vec.shape == (ve.dimension,)
        assert np.all(vec == 0.0)
        assert not np.isnan(vec).any()
        assert not np.isinf(vec).any()

    def test_cosine_similarity_with_zero_vectors_returns_zero(self):
        """Cosine similarity with zero vectors returns 0.0 exactly, no NaNs."""
        ve = VectorEngine()
        u = ve.get_cold_start_profile(strategy="isotropic")
        zero = np.zeros(ve.dimension, dtype=np.float64)

        sim1 = ve.cosine_similarity(u, zero)
        sim2 = ve.cosine_similarity(zero, u)
        sim3 = ve.cosine_similarity(zero, zero)

        assert sim1 == 0.0
        assert sim2 == 0.0
        assert sim3 == 0.0
        assert not np.isnan([sim1, sim2, sim3]).any()

    def test_batch_matrix_similarity_with_zero_vectors(self):
        """Batch matrix scoring against zero vectors produces zero scores without NaNs."""
        ve = VectorEngine()
        u = ve.get_cold_start_profile(strategy="isotropic")
        matrix_with_zeros = np.zeros((10, ve.dimension), dtype=np.float64)

        scores = ve.compute_similarity(u, matrix_with_zeros)
        assert scores.shape == (10,)
        assert np.all(scores == 0.0)
        assert not np.isnan(scores).any()

    def test_learning_engine_cosine_similarity_with_zero_vectors(self):
        """LearningEngine.cosine_similarity returns 0.0 for zero vectors."""
        engine = LearningEngine()
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        zero = np.zeros(3, dtype=np.float64)

        assert engine.cosine_similarity(u, zero) == 0.0
        assert engine.cosine_similarity(zero, zero) == 0.0

    def test_spherical_projection_of_zero_and_sub_epsilon_vectors(self):
        """Normalizing a zero or sub-epsilon vector returns a zero vector safely."""
        ve = VectorEngine()
        v_zero = np.zeros(10, dtype=np.float64)
        v_sub_eps = np.full(10, 1e-15, dtype=np.float64)

        res_zero = ve.normalize_vector(v_zero)
        res_sub_eps = ve.normalize_vector(v_sub_eps)

        assert np.all(res_zero == 0.0)
        assert np.all(res_sub_eps == 0.0)


# ===========================================================================
# 5. Partitioning Edge Cases
# ===========================================================================

class TestPartitioningEdgeCases:
    """
    Stress-tests partition_k and generate_recommendations across:
    K in [1, 2, 3, 5, 10, 50, 100] and pool sizes smaller than K.
    """

    @pytest.mark.parametrize("top_k", [1, 2, 3, 5, 10, 50, 100])
    @pytest.mark.parametrize("pool_size", [0, 1, 2, 3, 4, 5, 8, 10, 20, 50, 100, 200])
    def test_partition_k_invariants_across_all_combinations(self, top_k: int, pool_size: int):
        """Verifies K_aff + K_disc == min(top_k, pool_size) and bounds for all (K, P) pairs."""
        k_aff, k_disc = RecommendationEngine.partition_k(top_k, pool_size)

        expected_total = min(top_k, pool_size)
        assert k_aff + k_disc == expected_total
        assert k_aff >= 0
        assert k_disc >= 0
        assert k_aff <= pool_size
        assert k_disc <= pool_size

        if pool_size == 0 or top_k == 0:
            assert k_aff == 0 and k_disc == 0
        elif pool_size < top_k:
            # Pool smaller than K: total returned must equal pool size exactly
            assert k_aff + k_disc == pool_size

    @pytest.mark.parametrize("top_k", [1, 2, 3, 5, 10, 50, 100])
    def test_generate_recommendations_with_pool_smaller_than_k(self, top_k: int):
        """RecommendationEngine handles catalog pools smaller than top_k cleanly."""
        rec = RecommendationEngine()
        user = UserProfileData(user_id="test_partition_user", feature_weights={"Action": 1.0})

        # Test with pools smaller than top_k
        test_pool_sizes = [0, 1, 2, 3, 5]
        for p in test_pool_sizes:
            catalog = [
                AnimeEntity(
                    id=i + 1,
                    title_romaji=f"Anime {i + 1}",
                    cover_image="http://example.com/cover.jpg",
                    genres=["Action"] if i % 2 == 0 else ["Drama"],
                    tags=[],
                    mean_score=70 + i,
                    popularity=500 - i,
                )
                for i in range(p)
            ]

            resp = rec.generate_recommendations(profile=user, catalog=catalog, history=[], top_k=top_k)
            assert isinstance(resp, RecommendationResponse)

            k_aff_expected, k_disc_expected = RecommendationEngine.partition_k(top_k, p)
            assert len(resp.affinity_track) == k_aff_expected
            assert len(resp.discovery_track) == k_disc_expected
            assert len(resp.affinity_track) + len(resp.discovery_track) == min(top_k, p)

            # Check mutually disjoint invariant
            aff_ids = {a.id for a in resp.affinity_track}
            disc_ids = {d.id for d in resp.discovery_track}
            assert aff_ids.isdisjoint(disc_ids)


# ===========================================================================
# 6. Adversarial Numerical Boundaries
# ===========================================================================

class TestAdversarialNumericalBoundaries:
    """
    Tests extreme adversarial inputs, numerical bounds, and floating-point stability.
    """

    def test_rating_gradient_delta_defensive_clamping(self):
        """Extreme and malformed ratings are clamped without NaN/Inf or crashing."""
        engine = LearningEngine()
        v = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        adversarial_ratings = [
            float("inf"),
            float("-inf"),
            float("nan"),
            1e30,
            -1e30,
            "corrupted_input",
            None,
        ]

        for bad in adversarial_ratings:
            delta = engine.rating_gradient_delta(bad, v)  # type: ignore
            assert not np.isnan(delta).any(), f"NaN generated for input: {bad}"
            assert not np.isinf(delta).any(), f"Inf generated for input: {bad}"
            assert abs(delta[0]) <= 0.675 + 1e-9, f"Gradient exceeded maximal delta: {delta}"

    def test_extreme_vector_magnitudes_without_overflow(self):
        """Vectors with magnitude 1e150 or 1e-150 normalize cleanly without overflow."""
        ve = VectorEngine()
        v_huge = np.array([1e150, 1e150], dtype=np.float64)
        norm_huge = ve.normalize_vector(v_huge)

        assert abs(np.linalg.norm(norm_huge) - 1.0) < 1e-6
        assert not np.isnan(norm_huge).any()

        v_tiny = np.array([1e-15, 1e-15], dtype=np.float64)
        norm_tiny = ve.normalize_vector(v_tiny)
        assert np.all(norm_tiny == 0.0)

    def test_neutral_rating_stability_invariance(self):
        """Neutral rating R=5.5 produces zero gradient delta and preserves vector direction."""
        engine = LearningEngine()
        dim = 5
        raw_u = np.array([0.577, 0.577, 0.577, 0.0, 0.0], dtype=np.float64)
        u = raw_u / np.linalg.norm(raw_u)
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        delta = engine.rating_gradient_delta(5.5, v)
        assert np.allclose(delta, 0.0, atol=1e-12)

        u_updated = engine.update_vector(u, v, rating=5.5)
        # Vector decayed by gamma and projected back onto sphere = same direction
        cosine = engine.cosine_similarity(u, u_updated)
        assert np.isclose(cosine, 1.0, atol=1e-6)
