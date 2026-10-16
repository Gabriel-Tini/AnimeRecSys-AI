"""
tests/unit/test_vector_engine.py
Unit test suite for backend.app.services.vector_engine.
Verifies Feature Vocabulary Construction, L2 Spherical Projection Invariants,
Cosine Similarity Kernels, Cold-Start Profiles, and Dual Spoiler Policy compliance.
"""

import math
import numpy as np
import pytest

from backend.app.models.anime import AnimeEntity, AnimeTag
from backend.app.models.user import UserProfileData
from backend.app.services.vector_engine import (
    VectorEngine,
    build_feature_vector,
    normalize_vector,
    l2_normalize,
    cosine_similarity,
    compute_similarity,
    score_candidates,
    get_cold_start_vector,
    CANONICAL_FIXTURE_TAGS,
)


class TestVectorVocabularyConstruction:
    """Tests feature vocabulary initialization, indexing, and spoiler preservation."""

    def test_default_vocabulary_initialization(self):
        engine = VectorEngine()
        assert engine.dimension == len(CANONICAL_FIXTURE_TAGS)
        assert engine.dimension == 20
        assert engine.term_index("Action") == 0
        assert engine.term_index("Music") == 19

    def test_custom_vocabulary_bidirectional_indexing(self):
        custom_vocab = ["Sci-Fi", "Cyberpunk", "Tragedy", "Mecha"]
        engine = VectorEngine(custom_vocab)
        assert engine.dimension == 4
        for idx, term in enumerate(custom_vocab):
            assert engine.term_index(term) == idx
            assert engine._index_to_term[idx] == term

    def test_case_insensitive_and_whitespace_index_lookup(self):
        engine = VectorEngine(["Sci-Fi", "Time Manipulation"])
        assert engine.term_index("sci-fi") == 0
        assert engine.term_index(" SCI-FI ") == 0
        assert engine.term_index("time manipulation") == 1
        assert engine.term_index("NonExistent") is None

    def test_spoiler_tags_present_in_vocabulary(self):
        """R3 Defense: Critical plot spoiler tags must remain active in vocabulary."""
        engine = VectorEngine()
        spoiler_tags = ["Plot Twist", "Betrayal", "Tragedy", "Conspiracy", "Memory Manipulation"]
        for tag in spoiler_tags:
            idx = engine.term_index(tag)
            assert idx is not None, f"Spoiler tag {tag} missing from VSM vocabulary"

    def test_vocabulary_from_catalog_expansion(self):
        catalog = [
            AnimeEntity(
                id=1,
                title_romaji="Anime A",
                cover_image="http://img/1.png",
                genres=["Action", "Mecha"],
                tags=[
                    AnimeTag(id=101, name="Space", rank=90),
                    AnimeTag(id=102, name="Cyberpunk", rank=95),
                ],
            ),
            AnimeEntity(
                id=2,
                title_romaji="Anime B",
                cover_image="http://img/2.png",
                genres=["Drama", "Action"],
                tags=[AnimeTag(id=103, name="Post-Apocalyptic", rank=80)],
            ),
        ]
        base_vocab = ["Action", "Cyberpunk"]
        engine = VectorEngine.from_catalog(catalog, base_vocabulary=base_vocab)
        # Should include base vocab first, then sorted new genres and tags
        assert engine.dimension > len(base_vocab)
        assert engine.term_index("Action") == 0
        assert engine.term_index("Cyberpunk") == 1
        assert engine.term_index("Mecha") is not None
        assert engine.term_index("Space") is not None
        assert engine.term_index("Post-Apocalyptic") is not None


class TestFeatureVectorConstruction:
    """Tests dense vector generation for AnimeEntity instances and fixtures."""

    def test_feature_vector_mapping_genres_and_tags(self):
        vocab = ["Drama", "Sci-Fi", "Time Manipulation", "Plot Twist"]
        engine = VectorEngine(vocab)
        anime = AnimeEntity(
            id=9253,
            title_romaji="Steins;Gate",
            cover_image="http://img/sg.png",
            genres=["Drama", "Sci-Fi"],
            tags=[
                AnimeTag(id=6, name="Time Manipulation", rank=98, is_media_spoiler=False),
                AnimeTag(id=11, name="Plot Twist", rank=96, is_media_spoiler=True),
            ],
        )

        vec = engine.build_feature_vector(anime)
        # Unit norm verification
        norm = np.linalg.norm(vec)
        assert abs(norm - 1.0) < 1e-6

        # Check relative weights: genres have base 1.0, tags have rank / 100
        # Steins;Gate raw: Drama=1.0, Sci-Fi=1.0, Time=0.98, Plot Twist=0.96
        # Expected unnormalized: [1.0, 1.0, 0.98, 0.96]
        expected_raw = np.array([1.0, 1.0, 0.98, 0.96], dtype=np.float64)
        expected_norm = expected_raw / (np.linalg.norm(expected_raw) + 1e-9)
        assert np.allclose(vec, expected_norm, atol=1e-6)

    def test_r3_spoiler_tags_participate_in_vector(self):
        """R3.1 Verification: Internal vector MUST include weights for spoiler tags."""
        engine = VectorEngine(CANONICAL_FIXTURE_TAGS)
        anime = AnimeEntity(
            id=16498,
            title_romaji="Shingeki no Kyojin",
            cover_image="http://img/aot.png",
            genres=["Action", "Drama"],
            tags=[
                AnimeTag(id=11, name="Plot Twist", rank=94, is_media_spoiler=True),
                AnimeTag(id=12, name="Betrayal", rank=91, is_media_spoiler=True),
                AnimeTag(id=13, name="Tragedy", rank=93, is_media_spoiler=True),
            ],
        )
        vec = engine.build_feature_vector(anime)

        for spoiler in ["Plot Twist", "Betrayal", "Tragedy"]:
            idx = engine.term_index(spoiler)
            assert vec[idx] > 0.0, f"Spoiler tag {spoiler} has zero weight in internal VSM"

    def test_category_weighting_multipliers(self):
        vocab = ["CastTag", "TechTag"]
        cat_weights = {"Cast-Traits": 0.8, "Technical": 0.5}
        engine = VectorEngine(vocab, category_weights=cat_weights)
        anime = AnimeEntity(
            id=10,
            title_romaji="Test",
            cover_image="http://img.png",
            tags=[
                AnimeTag(id=1, name="CastTag", rank=100, category="Cast-Traits"),
                AnimeTag(id=2, name="TechTag", rank=100, category="Technical"),
            ],
        )
        vec = engine.build_feature_vector(anime)
        # Ratio of raw weights before normalization was 0.8 : 0.5
        assert np.isclose(vec[0] / vec[1], 0.8 / 0.5, atol=1e-5)

    def test_empty_genres_and_tags_produces_zero_vector(self):
        """Tier 2 boundary: anime with no metadata produces zero vector without NaN."""
        engine = VectorEngine()
        empty_anime = AnimeEntity(
            id=999,
            title_romaji="Empty",
            cover_image="http://img.png",
            genres=[],
            tags=[],
        )
        vec = engine.build_feature_vector(empty_anime)
        assert not np.isnan(vec).any()
        assert np.allclose(vec, 0.0, atol=1e-9)

    def test_unknown_and_hallucinated_tags_ignored(self):
        """R2.3 Defense: tags outside vocabulary do not corrupt vector dimensionality."""
        engine = VectorEngine(["Action", "Sci-Fi"])
        anime = AnimeEntity(
            id=20,
            title_romaji="Test",
            cover_image="http://img.png",
            genres=["FakeGenre123"],
            tags=[AnimeTag(id=999, name="CompletelyHallucinatedTag", rank=100)],
        )
        vec = engine.build_feature_vector(anime)
        assert len(vec) == 2
        assert np.allclose(vec, 0.0)

    def test_dict_calling_convention_interoperability(self):
        """Tests compatibility with raw dictionary payloads."""
        engine = VectorEngine(CANONICAL_FIXTURE_TAGS)
        raw_dict = {
            "genres": ["Action", "Sci-Fi"],
            "tags": [{"name": "Cyberpunk", "rank": 95}],
        }
        vec = engine.build_feature_vector(raw_dict)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6
        assert vec[engine.term_index("Cyberpunk")] > 0.0


class TestL2SphericalProjectionInvariant:
    """Tests L2 normalization stability and extreme value resistance."""

    def test_unit_norm_invariant_random_vectors(self):
        """R1.2: | ||v||_2 - 1.0 | < 10^-6 across diverse random vectors."""
        rng = np.random.default_rng(42)
        dim = 50
        for _ in range(100):
            raw = rng.uniform(-10.0, 10.0, size=dim)
            normed = normalize_vector(raw)
            norm = np.linalg.norm(normed)
            assert abs(norm - 1.0) < 1e-6

    def test_zero_vector_safe_fallback(self):
        zero = np.zeros(20, dtype=np.float64)
        normed = normalize_vector(zero)
        assert not np.isnan(normed).any()
        assert np.allclose(normed, 0.0, atol=1e-12)

    def test_sub_epsilon_vector_safe_fallback(self):
        tiny = np.full(10, 1e-15, dtype=np.float64)
        normed = normalize_vector(tiny)
        assert not np.isnan(normed).any()
        assert np.allclose(normed, 0.0, atol=1e-12)

    def test_extreme_magnitudes_without_overflow(self):
        huge = np.full(10, 1e20, dtype=np.float64)
        normed = normalize_vector(huge)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6

    def test_l2_normalize_alias(self):
        v = np.array([3.0, 4.0], dtype=np.float64)
        normed = l2_normalize(v)
        assert abs(np.linalg.norm(normed) - 1.0) < 1e-6


class TestCosineSimilarityKernel:
    """Tests pairwise and batch matrix cosine similarity."""

    def test_cosine_similarity_cardinal_properties(self):
        u = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_same = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        v_ortho = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        v_opp = np.array([-1.0, 0.0, 0.0], dtype=np.float64)

        assert np.isclose(cosine_similarity(u, v_same), 1.0, atol=1e-6)
        assert np.isclose(cosine_similarity(u, v_ortho), 0.0, atol=1e-6)
        assert np.isclose(cosine_similarity(u, v_opp), -1.0, atol=1e-6)

    def test_zero_vector_similarity_does_not_nan(self):
        """Tier 2.1: Sim(0, v) with zero user vector returns 0.0, not NaN."""
        zero = np.zeros(10, dtype=np.float64)
        item = normalize_vector(np.ones(10, dtype=np.float64))
        sim = cosine_similarity(zero, item)
        assert not math.isnan(sim)
        assert np.isclose(sim, 0.0, atol=1e-9)

    def test_batch_matrix_cosine_similarity(self):
        dim = 5
        u = normalize_vector(np.array([1.0, 1.0, 0.0, 0.0, 0.0]))
        items = np.array([
            normalize_vector(np.array([1.0, 1.0, 0.0, 0.0, 0.0])),  # sim ~ 1.0
            normalize_vector(np.array([0.0, 0.0, 1.0, 1.0, 0.0])),  # sim ~ 0.0
            normalize_vector(np.array([1.0, 0.0, 0.0, 0.0, 0.0])),  # sim ~ 0.707
        ])

        engine = VectorEngine(["t0", "t1", "t2", "t3", "t4"])
        scores = engine.compute_similarity(u, items)
        assert scores.shape == (3,)
        assert np.isclose(scores[0], 1.0, atol=1e-6)
        assert np.isclose(scores[1], 0.0, atol=1e-6)
        assert np.isclose(scores[2], 1.0 / np.sqrt(2), atol=1e-5)

    def test_score_candidates_batch_blas(self):
        u = normalize_vector(np.array([1.0, 0.0, 0.0]))
        matrix = np.array([
            normalize_vector(np.array([1.0, 0.0, 0.0])),
            normalize_vector(np.array([0.0, 1.0, 0.0])),
        ])
        scores = score_candidates(u, matrix)
        assert len(scores) == 2
        assert np.isclose(scores[0], 1.0, atol=1e-6)
        assert np.isclose(scores[1], 0.0, atol=1e-6)


class TestColdStartProfiles:
    """Tests isotropic and zero cold-start profile generation."""

    def test_isotropic_cold_start_unit_norm(self):
        dim = 20
        u0 = get_cold_start_vector(dimension=dim, strategy="isotropic")
        assert len(u0) == dim
        norm = np.linalg.norm(u0)
        assert abs(norm - 1.0) < 1e-6
        expected_val = 1.0 / np.sqrt(20)
        assert np.allclose(u0, expected_val, atol=1e-6)

    def test_zero_cold_start_strategy(self):
        u0 = get_cold_start_vector(dimension=15, strategy="zero")
        assert len(u0) == 15
        assert np.allclose(u0, 0.0)

    def test_encode_and_decode_user_profile(self):
        engine = VectorEngine(["Sci-Fi", "Action", "Cyberpunk"])
        profile = UserProfileData(
            user_id="test_user",
            total_ratings=1,
            feature_weights={"Sci-Fi": 0.8, "Cyberpunk": 0.6},
        )
        vec = engine.encode_user_profile(profile)
        assert abs(np.linalg.norm(vec) - 1.0) < 1e-6
        assert vec[0] > 0.0  # Sci-Fi
        assert vec[1] == 0.0  # Action
        assert vec[2] > 0.0  # Cyberpunk

        # Decode back to dictionary
        decoded = engine.decode_user_profile(vec)
        assert "Sci-Fi" in decoded
        assert "Cyberpunk" in decoded
        assert "Action" not in decoded
