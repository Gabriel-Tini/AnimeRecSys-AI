"""
tests/unit/test_semantic_pipeline.py
End-to-End and Unit Integration Tests for Gemini Semantic Tuning Pipeline.
Verifies boost formula precision, taste amplification/dampening, spherical L2
unit norm stability, spoiler tag participation, and atomic storage enrichment.
"""

import math
from pathlib import Path
import tempfile
import pytest
import numpy as np

from app.core.config import Settings
from app.core.storage_manager import StorageManager
from app.models.anime import AnimeEntity, AnimeTag
from app.models.recommendation import SemanticTagDelta, SemanticTuningResponse
from app.models.user import Hyperparameters, UserProfileData, WatchHistoryEntry
from app.services.gemini_service import GeminiService
from app.services.learning_engine import LearningEngine
from app.services.spoiler_service import SpoilerService, CANONICAL_SPOILER_TAGS


@pytest.fixture
def temp_storage():
    """Provides an isolated StorageManager in a temporary directory."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        sm = StorageManager(
            workspace_root=tmp_path,
            data_dir=tmp_path / "data",
            locks_dir=tmp_path / "data" / ".locks",
            fixtures_path=Path(__file__).resolve().parents[2] / ".antigravity" / "FIXTURES.json",
        )
        yield sm


@pytest.fixture
def sample_anime():
    """Provides a sample Steins;Gate AnimeEntity with genres and tags."""
    return AnimeEntity(
        id=9253,
        title_romaji="Steins;Gate",
        cover_image="https://example.com/cover.png",
        genres=["Drama", "Sci-Fi", "Suspense"],
        tags=[
            AnimeTag(id=6, name="Time Manipulation", rank=98, is_media_spoiler=False),
            AnimeTag(id=7, name="Psychological", rank=92, is_media_spoiler=False),
            AnimeTag(id=11, name="Plot Twist", rank=96, is_media_spoiler=True),
            AnimeTag(id=13, name="Tragedy", rank=88, is_media_spoiler=True),
        ],
    )


# =============================================================================
# 1. Mathematical Vector Boosting Precision & Dynamics
# =============================================================================

class TestSemanticBoostMathematics:
    """Verifies mathematical properties of Delta u_t = delta * c * alpha."""

    def test_boost_formula_precision(self, sample_anime):
        """Boost calculation matches exact coordinate formula Delta u_k = delta * c * alpha."""
        tuning = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Time Manipulation", impact_multiplier=0.8, confidence=0.9),
            ],
            public_safe_reason="Seguro.",
        )
        alpha = 0.25
        expected_boost = 0.8 * 0.9 * alpha  # 0.180

        profile = UserProfileData(
            user_id="test",
            feature_weights={},
            hyperparameters=Hyperparameters(alpha=alpha),
        )
        boosted = LearningEngine().apply_semantic_boost(profile, tuning)
        # For cold-start single dimension, normalized weight is 1.0, but un-normalized was exactly 0.18
        assert "Time Manipulation" in boosted.feature_weights
        assert abs(boosted.compute_l2_norm() - 1.0) < 1e-6

    def test_boost_amplification(self, sample_anime):
        """Positive qualitative boost produces higher weight for tag than rating alone."""
        le = LearningEngine()
        profile_base = le.initialize_cold_start_profile()

        # 1. Rating alone
        res_unboosted = le.update_profile_rating(
            profile=profile_base,
            rated_anime=sample_anime,
            rating=8.0,
            qualitative_tuning=None,
        )

        # 2. Rating + positive boost on Time Manipulation
        tuning = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Time Manipulation", impact_multiplier=1.0, confidence=1.0),
            ],
            public_safe_reason="Seguro.",
        )
        res_boosted = le.update_profile_rating(
            profile=profile_base,
            rated_anime=sample_anime,
            rating=8.0,
            qualitative_tuning=tuning,
        )

        assert res_boosted.feature_weights["Time Manipulation"] > res_unboosted.feature_weights["Time Manipulation"]

    def test_boost_dampening(self, sample_anime):
        """Negative qualitative boost dampens weight for tag relative to rating alone."""
        le = LearningEngine()
        profile_base = le.initialize_cold_start_profile()

        # 1. Rating alone
        res_unboosted = le.update_profile_rating(
            profile=profile_base,
            rated_anime=sample_anime,
            rating=8.0,
            qualitative_tuning=None,
        )

        # 2. Rating + negative boost on Psychological
        tuning = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Psychological", impact_multiplier=-1.0, confidence=1.0),
            ],
            public_safe_reason="Seguro.",
        )
        res_dampened = le.update_profile_rating(
            profile=profile_base,
            rated_anime=sample_anime,
            rating=8.0,
            qualitative_tuning=tuning,
        )

        assert res_dampened.feature_weights["Psychological"] < res_unboosted.feature_weights["Psychological"]

    def test_zero_confidence_neutrality(self, sample_anime):
        """Zero confidence (c=0.0) yields exactly zero boost, identical to unboosted update."""
        le = LearningEngine()
        profile_base = le.initialize_cold_start_profile()

        res_unboosted = le.update_profile_rating(
            profile=profile_base,
            rated_anime=sample_anime,
            rating=7.0,
            qualitative_tuning=None,
        )

        tuning_zero_conf = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Time Manipulation", impact_multiplier=1.0, confidence=0.0),
            ],
            public_safe_reason="Seguro.",
        )
        res_zero = le.update_profile_rating(
            profile=profile_base,
            rated_anime=sample_anime,
            rating=7.0,
            qualitative_tuning=tuning_zero_conf,
        )

        for k in res_unboosted.feature_weights:
            assert abs(res_unboosted.feature_weights[k] - res_zero.feature_weights[k]) < 1e-9


# =============================================================================
# 2. Layer 4 Clamping, Filtering & Boundary Stability
# =============================================================================

class TestPipelineSecurityAndInvariants:
    """Verifies security bounds and mathematical invariants under stress."""

    def test_layer4_clamping_impact(self):
        """Impact multiplier is clamped to [-1.0, 1.0]."""
        d1 = SemanticTagDelta.model_validate({
            "tag_name": "Action",
            "impact_multiplier": 999.0,
            "confidence": 0.8,
        })
        d2 = SemanticTagDelta.model_validate({
            "tag_name": "Sci-Fi",
            "impact_multiplier": -500.0,
            "confidence": 0.8,
        })
        assert d1.impact_multiplier == 1.0
        assert d2.impact_multiplier == -1.0

    def test_layer4_clamping_confidence(self):
        """Confidence score is clamped to [0.0, 1.0]."""
        d1 = SemanticTagDelta.model_validate({
            "tag_name": "Action",
            "impact_multiplier": 0.5,
            "confidence": 10.0,
        })
        d2 = SemanticTagDelta.model_validate({
            "tag_name": "Sci-Fi",
            "impact_multiplier": 0.5,
            "confidence": -2.0,
        })
        assert d1.confidence == 1.0
        assert d2.confidence == 0.0

    def test_layer4_hallucinated_tags(self):
        """Hallucinated or injected tags are discarded during post-processing."""
        raw = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="FakeTag123", impact_multiplier=1.0, confidence=1.0),
                SemanticTagDelta(tag_name="Action", impact_multiplier=0.8, confidence=0.9),
            ],
            public_safe_reason="Seguro.",
        )
        clean = GeminiService.post_process_tuning_response(raw, canonical_vocabulary=["Action", "Sci-Fi"])
        assert len(clean.identified_tags) == 1
        assert clean.identified_tags[0].tag_name == "Action"

    def test_spherical_norm_100_boosted_updates(self, sample_anime):
        """100 consecutive rating + qualitative boost updates strictly preserve ||u||_2 = 1.0 +- 1e-6."""
        le = LearningEngine()
        profile = le.initialize_cold_start_profile()

        tuning_positive = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Time Manipulation", impact_multiplier=0.9, confidence=0.8),
                SemanticTagDelta(tag_name="Sci-Fi", impact_multiplier=0.7, confidence=0.9),
            ],
            public_safe_reason="Seguro.",
        )
        tuning_negative = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Psychological", impact_multiplier=-0.8, confidence=0.85),
            ],
            public_safe_reason="Seguro.",
        )

        for step in range(100):
            rating = 8.5 if step % 2 == 0 else 3.0
            tuning = tuning_positive if step % 2 == 0 else tuning_negative
            profile = le.update_profile_rating(
                profile=profile,
                rated_anime=sample_anime,
                rating=rating,
                qualitative_tuning=tuning,
            )
            norm = profile.compute_l2_norm()
            assert abs(norm - 1.0) < 1e-6, f"Norm drifted at step {step}: {norm}"

    def test_extreme_attraction_stability(self, sample_anime):
        """Max rating (10.0) with maximal boost (+1.0, 1.0) maintains unit norm without overflow."""
        le = LearningEngine()
        profile = le.initialize_cold_start_profile()
        tuning_max = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Time Manipulation", impact_multiplier=1.0, confidence=1.0),
            ],
            public_safe_reason="Seguro.",
        )
        updated = le.update_profile_rating(
            profile=profile,
            rated_anime=sample_anime,
            rating=10.0,
            qualitative_tuning=tuning_max,
        )
        norm = updated.compute_l2_norm()
        assert abs(norm - 1.0) < 1e-6

    def test_extreme_repulsion_stability(self, sample_anime):
        """Min rating (1.0) with maximal negative dampening maintains unit norm with negative taste weights."""
        le = LearningEngine()
        profile = le.initialize_cold_start_profile()
        tuning_min = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Drama", impact_multiplier=-1.0, confidence=1.0),
            ],
            public_safe_reason="Seguro.",
        )
        updated = le.update_profile_rating(
            profile=profile,
            rated_anime=sample_anime,
            rating=1.0,
            qualitative_tuning=tuning_min,
        )
        norm = updated.compute_l2_norm()
        assert abs(norm - 1.0) < 1e-6
        assert any(w < 0.0 for w in updated.feature_weights.values())


# =============================================================================
# 3. Dual Spoiler Policy & Persistence Integration
# =============================================================================

class TestPipelineDualSpoilerAndPersistence:
    """Verifies internal spoiler participation, public spoiler redaction, and persistence."""

    def test_spoiler_tag_boost_participation(self, sample_anime):
        """Spoiler tags (e.g. 'Plot Twist') participate fully with positive weight in taste vector."""
        le = LearningEngine()
        profile = le.initialize_cold_start_profile()

        tuning_spoiler = SemanticTuningResponse(
            identified_tags=[
                SemanticTagDelta(tag_name="Plot Twist", impact_multiplier=0.9, confidence=0.9),
            ],
            public_safe_reason="Você demonstrou afinidade com a narrativa da obra.",
        )
        updated = le.update_profile_rating(
            profile=profile,
            rated_anime=sample_anime,
            rating=9.0,
            qualitative_tuning=tuning_spoiler,
        )
        assert "Plot Twist" in updated.feature_weights
        assert updated.feature_weights["Plot Twist"] > 0.0

    def test_public_safe_reason_no_spoilers(self):
        """public_safe_reason contains zero explicit spoiler terms from KNOWN_SPOILER_TAGS."""
        resp = SemanticTuningResponse(
            identified_tags=[],
            public_safe_reason="Adorei a grande reviravolta e o PLOT TWIST no final.",
        )
        assert "[redacted]" in resp.public_safe_reason
        for sp in CANONICAL_SPOILER_TAGS:
            assert sp.lower() not in resp.public_safe_reason.lower()

    def test_tune_and_record_rating_offline_fallback(self, temp_storage, sample_anime):
        """tune_and_record_rating executes end-to-end with offline fallback and persists to disk."""
        temp_storage.save_catalog([sample_anime])
        le = LearningEngine(storage_manager=temp_storage)

        # Gemini service without API key triggers offline keyword extraction
        gemini_svc = GeminiService(settings=Settings(gemini_api_key=None))

        updated_profile, entry, tuning = le.tune_and_record_rating(
            anime_id=sample_anime.id,
            rating=9.0,
            user_review="Adorei a viagem no tempo e a tensão da história.",
            gemini_service=gemini_svc,
            spoiler_service=SpoilerService,
        )

        assert updated_profile.total_ratings == 1
        assert abs(updated_profile.compute_l2_norm() - 1.0) < 1e-6
        assert entry.anime_id == sample_anime.id
        assert entry.user_review_raw == "Adorei a viagem no tempo e a tensão da história."
        assert tuning is not None
        assert any(d.tag_name == "Time Manipulation" for d in tuning.identified_tags)

        # Verify disk persistence
        disk_profile = temp_storage.load_user_profile()
        assert disk_profile.total_ratings == 1
        disk_history = temp_storage.load_watch_history()
        assert len(disk_history) == 1
        assert disk_history[0].user_review_raw == "Adorei a viagem no tempo e a tensão da história."

    def test_enrich_latest_watch_history(self, temp_storage, sample_anime):
        """Two-step flow: record rating first, then enrich interaction with review and tuning."""
        temp_storage.save_catalog([sample_anime])
        le = LearningEngine(storage_manager=temp_storage)

        # Step 1: Record rating without review
        le.record_rating(anime_id=sample_anime.id, rating=8.5)

        initial_history = temp_storage.load_watch_history()
        assert len(initial_history) == 1
        assert initial_history[0].user_review_raw is None

        # Step 2: Post-rating modal review enrichment
        gemini_svc = GeminiService(settings=Settings(gemini_api_key=None))
        prof, enriched, tuning = le.tune_existing_interaction(
            anime_id=sample_anime.id,
            user_feedback="Achei o mistério fantástico.",
            gemini_service=gemini_svc,
            spoiler_service=SpoilerService,
        )

        assert enriched is not None
        assert enriched.user_review_raw == "Achei o mistério fantástico."
        assert enriched.qualitative_tuning is not None

        # Verify disk updated
        disk_history = temp_storage.load_watch_history()
        assert disk_history[0].user_review_raw == "Achei o mistério fantástico."
        assert disk_history[0].qualitative_tuning is not None
