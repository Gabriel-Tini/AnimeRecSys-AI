"""
tests/unit/test_models.py
Comprehensive unit test suite for Pydantic v2 domain models:
- Anime metadata, tag parsing, and strict spoiler shielding on AnimePublicResponse (R3).
- User profile, hyperparameters, L2 norm calculations, and watch history.
- Recommendation responses with disjoint-track invariant enforcement (R1).
- Layer 4 clamping and sanitization on SemanticTagDelta / SemanticTuningResponse (R2, R6).
- Full compatibility with .antigravity/FIXTURES.json.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.models.anime import (
    KNOWN_SPOILER_TAGS,
    AnimeCatalogResponse,
    AnimeEntity,
    AnimePublicResponse,
    AnimeTag,
    AnimeTitle,
)
from app.models.recommendation import (
    RateRequest,
    RecommendationResponse,
    SemanticTagDelta,
    SemanticTuningResponse,
    TuneRequest,
)
from app.models.user import (
    Hyperparameters,
    UserProfileData,
    WatchHistoryEntry,
)


# -----------------------------------------------------------------------------
# 1. AnimeTag & AnimeTitle Tests
# -----------------------------------------------------------------------------
def test_anime_tag_validation():
    """Verify AnimeTag field constraints and alias handling."""
    tag1 = AnimeTag(id=1, name="Action", rank=95, is_media_spoiler=False)
    assert tag1.id == 1
    assert tag1.name == "Action"
    assert tag1.rank == 95
    assert tag1.is_media_spoiler is False

    # Alias isMediaSpoiler
    tag2 = AnimeTag(id=11, name="Plot Twist", rank=90, isMediaSpoiler=True)
    assert tag2.is_media_spoiler is True

    # Rank bounds [0, 100]
    with pytest.raises(ValidationError):
        AnimeTag(id=2, name="BadRank", rank=101)
    with pytest.raises(ValidationError):
        AnimeTag(id=2, name="BadRank", rank=-1)


def test_anime_title_display():
    """Verify display_title falls back to romaji when english is absent."""
    t1 = AnimeTitle(romaji="Sousou no Frieren", english="Frieren: Beyond Journey's End")
    assert t1.display_title == "Frieren: Beyond Journey's End"

    t2 = AnimeTitle(romaji="Sousou no Frieren", english=None)
    assert t2.display_title == "Sousou no Frieren"


# -----------------------------------------------------------------------------
# 2. AnimeEntity & AniList Payload Ingestion
# -----------------------------------------------------------------------------
def test_anime_entity_from_flat_fixture():
    """Verify AnimeEntity parsing from flat fixture format."""
    data = {
        "id": 9253,
        "title_romaji": "Steins;Gate",
        "title_english": "Steins;Gate",
        "cover_image": "https://example.com/steins.png",
        "genres": ["Drama", "Sci-Fi"],
        "tags": [
            {"id": 6, "name": "Time Manipulation", "rank": 98, "is_media_spoiler": False},
            {"id": 11, "name": "Plot Twist", "rank": 96, "is_media_spoiler": True},
        ],
    }
    entity = AnimeEntity.model_validate(data)
    assert entity.id == 9253
    assert entity.title_romaji == "Steins;Gate"
    assert entity.title is not None
    assert entity.title.romaji == "Steins;Gate"
    assert len(entity.tags) == 2
    # Preserves spoiler tags internally
    assert any(t.is_media_spoiler for t in entity.tags)


def test_anime_entity_from_nested_anilist_payload():
    """Verify AnimeEntity parsing from raw AniList GraphQL response with nested objects."""
    data = {
        "id": 16498,
        "title": {
            "romaji": "Shingeki no Kyojin",
            "english": "Attack on Titan",
            "native": "進撃の巨人",
        },
        "coverImage": {
            "extraLarge": "https://example.com/aot_xl.png",
            "large": "https://example.com/aot_l.png",
        },
        "bannerImage": "https://example.com/aot_banner.png",
        "format": "TV",
        "seasonYear": 2013,
        "averageScore": 85,
        "popularity": 540200,
        "genres": ["Action", "Drama"],
        "tags": [
            {"id": 1, "name": "Action", "rank": 95, "isMediaSpoiler": False},
            {"id": 12, "name": "Betrayal", "rank": 91, "isMediaSpoiler": True},
        ],
    }
    entity = AnimeEntity.model_validate(data)
    assert entity.id == 16498
    assert entity.title_romaji == "Shingeki no Kyojin"
    assert entity.title_english == "Attack on Titan"
    assert entity.cover_image == "https://example.com/aot_xl.png"
    assert entity.banner_image == "https://example.com/aot_banner.png"
    assert entity.season_year == 2013
    assert entity.average_score == 85


# -----------------------------------------------------------------------------
# 3. R3 Dual Spoiler Shielding on AnimePublicResponse
# -----------------------------------------------------------------------------
def test_anime_public_response_spoiler_filtering():
    """
    R3 Acceptance Criteria:
    Zero presence of is_media_spoiler: true or KNOWN_SPOILER_TAGS in public response.
    """
    entity = AnimeEntity(
        id=9253,
        title_romaji="Steins;Gate",
        cover_image="https://example.com/steins.png",
        genres=["Drama", "Sci-Fi"],
        tags=[
            AnimeTag(id=6, name="Time Manipulation", rank=98, is_media_spoiler=False),
            AnimeTag(id=7, name="Psychological", rank=92, is_media_spoiler=False),
            AnimeTag(id=11, name="Plot Twist", rank=96, is_media_spoiler=True),     # Spoiler!
            AnimeTag(id=13, name="Tragedy", rank=88, is_media_spoiler=True),        # Spoiler!
            AnimeTag(id=15, name="Conspiracy", rank=85, is_media_spoiler=True),     # Spoiler!
        ],
    )

    public_resp = entity.to_public_response(compatibility_score=0.942, origin_track="affinity")

    # Tags must strictly contain ONLY safe tags
    assert "Time Manipulation" in public_resp.tags
    assert "Psychological" in public_resp.tags
    assert "Plot Twist" not in public_resp.tags
    assert "Tragedy" not in public_resp.tags
    assert "Conspiracy" not in public_resp.tags

    for tag in public_resp.tags:
        assert tag not in KNOWN_SPOILER_TAGS

    # Match percentage calculated accurately
    assert public_resp.match_percentage == 94
    assert public_resp.origin_track == "affinity"


def test_public_response_direct_dict_spoiler_stripping():
    """Verify tags validator sanitizes list of dicts, strings, or tags."""
    resp = AnimePublicResponse(
        id=10,
        title_romaji="Test Anime",
        cover_image="https://example.com/test.png",
        tags=[
            {"name": "Action", "is_media_spoiler": False},
            {"name": "Plot Twist", "is_media_spoiler": True},
            {"name": "Tragedy", "isMediaSpoiler": False},  # Name in KNOWN_SPOILER_TAGS
            "Betrayal",  # String in KNOWN_SPOILER_TAGS
            "Comedy",    # Safe string
        ],
    )
    assert resp.tags == ["Action", "Comedy"]


@pytest.mark.parametrize("casing_variant", [
    "plot twist", "PLOT TWIST", "Plot Twist", "pLoT tWiSt",
    "tragedy", "TRAGEDY", "Tragedy", "tRaGeDy",
    "betrayal", "BETRAYAL", "Betrayal",
    "conspiracy", "CONSPIRACY",
    "memory manipulation", "MEMORY MANIPULATION",
])
def test_public_response_strips_case_variant_spoilers(casing_variant: str):
    """Verify that AnimePublicResponse suppresses spoiler tags regardless of casing."""
    resp = AnimePublicResponse(
        id=1,
        title_romaji="Attack on Titan",
        cover_image="https://example.com/cover.jpg",
        tags=[casing_variant, "Action", "Drama"],
    )
    lowered_tags = [t.lower() for t in resp.tags]
    assert casing_variant.strip().lower() not in lowered_tags
    assert resp.tags == ["Action", "Drama"]


@pytest.mark.parametrize("whitespace_variant", [
    " Plot Twist", "Plot Twist ", "   Plot Twist   ", "\tPlot Twist\n",
    "  Tragedy  ", " Betrayal ",
])
def test_public_response_strips_whitespace_padded_spoilers(whitespace_variant: str):
    """Verify that leading/trailing whitespace variations do not bypass spoiler filter."""
    resp = AnimePublicResponse(
        id=2,
        title_romaji="Madoka Magica",
        cover_image="https://example.com/cover.jpg",
        tags=[whitespace_variant, "Magic", "Psychological"],
    )
    lowered_stripped = [t.strip().lower() for t in resp.tags]
    assert whitespace_variant.strip().lower() not in lowered_stripped
    assert resp.tags == ["Magic", "Psychological"]


def test_public_response_strips_dict_spoiler_variants():
    """Verify dict tags with is_media_spoiler: False but spoiler names are suppressed."""
    resp = AnimePublicResponse(
        id=3,
        title_romaji="Steins;Gate",
        cover_image="https://example.com/cover.jpg",
        tags=[
            {"name": "plot twist", "is_media_spoiler": False},
            {"name": "  TRAGEDY  ", "isMediaSpoiler": False},
            {"name": "Sci-Fi", "is_media_spoiler": False},
            {"name": "  Action  ", "is_media_spoiler": False},
        ],
    )
    assert "plot twist" not in [t.lower() for t in resp.tags]
    assert "tragedy" not in [t.lower() for t in resp.tags]
    assert resp.tags == ["Sci-Fi", "Action"]


def test_anime_entity_to_public_response_casing_variants():
    """Verify AnimeEntity.to_public_response() strips casing variants in AnimeTag instances."""
    entity = AnimeEntity(
        id=4,
        title_romaji="Berserk",
        cover_image="https://example.com/cover.jpg",
        tags=[
            AnimeTag(id=1, name="betrayal", rank=90, is_media_spoiler=False),
            AnimeTag(id=2, name="TRAGEDY", rank=95, is_media_spoiler=False),
            AnimeTag(id=3, name="Dark Fantasy", rank=99, is_media_spoiler=False),
        ],
    )
    pub = entity.to_public_response()
    lowered = [t.lower() for t in pub.tags]
    assert "betrayal" not in lowered
    assert "tragedy" not in lowered
    assert pub.tags == ["Dark Fantasy"]


def test_public_response_trims_whitespace_on_safe_tags():
    """Verify that safe non-spoiler tags have leading/trailing whitespace trimmed."""
    resp = AnimePublicResponse(
        id=5,
        title_romaji="Frieren",
        cover_image="https://example.com/cover.jpg",
        tags=["  Fantasy  ", "\tAdventure\n", "Slice of Life"],
    )
    assert resp.tags == ["Fantasy", "Adventure", "Slice of Life"]


# -----------------------------------------------------------------------------
# 4. UserProfileData & Hyperparameters Tests
# -----------------------------------------------------------------------------
def test_user_profile_defaults_and_cold_start():
    """Verify UserProfileData defaults and cold start detection."""
    profile = UserProfileData()
    assert profile.user_id == "default_user"
    assert profile.total_ratings == 0
    assert profile.feature_weights == {}
    assert profile.is_cold_start() is True
    assert profile.compute_l2_norm() == 0.0

    # Hyperparameters defaults
    assert profile.hyperparameters.eta == 0.15
    assert profile.hyperparameters.alpha == 0.25
    assert profile.hyperparameters.gamma == 0.98
    assert profile.hyperparameters.lambda_mmr == 0.50


def test_user_profile_l2_norm_computation():
    """Verify L2 norm computation of feature weights."""
    profile = UserProfileData(
        total_ratings=3,
        feature_weights={"Sci-Fi": 3.0, "Drama": 4.0},
    )
    assert profile.is_cold_start() is False
    # norm = sqrt(3^2 + 4^2) = 5.0
    assert pytest.approx(profile.compute_l2_norm(), abs=1e-6) == 5.0


# -----------------------------------------------------------------------------
# 5. WatchHistoryEntry Tests
# -----------------------------------------------------------------------------
def test_watch_history_validation():
    """Verify WatchHistoryEntry bounds and length restrictions."""
    entry = WatchHistoryEntry(
        anime_id=9253,
        anime_title="Steins;Gate",
        rating=9.5,
        origin_track="affinity",
        user_review_raw="Masterpiece of time travel fiction.",
    )
    assert entry.id.startswith("wh-")
    assert entry.rating == 9.5
    assert entry.origin_track == "affinity"

    # Rating bounds [1.0, 10.0]
    with pytest.raises(ValidationError):
        WatchHistoryEntry(anime_id=1, anime_title="A", rating=0.5, origin_track="affinity")
    with pytest.raises(ValidationError):
        WatchHistoryEntry(anime_id=1, anime_title="A", rating=10.5, origin_track="affinity")

    # Review length capped at 300 characters
    long_review = "x" * 301
    with pytest.raises(ValidationError):
        WatchHistoryEntry(
            anime_id=1,
            anime_title="A",
            rating=8.0,
            origin_track="affinity",
            user_review_raw=long_review,
        )


# -----------------------------------------------------------------------------
# 6. SemanticTagDelta & Clamping Defense (R2, R6)
# -----------------------------------------------------------------------------
def test_semantic_tag_delta_clamping():
    """
    R6 Defense: Verify that prompt injection attempts or numerical drift
    are deterministically clamped to [-1.0, 1.0] and [0.0, 1.0].
    """
    # Over-inflated positive impact
    d1 = SemanticTagDelta(tag_name="Cyberpunk", impact_multiplier=5.0, confidence=1.8)
    assert d1.impact_multiplier == 1.0
    assert d1.confidence == 1.0

    # Over-inflated negative impact
    d2 = SemanticTagDelta(tag_name="Romance", impact_multiplier=-10.0, confidence=-0.5)
    assert d2.impact_multiplier == -1.0
    assert d2.confidence == 0.0

    # String numeric inputs
    d3 = SemanticTagDelta(tag_name="Sci-Fi", impact_multiplier="0.85", confidence="0.95")
    assert d3.impact_multiplier == 0.85
    assert d3.confidence == 0.95

    # Malformed non-numeric values
    d4 = SemanticTagDelta(tag_name="Mystery", impact_multiplier="invalid", confidence="bad")
    assert d4.impact_multiplier == 0.0
    assert d4.confidence == 0.0


def test_semantic_tuning_response_spoiler_audit():
    """Verify that public_safe_reason redacts any accidental spoiler term."""
    tuning = SemanticTuningResponse(
        identified_tags=[
            SemanticTagDelta(tag_name="Time Manipulation", impact_multiplier=0.8, confidence=0.9)
        ],
        public_safe_reason="The anime features an incredible Plot Twist and Tragedy in episode 12.",
    )
    # Plot Twist and Tragedy must be redacted
    assert "Plot Twist" not in tuning.public_safe_reason
    assert "Tragedy" not in tuning.public_safe_reason
    assert "[redacted]" in tuning.public_safe_reason


def test_semantic_tag_delta_nan_and_inf_handling():
    """
    R6 Defense: Verify that NaN and invalid non-finite values are neutralized
    to safe 0.0, preventing unintentional vector boost exploits.
    """
    # 1. IEEE-754 float("nan")
    d_nan = SemanticTagDelta(
        tag_name="Action",
        impact_multiplier=float("nan"),
        confidence=float("nan"),
    )
    assert d_nan.impact_multiplier == 0.0
    assert d_nan.confidence == 0.0

    # 2. String representation "nan" / "NaN"
    d_str_nan = SemanticTagDelta(
        tag_name="Action",
        impact_multiplier="nan",
        confidence="NaN",
    )
    assert d_str_nan.impact_multiplier == 0.0
    assert d_str_nan.confidence == 0.0

    # 3. None and empty inputs fallback to 0.0
    d_none = SemanticTagDelta(
        tag_name="Drama",
        impact_multiplier=None,
        confidence=None,
    )
    assert d_none.impact_multiplier == 0.0
    assert d_none.confidence == 0.0

    # 4. Extreme values clamping
    d_extreme = SemanticTagDelta(
        tag_name="Sci-Fi",
        impact_multiplier=999.0,
        confidence=5.0,
    )
    assert d_extreme.impact_multiplier == 1.0
    assert d_extreme.confidence == 1.0

    d_neg_extreme = SemanticTagDelta(
        tag_name="Sci-Fi",
        impact_multiplier=-999.0,
        confidence=-5.0,
    )
    assert d_neg_extreme.impact_multiplier == -1.0
    assert d_neg_extreme.confidence == 0.0


@pytest.mark.parametrize("input_reason,expected_redacted_count", [
    ("The major PLOT TWIST changed everything.", 1),
    ("A terrible TRAGEDY and painful BETRAYAL occurred.", 2),
    ("Secret CONSPIRACY and MEMORY MANIPULATION were discovered.", 2),
    ("Mixed case: pLoT tWiSt with tRaGeDy.", 2),
    ("Already safe explanation without any spoilers.", 0),
    ("Plot Twist, plot twist, and PLOT TWIST in the same show.", 3),
])
def test_semantic_tuning_response_spoiler_audit_case_variations(input_reason, expected_redacted_count):
    """
    R3/R6 Acceptance Criteria:
    Verify that public_safe_reason replaces all casing permutations of KNOWN_SPOILER_TAGS
    with '[redacted]' and leaks zero known spoiler terms.
    """
    tuning = SemanticTuningResponse(
        identified_tags=[],
        public_safe_reason=input_reason,
    )
    assert tuning.public_safe_reason.count("[redacted]") == expected_redacted_count

    # Verify none of the known spoiler tags appear in any casing
    lowered = tuning.public_safe_reason.lower()
    for spoiler in KNOWN_SPOILER_TAGS:
        assert spoiler.lower() not in lowered


# -----------------------------------------------------------------------------
# 7. RecommendationResponse & Disjoint-Track Invariant (R1)
# -----------------------------------------------------------------------------
def test_recommendation_response_disjoint_invariant_success():
    """Verify valid RecommendationResponse when tracks are strictly disjoint."""
    item1 = AnimePublicResponse(
        id=1, title_romaji="Anime 1", cover_image="https://example.com/1.png"
    )
    item2 = AnimePublicResponse(
        id=2, title_romaji="Anime 2", cover_image="https://example.com/2.png"
    )
    item3 = AnimePublicResponse(
        id=3, title_romaji="Anime 3", cover_image="https://example.com/3.png"
    )

    resp = RecommendationResponse(
        affinity_track=[item1, item2],
        discovery_track=[item3],
    )
    assert len(resp.affinity_track) == 2
    assert len(resp.discovery_track) == 1


def test_recommendation_response_disjoint_invariant_violation():
    """
    R1 Acceptance Criteria:
    S_aff ∩ S_disc must be empty. Overlap raises ValueError.
    """
    item1 = AnimePublicResponse(
        id=1, title_romaji="Anime 1", cover_image="https://example.com/1.png"
    )
    item2 = AnimePublicResponse(
        id=2, title_romaji="Anime 2", cover_image="https://example.com/2.png"
    )

    with pytest.raises(ValidationError) as exc_info:
        RecommendationResponse(
            affinity_track=[item1, item2],
            discovery_track=[item2],  # Duplicate ID 2 in both tracks!
        )
    assert "Disjoint-track invariant violated" in str(exc_info.value)


# -----------------------------------------------------------------------------
# 8. API Request Models: RateRequest & TuneRequest
# -----------------------------------------------------------------------------
def test_rate_request_alias_and_bounds():
    """Verify RateRequest aliases (rating -> score) and bounds."""
    r1 = RateRequest(anime_id=10, rating=8.5, origin_track="affinity")
    assert r1.score == 8.5
    assert r1.anime_id == 10
    assert r1.origin_track == "affinity"

    r2 = RateRequest(anime_id=10, score=7.0, qualitative_feedback="Loved it!")
    assert r2.score == 7.0
    assert r2.user_review == "Loved it!"

    with pytest.raises(ValidationError):
        RateRequest(anime_id=10, score=11.0)


def test_tune_request_validation():
    """Verify TuneRequest aliases and length cap."""
    t1 = TuneRequest(anime_id=20, feedback="More sci-fi, less romance please")
    assert t1.user_feedback == "More sci-fi, less romance please"

    with pytest.raises(ValidationError):
        TuneRequest(anime_id=20, feedback="a" * 301)


# -----------------------------------------------------------------------------
# 9. FIXTURES.json Cross-Verification
# -----------------------------------------------------------------------------
def test_fixtures_deserialization():
    """Verify that all entities in .antigravity/FIXTURES.json deserialize cleanly."""
    fixtures_file = Path(__file__).resolve().parents[2] / ".antigravity" / "FIXTURES.json"
    assert fixtures_file.exists(), f"FIXTURES.json missing at {fixtures_file}"

    fixtures = json.loads(fixtures_file.read_text(encoding="utf-8"))

    # 1. tags_vocabulary
    tags = [AnimeTag.model_validate(t) for t in fixtures["tags_vocabulary"]]
    assert len(tags) == 20

    # 2. sample_animes_full
    animes = [AnimeEntity.model_validate(a) for a in fixtures["sample_animes_full"]]
    assert len(animes) >= 6

    # 3. mock_user_profile
    profile = UserProfileData.model_validate(fixtures["mock_user_profile"])
    assert profile.user_id == "default_user"
    assert profile.total_ratings == 3
    assert len(profile.feature_weights) > 0

    # 4. mock_watch_history
    history = [WatchHistoryEntry.model_validate(h) for h in fixtures["mock_watch_history"]]
    assert len(history) == 2

    # 5. mock_gemini_response
    gemini_resp = SemanticTuningResponse.model_validate(fixtures["mock_gemini_response"])
    assert len(gemini_resp.identified_tags) > 0

    # 6. mock_recommendation_response
    rec_resp = RecommendationResponse.model_validate(fixtures["mock_recommendation_response"])
    assert len(rec_resp.affinity_track) > 0
    assert len(rec_resp.discovery_track) > 0
