"""
tests/unit/test_spoiler_service.py
Comprehensive unit test suite for backend/app/services/spoiler_service.py.
Verifies R3 Dual Spoiler Policy, case/whitespace/punctuation normalization,
boundary word detection, adversarial bypass evasion, and public DTO transformations.
"""

import time
import pytest
from typing import Dict, List

from app.models.anime import AnimeEntity, AnimePublicResponse, AnimeTag, AnimeTitle
from app.models.recommendation import RecommendationResponse
from app.services.spoiler_service import (
    CANONICAL_SPOILER_TAGS,
    SAFE_THEMATIC_DESCRIPTORS,
    SpoilerService,
    filter_safe_tags,
    has_spoiler,
    is_spoiler_tag,
    sanitize_anime_catalog,
    sanitize_reason,
    sanitize_text,
    to_public_response,
)


# =============================================================================
# 1. Canonical Registry & Tag Normalization Tests
# =============================================================================

class TestTagNormalizationAndRegistry:
    """Verifies tag detection across casing, whitespace, and punctuation variations."""

    @pytest.mark.parametrize("canonical_tag", [
        "Plot Twist", "Betrayal", "Tragedy", "Conspiracy", "Memory Manipulation"
    ])
    def test_canonical_tags_detection(self, canonical_tag: str):
        """All 5 AniList canonical spoiler tags are identified."""
        assert is_spoiler_tag(canonical_tag) is True
        assert is_spoiler_tag(AnimeTag(id=1, name=canonical_tag)) is True
        assert is_spoiler_tag({"name": canonical_tag, "is_media_spoiler": False}) is True

    def test_dynamic_boolean_flag_overrides_name(self):
        """A tag with is_media_spoiler=True is a spoiler regardless of name."""
        tag1 = AnimeTag(id=101, name="CustomSecret", is_media_spoiler=True)
        assert is_spoiler_tag(tag1) is True

        tag2 = {"name": "UnusualPlotPoint", "isMediaSpoiler": True}
        assert is_spoiler_tag(tag2) is True

    @pytest.mark.parametrize("casing_variant", [
        "plot twist", "PLOT TWIST", "pLoT tWiSt",
        "tragedy", "TRAGEDY", "TrAgEdY",
        "betrayal", "BETRAYAL", "BeTrAyAl",
        "conspiracy", "CONSPIRACY",
        "memory manipulation", "MEMORY MANIPULATION",
    ])
    def test_casing_variations(self, casing_variant: str):
        """Case folding correctly identifies spoiler tags."""
        assert is_spoiler_tag(casing_variant) is True

    @pytest.mark.parametrize("whitespace_variant", [
        "  Plot Twist  ", "\tPlot Twist\n", "   Plot   Twist   ",
        "  Tragedy  ", "\nBetrayal\t",
    ])
    def test_whitespace_variations(self, whitespace_variant: str):
        """Leading, trailing, and redundant internal whitespace do not bypass filter."""
        assert is_spoiler_tag(whitespace_variant) is True

    @pytest.mark.parametrize("punctuation_variant", [
        "plot-twist", "plot_twist", "Plot: Twist", "Plot/Twist",
        "P.l.o.t T.w.i.s.t", "p-l-o-t-t-w-i-s-t", "P.L.O.T  T.W.I.S.T",
        "memory-manipulation", "memory_manipulation", "M.e.m.o.r.y M.a.n.i.p.u.l.a.t.i.o.n",
        "T.r.a.g.e.d.y", "B.e.t.r.a.y.a.l", "C.o.n.s.p.i.r.a.c.y",
    ])
    def test_punctuation_and_adversarial_tag_variants(self, punctuation_variant: str):
        """Punctuation-padded and dotted variations are recognized as spoilers."""
        assert is_spoiler_tag(punctuation_variant) is True

    @pytest.mark.parametrize("safe_tag", [
        "Action", "Adventure", "Fantasy", "Sci-Fi", "Cyberpunk",
        "Time Manipulation", "Psychological", "Mystery", "Supernatural",
        "Shounen", "Anti-Hero", "Philosophy", "Coming of Age",
        "Slice of Life", "Music", "Drama", "Comedy", "Romance",
    ])
    def test_safe_tags_false_positive_immunity(self, safe_tag: str):
        """Standard AniList genres and safe tags NEVER trigger false-positive spoiler detection."""
        assert is_spoiler_tag(safe_tag) is False
        assert is_spoiler_tag(AnimeTag(id=2, name=safe_tag, is_media_spoiler=False)) is False
        assert is_spoiler_tag({"name": safe_tag, "is_media_spoiler": False}) is False


# =============================================================================
# 2. Text Scanning & Boundary Detection Tests
# =============================================================================

class TestTextScanningAndBoundaryWords:
    """Verifies NLP text scanning, word boundaries, and anti-overredaction safeguards."""

    def test_has_spoiler_positive(self):
        """Identifies spoilers embedded in conversational sentences."""
        assert has_spoiler("The massive PLOT TWIST in episode 10 shocked everyone.") is True
        assert has_spoiler("A painful betrayal occurred.") is True
        assert has_spoiler("This tragedy ruined their lives.") is True

    def test_has_spoiler_negative(self):
        """Returns False for safe conversational justification text."""
        assert has_spoiler("A gripping sci-fi drama with psychological depth.") is False
        assert has_spoiler("Você valorizou mistérios de alta intensidade e combates com grande peso dramático.") is False

    @pytest.mark.parametrize("safe_sentence,word_to_protect", [
        ("The protagonist studies hard every day.", "studies"),
        ("Ancient remedies cured the illness.", "remedies"),
        ("Defending the castle was the ultimate goal.", "defending"),
        ("Finally, the heroes reached home.", "finally"),
        ("The company will finalize the agreement tomorrow.", "finalize"),
        ("Spending time together was healing.", "spending"),
    ])
    def test_boundary_word_false_positive_avoidance(self, safe_sentence: str, word_to_protect: str):
        """
        CRITICAL DEFENSE: Words like 'studies' or 'defending' do NOT trigger redaction
        for substrings 'dies' or 'ending'.
        """
        assert has_spoiler(safe_sentence) is False
        sanitized = sanitize_reason(safe_sentence)
        assert sanitized == safe_sentence
        assert word_to_protect.lower() in sanitized.lower()
        assert "[redacted]" not in sanitized


# =============================================================================
# 3. Text Redaction & Sanitization Tests
# =============================================================================

class TestTextRedaction:
    """Verifies redaction of single, multiple, adversarial, and inflected spoiler terms."""

    def test_single_spoiler_redaction(self):
        """Redacts single canonical spoiler tag."""
        raw = "The major PLOT TWIST changed everything."
        expected = "The major [redacted] changed everything."
        assert sanitize_reason(raw) == expected

    def test_multiple_distinct_spoilers_redaction(self):
        """Redacts multiple distinct spoiler tags in a single sentence."""
        raw = "A terrible TRAGEDY and painful BETRAYAL occurred."
        expected = "A terrible [redacted] and painful [redacted] occurred."
        assert sanitize_reason(raw) == expected

    def test_repeated_spoiler_redaction(self):
        """Redacts all instances of repeated spoiler tags."""
        raw = "Plot Twist, plot twist, and PLOT TWIST in the same show."
        expected = "[redacted], [redacted], and [redacted] in the same show."
        assert sanitize_reason(raw) == expected

    @pytest.mark.parametrize("adversarial_sentence,expected_redacted", [
        ("Beware of P.l.o.t T.w.i.s.t in episode 12.", "Beware of [redacted] in episode 12."),
        ("An unexpected plot-twist awaits you.", "An unexpected [redacted] awaits you."),
        ("A plot_twist was revealed.", "A [redacted] was revealed."),
        ("The T.r.a.g.e.d.y was heartbreaking.", "The [redacted] was heartbreaking."),
        ("He committed a b-e-t-r-a-y-a-l against them.", "He committed a [redacted] against them."),
    ])
    def test_adversarial_bypass_in_reason(self, adversarial_sentence: str, expected_redacted: str):
        """Neutralizes dotted, hyphenated, and spaced evasion attempts in free text."""
        assert sanitize_reason(adversarial_sentence) == expected_redacted

    @pytest.mark.parametrize("inflection_sentence", [
        "Multiple tragedies happened throughout the season.",
        "Several betrayals ruined the alliance.",
        "Deep conspiracies were unraveled.",
    ])
    def test_plural_and_inflection_redaction(self, inflection_sentence: str):
        """Redacts plural inflections (tragedies, betrayals, conspiracies)."""
        sanitized = sanitize_reason(inflection_sentence)
        assert "[redacted]" in sanitized
        for forbidden in ["tragedies", "betrayals", "conspiracies"]:
            assert forbidden not in sanitized.lower()

    def test_custom_replacement_descriptor(self):
        """Supports semantic replacement descriptors instead of generic '[redacted]'."""
        raw = "This anime features an unforgettable plot twist and a deep tragedy."
        result = sanitize_reason(raw, descriptors=SAFE_THEMATIC_DESCRIPTORS)
        assert "narrative progression" in result
        assert "emotional drama" in result
        assert "plot twist" not in result.lower()
        assert "tragedy" not in result.lower()

    def test_sanitize_text_alias(self):
        """Verifies PROJECT.md interface contract sanitize_text compatibility."""
        raw = "Unbelievable plot twist!"
        assert sanitize_text(raw) == "Unbelievable [redacted]!"


# =============================================================================
# 4. Public DTO Transformation Helper Tests
# =============================================================================

class TestPublicDTOTransformation:
    """Verifies DTO sanitization and zero spoiler leakage in public responses."""

    @pytest.fixture
    def mixed_anime_entity(self) -> AnimeEntity:
        return AnimeEntity(
            id=9253,
            title_romaji="Steins;Gate",
            title_english="Steins;Gate",
            cover_image="https://example.com/cover.png",
            genres=["Drama", "Sci-Fi", "Suspense"],
            tags=[
                AnimeTag(id=6, name="Time Manipulation", rank=98, is_media_spoiler=False),
                AnimeTag(id=7, name="Psychological", rank=92, is_media_spoiler=False),
                AnimeTag(id=11, name="Plot Twist", rank=96, is_media_spoiler=True),
                AnimeTag(id=13, name="Tragedy", rank=88, is_media_spoiler=True),
                AnimeTag(id=15, name="Conspiracy", rank=85, is_media_spoiler=True),
                AnimeTag(id=20, name="Philosophy", rank=80, is_media_spoiler=False),
            ],
        )

    def test_filter_safe_tags_helper(self):
        """filter_safe_tags strips spoiler tags from mixed lists of tags, dicts, and strings."""
        mixed = [
            AnimeTag(id=1, name="Action", is_media_spoiler=False),
            AnimeTag(id=2, name="Plot Twist", is_media_spoiler=True),
            {"name": "Tragedy", "is_media_spoiler": True},
            {"name": "Sci-Fi", "is_media_spoiler": False},
            "Betrayal",
            "  Mystery  ",
            "plot-twist",
        ]
        clean = filter_safe_tags(mixed)
        assert clean == ["Action", "Sci-Fi", "Mystery"]

    def test_to_public_response_filters_all_spoilers(self, mixed_anime_entity: AnimeEntity):
        """to_public_response strips all spoilers and computes percentage correctly."""
        pub = to_public_response(
            mixed_anime_entity,
            compatibility_score=0.942,
            origin_track="affinity",
            reason="Incredible show with a shocking plot twist!",
        )

        assert isinstance(pub, AnimePublicResponse)
        assert pub.id == 9253
        assert pub.title_romaji == "Steins;Gate"
        assert pub.match_percentage == 94
        assert pub.origin_track == "affinity"

        # Tags check: strictly safe tags only
        assert "Time Manipulation" in pub.tags
        assert "Psychological" in pub.tags
        assert "Philosophy" in pub.tags
        assert "Plot Twist" not in pub.tags
        assert "Tragedy" not in pub.tags
        assert "Conspiracy" not in pub.tags

        # Reason check: spoiler was redacted
        assert pub.reason == "Incredible show with a shocking [redacted]!"

    def test_sanitize_anime_catalog_batch(self, mixed_anime_entity: AnimeEntity):
        """sanitize_anime_catalog transforms a sequence of entities with 0% leak."""
        catalog = [mixed_anime_entity] * 5
        public_catalog = sanitize_anime_catalog(catalog)

        assert len(public_catalog) == 5
        for item in public_catalog:
            for tag in item.tags:
                assert tag not in CANONICAL_SPOILER_TAGS

    def test_sanitize_recommendations_response(self, mixed_anime_entity: AnimeEntity):
        """sanitize_recommendations cleans both affinity and discovery tracks."""
        aff_item = to_public_response(mixed_anime_entity, compatibility_score=0.95, origin_track="affinity")
        disc_item = AnimePublicResponse(
            id=10,
            title_romaji="Test Disc",
            cover_image="https://example.com/cover.png",
            tags=["Fantasy", "Plot Twist"],  # Unsanitized input
            reason="Watch out for the betrayal!",
        )

        raw_resp = RecommendationResponse(
            affinity_track=[aff_item],
            discovery_track=[disc_item],
        )

        clean_resp = SpoilerService.sanitize_recommendations(raw_resp)
        assert "Plot Twist" not in clean_resp.discovery_track[0].tags
        assert clean_resp.discovery_track[0].tags == ["Fantasy"]
        assert clean_resp.discovery_track[0].reason == "Watch out for the [redacted]!"


# =============================================================================
# 5. Performance, Edge Cases & Robustness Tests
# =============================================================================

class TestEdgeCasesAndPerformance:
    """Verifies handling of empty inputs, edge cases, and ReDoS immunity."""

    def test_empty_and_none_inputs(self):
        """Handles None and empty string safely without exceptions."""
        assert is_spoiler_tag(None) is False
        assert is_spoiler_tag("") is False
        assert has_spoiler(None) is False
        assert has_spoiler("") is False
        assert sanitize_reason(None) == ""
        assert sanitize_reason("") == ""
        assert filter_safe_tags([]) == []

    def test_large_input_redos_immunity(self):
        """10,000 character string with repeated punctuation completes in < 50ms (no ReDoS)."""
        large_text = (
            "This is a long safe text describing character progression and world building. " * 150
            + " Suddenly, a P.l.o.t   T.w.i.s.t occurred! "
            + "More world building text continuing for hundreds of lines. " * 150
        )
        assert len(large_text) > 10000

        t0 = time.perf_counter()
        has_sp = has_spoiler(large_text)
        sanitized = sanitize_reason(large_text)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        assert has_sp is True
        assert "[redacted]" in sanitized
        assert elapsed_ms < 50.0, f"ReDoS vulnerability detected: took {elapsed_ms:.2f}ms"
