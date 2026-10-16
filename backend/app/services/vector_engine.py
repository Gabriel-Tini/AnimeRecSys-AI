"""
backend/app/services/vector_engine.py
Mathematical Vector Space Model (VSM) Engine for AnimeRecSys AI.
Handles feature vocabulary mapping, dense vector encoding, L2 hypersphere projection,
cosine similarity kernels, and cold-start profile initialization.
Strict dual spoiler policy compliant (preserves internal spoiler tags per R3).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import numpy as np

try:
    from app.models.anime import AnimeEntity, AnimeTag, KNOWN_SPOILER_TAGS
    from app.models.user import UserProfileData
except ImportError:
    try:
        from backend.app.models.anime import AnimeEntity, AnimeTag, KNOWN_SPOILER_TAGS
        from backend.app.models.user import UserProfileData
    except ImportError:
        # Standalone or local import fallback
        from ..models.anime import AnimeEntity, AnimeTag, KNOWN_SPOILER_TAGS
        from ..models.user import UserProfileData

logger = logging.getLogger("animerecsys.vector_engine")

# Canonical default category multipliers for community tags
DEFAULT_CATEGORY_WEIGHTS: Dict[str, float] = {
    "Theme": 1.0,
    "Setting": 0.9,
    "Cast-Traits": 0.8,
    "Demographic": 0.85,
    "Technical": 0.5,
}

# Canonical 20 base tags from .antigravity/FIXTURES.json
CANONICAL_FIXTURE_TAGS: List[str] = [
    "Action",
    "Adventure",
    "Fantasy",
    "Sci-Fi",
    "Cyberpunk",
    "Time Manipulation",
    "Psychological",
    "Mystery",
    "Supernatural",
    "Shounen",
    "Plot Twist",             # Spoiler tag (R3 preserved in internal VSM)
    "Betrayal",               # Spoiler tag (R3 preserved in internal VSM)
    "Tragedy",                # Spoiler tag (R3 preserved in internal VSM)
    "Anti-Hero",
    "Conspiracy",             # Spoiler tag (R3 preserved in internal VSM)
    "Memory Manipulation",    # Spoiler tag (R3 preserved in internal VSM)
    "Philosophy",
    "Coming of Age",
    "Slice of Life",
    "Music",
]

# Standard 19 AniList official genres
CANONICAL_ANILIST_GENRES: List[str] = [
    "Action",
    "Adventure",
    "Comedy",
    "Drama",
    "Ecchi",
    "Fantasy",
    "Hentai",
    "Horror",
    "Mahou Shoujo",
    "Mecha",
    "Music",
    "Mystery",
    "Psychological",
    "Romance",
    "Sci-Fi",
    "Slice of Life",
    "Sports",
    "Supernatural",
    "Suspense",
]


class VectorEngine:
    """
    High-performance Vector Space Modeling engine.
    Manages deterministic vocabulary index spaces, dense item vectorization,
    L2 normalization invariants, matrix-vector cosine similarities, and cold-start profiles.
    """

    EPSILON: float = 1e-9
    UNIT_NORM_TOLERANCE: float = 1e-6

    def __init__(
        self,
        vocabulary: Optional[Sequence[str]] = None,
        category_weights: Optional[Dict[str, float]] = None,
        epsilon: float = EPSILON,
        combine_mode: Literal["tag_priority", "max", "additive"] = "tag_priority",
    ) -> None:
        """
        Initializes the VectorEngine with an ordered vocabulary.

        Args:
            vocabulary: Ordered sequence of unique feature terms. If None, uses CANONICAL_FIXTURE_TAGS.
            category_weights: Optional category multiplier dictionary.
            epsilon: Numerical protection term for hypersphere projection (default 1e-9).
            combine_mode: Strategy when an item has both a genre and tag with the same name.
        """
        self.epsilon = float(epsilon)
        self.combine_mode = combine_mode
        self.category_weights = (
            dict(category_weights) if category_weights is not None else dict(DEFAULT_CATEGORY_WEIGHTS)
        )

        if vocabulary is None:
            self._vocabulary = list(CANONICAL_FIXTURE_TAGS)
        else:
            self._vocabulary = list(vocabulary)

        self._build_index_mappings()

    def _build_index_mappings(self) -> None:
        """Constructs bidirectional index mapping tables and fast normalized lookups."""
        self._term_to_index: Dict[str, int] = {
            term: idx for idx, term in enumerate(self._vocabulary)
        }
        self._index_to_term: Dict[int, str] = {
            idx: term for idx, term in enumerate(self._vocabulary)
        }
        # Normalized lookup for defensive fallback (lowercase and stripped)
        self._norm_to_index: Dict[str, int] = {
            term.strip().lower(): idx for idx, term in enumerate(self._vocabulary)
        }

    @property
    def vocabulary(self) -> List[str]:
        """Returns a copy of the ordered vocabulary terms."""
        return list(self._vocabulary)

    @property
    def dimension(self) -> int:
        """Returns the dimensionality D of the vector space."""
        return len(self._vocabulary)

    @property
    def vocab_index(self) -> Dict[str, int]:
        """Returns a dictionary mapping term -> index."""
        return dict(self._term_to_index)

    def term_index(self, term: str) -> Optional[int]:
        """Returns the coordinate index for a given term, or None if not in vocabulary."""
        idx = self._term_to_index.get(term)
        if idx is not None:
            return idx
        return self._norm_to_index.get(term.strip().lower())

    # ------------------------------------------------------------------------
    # Vocabulary Factories
    # ------------------------------------------------------------------------

    @classmethod
    def from_fixtures(
        cls,
        fixtures_path: Optional[Union[str, Path]] = None,
        include_genres: bool = False,
    ) -> "VectorEngine":
        """
        Creates an engine instance using vocabulary loaded from .antigravity/FIXTURES.json.
        """
        path = (
            Path(fixtures_path)
            if fixtures_path
            else Path(__file__).resolve().parents[3] / ".antigravity" / "FIXTURES.json"
        )
        if not path.exists():
            logger.warning(f"Fixtures file not found at {path}, falling back to default tags.")
            return cls(vocabulary=CANONICAL_FIXTURE_TAGS)

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            tags_vocab = data.get("tags_vocabulary", [])
            vocab = [t["name"] for t in tags_vocab if "name" in t]
            if include_genres:
                for a in data.get("sample_animes_full", []):
                    for g in a.get("genres", []):
                        if g not in vocab:
                            vocab.append(g)
            return cls(vocabulary=vocab)
        except Exception as exc:
            logger.error(f"Error loading fixtures vocabulary: {exc}")
            return cls(vocabulary=CANONICAL_FIXTURE_TAGS)

    @classmethod
    def from_catalog(
        cls,
        catalog: List[AnimeEntity],
        base_vocabulary: Optional[Sequence[str]] = None,
    ) -> "VectorEngine":
        """
        Builds a comprehensive engine by enumerating all genres and tags across a catalog.
        Preserves base vocabulary ordering and appends newly discovered terms deterministically.
        """
        vocab = list(base_vocabulary) if base_vocabulary else list(CANONICAL_FIXTURE_TAGS)
        existing = set(vocab)

        discovered_genres: List[str] = []
        discovered_tags: List[str] = []

        for anime in catalog:
            for g in anime.genres:
                clean_g = g.strip()
                if clean_g and clean_g not in existing:
                    existing.add(clean_g)
                    discovered_genres.append(clean_g)
            for t in anime.tags:
                clean_t = t.name.strip() if hasattr(t, "name") else str(t.get("name", "")).strip()
                if clean_t and clean_t not in existing:
                    existing.add(clean_t)
                    discovered_tags.append(clean_t)

        discovered_genres.sort()
        discovered_tags.sort()
        vocab.extend(discovered_genres)
        vocab.extend(discovered_tags)
        return cls(vocabulary=vocab)

    # ------------------------------------------------------------------------
    # Vector Normalization & Projections
    # ------------------------------------------------------------------------

    def normalize_vector(self, v: np.ndarray) -> np.ndarray:
        """
        Projects a raw vector onto the unit hypersphere: hat{v} = v / (||v||_2 + eps).
        Guarantees | ||hat{v}||_2 - 1.0 | < 10^-6 for non-zero vectors.
        Provides safe fallback returning np.zeros_like(v) if ||v||_2 < eps.
        """
        vec = np.asarray(v, dtype=np.float64)
        norm = float(np.linalg.norm(vec))
        if norm < self.epsilon:
            return np.zeros_like(vec)
        normalized = vec / (norm + self.epsilon)
        return normalized

    l2_normalize = normalize_vector

    # ------------------------------------------------------------------------
    # Feature Vector Construction
    # ------------------------------------------------------------------------

    def build_feature_vector(
        self,
        anime: Union[AnimeEntity, Dict[str, Any], Sequence[str], None] = None,
        tags: Optional[Sequence[Any]] = None,
        vocabulary: Optional[Sequence[str]] = None,
    ) -> np.ndarray:
        """
        Constructs a dense L2-normalized feature vector for an anime.

        Supports multiple calling conventions for 100% interoperability:
        1. engine.build_feature_vector(anime_entity)
        2. engine.build_feature_vector(anime_dict)
        3. engine.build_feature_vector(genres_list, tags_list)
        4. Functional override: engine.build_feature_vector(..., vocabulary=custom_vocab)
        """
        active_engine = self
        if vocabulary is not None and list(vocabulary) != self._vocabulary:
            active_engine = VectorEngine(
                vocabulary=vocabulary,
                category_weights=self.category_weights,
                epsilon=self.epsilon,
                combine_mode=self.combine_mode,
            )

        D = active_engine.dimension
        vec = np.zeros(D, dtype=np.float64)

        if D == 0:
            return vec

        raw_genres: List[str] = []
        raw_tags: List[Any] = []

        # Parse inputs
        if isinstance(anime, (list, tuple)) and tags is not None:
            # Called as (genres, tags)
            raw_genres = list(anime)
            raw_tags = list(tags)
        elif hasattr(anime, "genres") and hasattr(anime, "tags"):
            raw_genres = list(getattr(anime, "genres", []))
            raw_tags = list(getattr(anime, "tags", []))
        elif isinstance(anime, dict):
            raw_genres = list(anime.get("genres", []))
            raw_tags = list(anime.get("tags", []))
        elif anime is None and tags is None:
            return vec

        # 1. Apply genres (weight 1.0)
        for g in raw_genres:
            idx = active_engine.term_index(g)
            if idx is not None:
                vec[idx] = 1.0

        # 2. Apply community tags (rank / 100.0 * category_weight)
        # Note: Internal VSM retains spoiler tags per R3
        for t in raw_tags:
            tag_name: str = ""
            rank_val: float = 100.0
            cat_name: Optional[str] = None

            if isinstance(t, AnimeTag):
                tag_name = t.name
                rank_val = float(t.rank)
                cat_name = t.category
            elif isinstance(t, dict):
                tag_name = str(t.get("name", ""))
                rank_val = float(t.get("rank", 100))
                cat_name = t.get("category")
            elif isinstance(t, str):
                tag_name = t
                rank_val = 100.0
            elif hasattr(t, "name"):
                tag_name = str(getattr(t, "name", ""))
                rank_val = float(getattr(t, "rank", 100))
                cat_name = getattr(t, "category", None)

            idx = active_engine.term_index(tag_name)
            if idx is not None:
                cat_multiplier = (
                    active_engine.category_weights.get(cat_name, 1.0)
                    if cat_name
                    else 1.0
                )
                tag_weight = (rank_val / 100.0) * cat_multiplier

                if active_engine.combine_mode == "tag_priority":
                    vec[idx] = tag_weight
                elif active_engine.combine_mode == "max":
                    vec[idx] = max(vec[idx], tag_weight)
                elif active_engine.combine_mode == "additive":
                    vec[idx] += tag_weight

        # 3. L2 hypersphere unit projection
        return active_engine.normalize_vector(vec)

    def build_catalog_matrix(self, catalog: List[AnimeEntity]) -> np.ndarray:
        """
        Vectorizes an entire anime catalog into an (N, D) row-normalized matrix.
        Optimized for high-speed batch matrix multiplications.
        """
        N = len(catalog)
        D = self.dimension
        matrix = np.zeros((N, D), dtype=np.float64)
        for i, anime in enumerate(catalog):
            matrix[i] = self.build_feature_vector(anime)
        return matrix

    def transform_entities(self, entities: Sequence[AnimeEntity]) -> np.ndarray:
        """Transforms a sequence of anime entities into a feature matrix (N, D)."""
        return self.build_catalog_matrix(list(entities))

    # ------------------------------------------------------------------------
    # Similarity Kernels & Scoring
    # ------------------------------------------------------------------------

    def cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """
        Calculates cosine similarity between two unit vectors: Sim(u, v) = u . v.
        Clamps result to [-1.0, 1.0] to eliminate floating-point drift.
        Safe against zero vectors (returns 0.0 without NaN).
        """
        va = np.asarray(vec_a, dtype=np.float64)
        vb = np.asarray(vec_b, dtype=np.float64)
        dot_product = float(np.dot(va, vb))
        if np.isnan(dot_product) or math.isnan(dot_product):
            return 0.0
        return float(np.clip(dot_product, -1.0, 1.0))

    def compute_similarity(
        self,
        user_vec: np.ndarray,
        item_vecs: Union[np.ndarray, List[np.ndarray]],
    ) -> np.ndarray:
        """
        Calculates cosine similarity between user vector and a collection/matrix of item vectors.

        Args:
            user_vec: (D,) 1D array representing user preference vector.
            item_vecs: (N, D) 2D array or list of (D,) 1D arrays representing candidate items.

        Returns:
            (N,) 1D array of similarity scores in [-1.0, 1.0].
        """
        u = np.asarray(user_vec, dtype=np.float64)
        V = np.asarray(item_vecs, dtype=np.float64)

        if V.ndim == 1:
            # Single item vector comparison
            return np.array([self.cosine_similarity(u, V)], dtype=np.float64)

        if V.shape[0] == 0:
            return np.empty((0,), dtype=np.float64)

        # High-performance BLAS GEMV: V @ u
        scores = np.dot(V, u)
        # Neutralize any potential NaN values from corrupt zero inputs
        scores = np.nan_to_num(scores, nan=0.0)
        return np.clip(scores, -1.0, 1.0)

    def score_candidates(
        self,
        user_vec: np.ndarray,
        candidate_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Scores candidate items via batch BLAS matrix-vector dot product.
        Alias / direct method for candidate ranking.
        """
        return self.compute_similarity(user_vec, candidate_matrix)

    # ------------------------------------------------------------------------
    # Cold-Start Profile
    # ------------------------------------------------------------------------

    def get_cold_start_profile(
        self,
        strategy: Literal["isotropic", "zero"] = "isotropic",
    ) -> np.ndarray:
        """
        Generates baseline cold-start profile vector.

        Strategies:
        - 'isotropic': u_0 = 1 / sqrt(D) * 1. Has exact unit norm (||u_0||_2 = 1.0)
          and equal uninformative affinity across all known features.
        - 'zero': u_0 = 0. Pure zero vector for clean first-update gradient addition.
        """
        D = self.dimension
        if D == 0 or strategy == "zero":
            return np.zeros(D, dtype=np.float64)

        # Isotropic hypersphere uniform vector
        val = 1.0 / np.sqrt(float(D))
        u0 = np.full(D, val, dtype=np.float64)
        return u0

    # ------------------------------------------------------------------------
    # User Profile Encoding & Decoding
    # ------------------------------------------------------------------------

    def encode_user_profile(
        self,
        profile: UserProfileData,
        default_strategy: Literal["isotropic", "zero"] = "zero",
    ) -> np.ndarray:
        """
        Encodes a UserProfileData domain model (sparse feature_weights dict) into a dense vector.
        """
        if profile.is_cold_start() or not profile.feature_weights:
            return self.get_cold_start_profile(strategy=default_strategy)

        vec = np.zeros(self.dimension, dtype=np.float64)
        for term, weight in profile.feature_weights.items():
            idx = self.term_index(term)
            if idx is not None:
                vec[idx] = float(weight)

        norm = float(np.linalg.norm(vec))
        if norm >= self.epsilon:
            return self.normalize_vector(vec)
        return self.get_cold_start_profile(strategy=default_strategy)

    def decode_user_profile(
        self,
        user_vec: np.ndarray,
        threshold: float = 1e-4,
    ) -> Dict[str, float]:
        """
        Decodes a dense user vector into a sparse dictionary mapping term -> weight.
        Filters out negligible weights below threshold to optimize JSON serialization size.
        """
        sparse_weights: Dict[str, float] = {}
        for idx, weight in enumerate(user_vec):
            if abs(weight) >= threshold:
                term = self._index_to_term.get(idx)
                if term:
                    sparse_weights[term] = round(float(weight), 6)
        return sparse_weights


# ============================================================================
# Functional Wrappers for Drop-in Interoperability & Backward Compatibility
# ============================================================================

_DEFAULT_ENGINE = VectorEngine(CANONICAL_FIXTURE_TAGS)


def normalize_vector(v: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Projects vector v to unit hypersphere with epsilon protection."""
    vec = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < epsilon:
        return np.zeros_like(vec)
    return vec / (norm + epsilon)


def l2_normalize(v: np.ndarray, epsilon: float = 1e-9) -> np.ndarray:
    """Alias for normalize_vector."""
    return normalize_vector(v, epsilon=epsilon)


def build_feature_vector(
    anime: Union[AnimeEntity, Dict[str, Any], Sequence[str]],
    tags: Optional[Sequence[Any]] = None,
    vocabulary: Optional[Sequence[str]] = None,
) -> np.ndarray:
    """
    Functional wrapper for building L2-normalized feature vectors.
    Compatible with:
    - build_feature_vector(anime, vocabulary=vocab)
    - build_feature_vector(genres, tags, vocabulary=vocab)
    """
    if vocabulary is not None:
        engine = VectorEngine(vocabulary=vocabulary)
        return engine.build_feature_vector(anime, tags=tags)
    return _DEFAULT_ENGINE.build_feature_vector(anime, tags=tags)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Computes scalar cosine similarity between unit vectors."""
    return _DEFAULT_ENGINE.cosine_similarity(vec_a, vec_b)


def compute_similarity(
    user_vec: np.ndarray,
    item_vecs: Union[np.ndarray, List[np.ndarray]],
) -> np.ndarray:
    """Computes batch cosine similarity between user vector and candidate item vectors."""
    return _DEFAULT_ENGINE.compute_similarity(user_vec, item_vecs)


def score_candidates(
    user_vec: np.ndarray,
    candidate_matrix: np.ndarray,
) -> np.ndarray:
    """Scores candidate items via batch BLAS matrix-vector dot product."""
    return _DEFAULT_ENGINE.score_candidates(user_vec, candidate_matrix)


def get_cold_start_vector(
    dimension: Optional[int] = None,
    strategy: Literal["isotropic", "zero"] = "isotropic",
) -> np.ndarray:
    """Generates baseline cold-start vector u_0 = 1 / sqrt(D) * 1."""
    if dimension is not None:
        if dimension <= 0 or strategy == "zero":
            return np.zeros(dimension, dtype=np.float64)
        return np.full(dimension, 1.0 / np.sqrt(float(dimension)), dtype=np.float64)
    return _DEFAULT_ENGINE.get_cold_start_profile(strategy=strategy)
