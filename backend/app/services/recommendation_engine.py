"""
backend/app/services/recommendation_engine.py
Continuous hybrid anime recommendation engine for AnimeRecSys AI.
Implements 80/20 Dual-Track Partitioning, Maximal Marginal Relevance (MMR),
and strict dual spoiler suppression for public DTO serialization.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np

SEQUEL_TITLE_REGEX = re.compile(
    r"("
    r"\bseason\s*[2-9]\b|"
    r"\b[2-9](?:nd|rd|th)\s*season\b|"
    r"\bpart\s*[2-9]\b|"
    r"\bcour\s*[2-9]\b|"
    r"\bthe\s*final\s*season\b|"
    r"\bfinal\s*season\b|"
    r"\bsecond\s*season\b|"
    r"\bthird\s*season\b|"
    r"\bfourth\s*season\b|"
    r"\bfifth\s*season\b|"
    r"\bsixth\s*season\b|"
    r"\b(?:ii|iii|iv|v|vi)\b|"
    r"\b2nd\s*cour\b|"
    r"\bzoku\b|"
    r"\bclimax\b|"
    r"\bshippuuden\b|"
    r"\bshippuden\b|"
    r"\bnext\s*generations\b"
    r")",
    re.IGNORECASE,
)

try:
    from app.core.storage_manager import StorageManager
    from app.models.anime import AnimeEntity, AnimePublicResponse, is_spoiler_tag, KNOWN_SPOILER_TAGS
    from app.models.recommendation import RecommendationResponse
    from app.models.user import UserProfileData, WatchHistoryEntry
    from app.services.vector_engine import VectorEngine
except ImportError:
    try:
        from backend.app.core.storage_manager import StorageManager
        from backend.app.models.anime import AnimeEntity, AnimePublicResponse, is_spoiler_tag, KNOWN_SPOILER_TAGS
        from backend.app.models.recommendation import RecommendationResponse
        from backend.app.models.user import UserProfileData, WatchHistoryEntry
        from backend.app.services.vector_engine import VectorEngine
    except ImportError:
        # Fallback relative imports for test execution environments
        from ..core.storage_manager import StorageManager
        from ..models.anime import AnimeEntity, AnimePublicResponse, is_spoiler_tag, KNOWN_SPOILER_TAGS
        from ..models.recommendation import RecommendationResponse
        from ..models.user import UserProfileData, WatchHistoryEntry
        from .vector_engine import VectorEngine

logger = logging.getLogger("animerecsys.recommendation_engine")


class RecommendationEngine:
    """
    Orchestrates candidate filtering, 80/20 dual-track partitioning,
    MMR diversity re-ranking, and DTO serialization with strict spoiler shielding.
    """

    DEFAULT_LAMBDA: float = 0.50
    DEFAULT_TOP_K: int = 10

    def __init__(
        self,
        vector_engine: Optional[VectorEngine] = None,
        storage_manager: Optional[StorageManager] = None,
        default_lambda: float = DEFAULT_LAMBDA,
        bipolar_scaling: bool = False,
    ) -> None:
        """
        Initializes the RecommendationEngine.

        Args:
            vector_engine: Optional instance of VectorEngine for vector projections.
            storage_manager: Optional instance of StorageManager for state loading.
            default_lambda: Default MMR trade-off parameter (0.50).
            bipolar_scaling: If True, maps [-1.0, 1.0] to [0.0, 1.0]; if False, clamps negative scores to 0.0.
        """
        self.vector_engine = vector_engine or VectorEngine()
        self.storage_manager = storage_manager
        self.default_lambda = float(default_lambda)
        self.bipolar_scaling = bool(bipolar_scaling)

    # ------------------------------------------------------------------------
    # Partitioning & Pool Management
    # ------------------------------------------------------------------------

    @staticmethod
    def partition_k(top_k: int, available_candidates: int) -> Tuple[int, int]:
        """
        Computes the 80/20 partition sizes for Affinity and Discovery tracks.

        Formula:
            K_eff = min(top_k, available_candidates)
            K_aff = min(round(0.80 * K_eff), available_candidates)
            K_disc = K_eff - K_aff

        Guarantees:
            K_aff + K_disc == min(top_k, available_candidates)
            K_aff >= 0, K_disc >= 0
        """
        if available_candidates <= 0 or top_k <= 0:
            return 0, 0

        k_eff = min(top_k, available_candidates)
        k_aff = int(round(0.80 * k_eff))
        # Ensure k_aff does not exceed available candidates
        k_aff = min(k_aff, available_candidates)
        k_disc = k_eff - k_aff
        return k_aff, k_disc

    @staticmethod
    def filter_candidate_pool(
        catalog: Sequence[AnimeEntity],
        history: Union[Sequence[WatchHistoryEntry], Set[int], Sequence[int], Sequence[Any]],
    ) -> List[AnimeEntity]:
        """
        Filters out any anime entity present in the user's watch history.

        Args:
            catalog: List of available AnimeEntity records.
            history: Watch history records or set of watched anime IDs.

        Returns:
            List of eligible AnimeEntity candidates strictly disjoint from history.
        """
        if not history:
            return list(catalog)

        watched_ids: Set[int] = set()
        if isinstance(history, set):
            watched_ids = history
        else:
            for item in history:
                if isinstance(item, int):
                    watched_ids.add(item)
                elif hasattr(item, "anime_id"):
                    watched_ids.add(item.anime_id)
                elif isinstance(item, dict) and "anime_id" in item:
                    watched_ids.add(int(item["anime_id"]))

        return [anime for anime in catalog if anime.id not in watched_ids]

    # ------------------------------------------------------------------------
    # Score & Percentage Normalization
    # ------------------------------------------------------------------------

    def similarity_to_score_and_percentage(
        self,
        sim: float,
        bipolar: Optional[bool] = None,
    ) -> Tuple[float, int]:
        """
        Converts raw cosine similarity to valid compatibility_score and match_percentage.

        Args:
            sim: Raw cosine similarity value.
            bipolar: Override instance bipolar_scaling setting if provided.

        Returns:
            Tuple[float, int]: (compatibility_score in [0.0, 1.0], match_percentage in [0, 100])
        """
        if math.isnan(sim) or math.isinf(sim) or np.isnan(sim) or np.isinf(sim):
            return 0.0, 0

        use_bipolar = self.bipolar_scaling if bipolar is None else bipolar

        if use_bipolar:
            # Linear mapping from [-1.0, 1.0] to [0.0, 1.0]
            score = (sim + 1.0) / 2.0
        else:
            # Standard rectified clamping: negative cosine is zero affinity
            score = sim

        score = max(0.0, min(1.0, score))
        pct = int(round(score * 100.0))
        pct = max(0, min(100, pct))
        return round(score, 4), pct

    # ------------------------------------------------------------------------
    # Vector Computation & Similarity Kernels
    # ------------------------------------------------------------------------

    def build_profile_vector(self, profile: UserProfileData) -> np.ndarray:
        """
        Constructs a normalized dense vector from the user's sparse profile weights.
        Falls back to cold-start uniform vector if profile is empty or cold start.
        """
        vocab = self.vector_engine.vocabulary
        dim = len(vocab)
        if dim == 0:
            return np.zeros(0, dtype=np.float64)

        if profile.is_cold_start() or not profile.feature_weights:
            # Cold-start uniform unit vector: 1 / sqrt(D) * 1
            return np.ones(dim, dtype=np.float64) / math.sqrt(dim)

        vec = np.zeros(dim, dtype=np.float64)
        vocab_index = self.vector_engine.vocab_index
        for term, weight in profile.feature_weights.items():
            idx = vocab_index.get(term)
            if idx is not None:
                vec[idx] = float(weight)

        norm = float(np.linalg.norm(vec))
        if norm < 1e-9:
            return np.ones(dim, dtype=np.float64) / math.sqrt(dim)

        return vec / norm

    def compute_candidate_vectors(
        self,
        candidates: Sequence[AnimeEntity],
    ) -> np.ndarray:
        """
        Computes dense L2-normalized feature vectors for a list of candidate anime entities.

        Returns:
            np.ndarray of shape (len(candidates), D).
        """
        if not candidates:
            return np.zeros((0, len(self.vector_engine.vocabulary)), dtype=np.float64)

        return self.vector_engine.transform_entities(candidates)

    # ------------------------------------------------------------------------
    # Maximal Marginal Relevance (MMR) Reranker
    # ------------------------------------------------------------------------

    def rerank_mmr(
        self,
        user_vec: np.ndarray,
        candidate_pool: List[AnimeEntity],
        candidate_vecs: np.ndarray,
        already_selected_vecs: List[np.ndarray],
        k_disc: int,
        lambda_param: float = DEFAULT_LAMBDA,
    ) -> List[Tuple[AnimeEntity, float]]:
        """
        Selects k_disc items using Maximal Marginal Relevance (MMR).

        MMR Equation:
            MMR(i) = lambda * Sim_1(i, u) - (1 - lambda) * max_{j in S} Sim_2(i, j)
            where S = already_selected_vecs

        Args:
            user_vec: Normalized user preference vector (shape: D,).
            candidate_pool: List of AnimeEntity candidates eligible for discovery.
            candidate_vecs: Matrix of candidate vectors (shape: P, D).
            already_selected_vecs: Vectors of already selected affinity items.
            k_disc: Target discovery item count.
            lambda_param: Diversity trade-off factor in [0.0, 1.0].

        Returns:
            List of Tuples: [(AnimeEntity, raw_user_similarity), ...]
        """
        if k_disc <= 0 or not candidate_pool:
            return []

        selected_items: List[Tuple[AnimeEntity, float]] = []
        selected_vecs: List[np.ndarray] = list(already_selected_vecs)

        # Remaining indices within candidate_pool
        remaining_indices = list(range(len(candidate_pool)))
        target_count = min(k_disc, len(candidate_pool))

        for _ in range(target_count):
            if not remaining_indices:
                break

            pool_vecs = candidate_vecs[remaining_indices]  # Shape: (rem_len, D)
            # Relevance to user: Sim_1(i, u)
            user_sims = np.dot(pool_vecs, user_vec)  # Shape: (rem_len,)

            # Maximum similarity to already selected items: max_{j in S} Sim_2(i, j)
            if selected_vecs:
                s_matrix = np.array(selected_vecs, dtype=np.float64)  # Shape: (|S|, D)
                redundancy_matrix = np.dot(pool_vecs, s_matrix.T)     # Shape: (rem_len, |S|)
                max_redundancy = np.max(redundancy_matrix, axis=1)    # Shape: (rem_len,)
            else:
                max_redundancy = np.zeros(len(remaining_indices), dtype=np.float64)

            # MMR Score computation
            mmr_scores = (lambda_param * user_sims) - ((1.0 - lambda_param) * max_redundancy)

            # Deterministic tie-breaking: add tiny deterministic weight based on average score & ID
            tie_breakers = np.array([
                (candidate_pool[idx].average_score or 50) * 1e-6 - (candidate_pool[idx].id) * 1e-12
                for idx in remaining_indices
            ])
            adjusted_scores = mmr_scores + tie_breakers

            best_local_idx = int(np.argmax(adjusted_scores))
            best_pool_idx = remaining_indices[best_local_idx]

            chosen_entity = candidate_pool[best_pool_idx]
            chosen_vec = candidate_vecs[best_pool_idx]
            chosen_sim = float(user_sims[best_local_idx])

            selected_items.append((chosen_entity, chosen_sim))
            selected_vecs.append(chosen_vec)
            remaining_indices.pop(best_local_idx)

        return selected_items

    # ------------------------------------------------------------------------
    # Primary Recommendation Pipeline
    # ------------------------------------------------------------------------

    def generate_recommendations(
        self,
        profile: UserProfileData,
        catalog: List[AnimeEntity],
        history: Union[Sequence[WatchHistoryEntry], Set[int], Sequence[int], Sequence[Any]],
        top_k: int = DEFAULT_TOP_K,
        lambda_param: Optional[float] = None,
        excluded_formats: Optional[Sequence[str]] = None,
        blocked_genres: Optional[Sequence[str]] = None,
    ) -> RecommendationResponse:
        """
        Generates 80/20 partitioned recommendations adhering to all mathematical invariants.

        Args:
            profile: User profile containing learned preference weights and hyperparameters.
            catalog: Ingested anime catalog entities.
            history: User watch history records to exclude.
            top_k: Total requested recommendations (default: 10).
            lambda_param: Optional override for MMR diversity parameter.
            excluded_formats: Optional override for formats to exclude (e.g. ['MOVIE', 'OVA']).
            blocked_genres: Optional override for genres to block (e.g. ['Mecha', 'Horror']).

        Returns:
            RecommendationResponse containing strictly disjoint affinity_track and discovery_track.
        """
        # Resolve lambda parameter
        eff_lambda = (
            lambda_param
            if lambda_param is not None
            else getattr(profile.hyperparameters, "lambda_mmr", self.default_lambda)
        )

        # 1. Candidate Pool Filtering (Strict Watch History, Skipped & Dropped Exclusion)
        eligible_candidates = self.filter_candidate_pool(catalog, history)
        excluded_profile_ids = set(getattr(profile, "skipped_anime_ids", [])) | set(
            getattr(profile, "dropped_anime_ids", [])
        )
        if excluded_profile_ids:
            eligible_candidates = [
                a for a in eligible_candidates if a.id not in excluded_profile_ids
            ]

        # 1a. Exclude Music Videos (format == 'MUSIC') and Unreleased Animes (status == 'NOT_YET_RELEASED' or 'CANCELLED')
        eligible_candidates = [
            a
            for a in eligible_candidates
            if (a.format or "TV").strip().upper() != "MUSIC"
            and (a.status or "FINISHED").strip().upper() not in ("NOT_YET_RELEASED", "CANCELLED")
        ]

        # 1a-bis. Sequel Gating: Never recommend a later season/sequel unless its predecessor has been positively rated
        positively_rated_ids: Set[int] = set()
        if history and not isinstance(history, set):
            for h in history:
                if hasattr(h, "anime_id") and hasattr(h, "rating"):
                    st = getattr(h, "interaction_status", "watched")
                    if st != "dropped" and float(h.rating) >= 5.5:
                        positively_rated_ids.add(int(h.anime_id))
                elif isinstance(h, dict) and "anime_id" in h:
                    if h.get("interaction_status") != "dropped" and float(h.get("rating", 7.0)) >= 5.5:
                        positively_rated_ids.add(int(h["anime_id"]))
        elif isinstance(history, set):
            positively_rated_ids = set(history)

        def _is_eligible_season(anime: AnimeEntity) -> bool:
            preq_ids = getattr(anime, "prequel_ids", None) or []
            if preq_ids:
                # Must have evaluated at least one direct prequel/parent positively
                return any(pid in positively_rated_ids for pid in preq_ids)
            if getattr(anime, "is_sequel", False):
                return False
            title_combined = f"{anime.title_romaji or ''} {anime.title_english or ''}"
            if SEQUEL_TITLE_REGEX.search(title_combined):
                return False
            return True

        eligible_candidates = [a for a in eligible_candidates if _is_eligible_season(a)]

        # 1b. Format Filtering (e.g. block MOVIE, OVA, ONA, SPECIAL when user wants only TV/Series)
        raw_excluded_formats = (
            excluded_formats
            if excluded_formats is not None
            else getattr(profile, "excluded_formats", [])
        )
        if raw_excluded_formats:
            blocked_fmt_set = {str(f).strip().upper() for f in raw_excluded_formats if str(f).strip()}
            if blocked_fmt_set:
                eligible_candidates = [
                    a
                    for a in eligible_candidates
                    if (a.format or "TV").strip().upper() not in blocked_fmt_set
                ]

        # 1c. Genre Blocking (e.g. block Mecha, Horror, Romance when user opts out of those genres)
        raw_blocked_genres = (
            blocked_genres
            if blocked_genres is not None
            else getattr(profile, "blocked_genres", [])
        )
        if raw_blocked_genres:
            blocked_genre_set = {str(g).strip().lower() for g in raw_blocked_genres if str(g).strip()}
            if blocked_genre_set:
                eligible_candidates = [
                    a
                    for a in eligible_candidates
                    if not any(g.strip().lower() in blocked_genre_set for g in (a.genres or []))
                ]

        available_count = len(eligible_candidates)

        # 2. Partition Sizing
        k_aff, k_disc = self.partition_k(top_k, available_count)

        if available_count == 0 or (k_aff == 0 and k_disc == 0):
            return RecommendationResponse(
                affinity_track=[],
                discovery_track=[],
            )

        # 3. Vector Embeddings
        user_vec = self.build_profile_vector(profile)
        candidate_vecs = self.compute_candidate_vectors(eligible_candidates)

        # 4. Affinity Track Ranking (Pure Cosine Similarity)
        similarities = np.dot(candidate_vecs, user_vec)  # Shape: (P,)

        # Sort indices deterministically: similarity desc, average_score desc, popularity desc, id asc
        ranked_indices = sorted(
            range(available_count),
            key=lambda i: (
                similarities[i],
                eligible_candidates[i].average_score or 0,
                eligible_candidates[i].popularity,
                -eligible_candidates[i].id,
            ),
            reverse=True,
        )

        # Select top k_aff for Affinity Track
        affinity_indices = ranked_indices[:k_aff]
        affinity_entities = [eligible_candidates[i] for i in affinity_indices]
        affinity_vecs = [candidate_vecs[i] for i in affinity_indices]
        affinity_sims = [float(similarities[i]) for i in affinity_indices]

        # 5. Discovery Track (MMR Reranker on Remaining Pool)
        remaining_pool_indices = ranked_indices[k_aff:]
        discovery_pool_entities = [eligible_candidates[i] for i in remaining_pool_indices]
        discovery_pool_vecs = (
            candidate_vecs[remaining_pool_indices]
            if remaining_pool_indices
            else np.zeros((0, candidate_vecs.shape[1]), dtype=np.float64)
        )

        discovery_selected = self.rerank_mmr(
            user_vec=user_vec,
            candidate_pool=discovery_pool_entities,
            candidate_vecs=discovery_pool_vecs,
            already_selected_vecs=affinity_vecs,
            k_disc=k_disc,
            lambda_param=eff_lambda,
        )

        # 6. DTO Mapping & Strict Spoiler Shielding
        affinity_public: List[AnimePublicResponse] = []
        for entity, sim in zip(affinity_entities, affinity_sims):
            score, pct = self.similarity_to_score_and_percentage(sim)
            pub_item = entity.to_public_response(
                compatibility_score=score,
                match_percentage=pct,
                origin_track="affinity",
                reason="Alta correspondência com seus gêneros e tags favoritos",
            )
            affinity_public.append(pub_item)

        discovery_public: List[AnimePublicResponse] = []
        for entity, sim in discovery_selected:
            score, pct = self.similarity_to_score_and_percentage(sim)
            pub_item = entity.to_public_response(
                compatibility_score=score,
                match_percentage=pct,
                origin_track="discovery",
                reason="Recomendado para expandir horizontes com diversidade temática (MMR)",
            )
            discovery_public.append(pub_item)

        # 7. Construct & Return Validated Response
        # Pydantic RecommendationResponse validates S_aff ∩ S_disc == ∅
        return RecommendationResponse(
            affinity_track=affinity_public,
            discovery_track=discovery_public,
        )

    # ------------------------------------------------------------------------
    # Persistence Integration & Convenience Entry Points
    # ------------------------------------------------------------------------

    def get_recommendations(
        self,
        top_k: int = DEFAULT_TOP_K,
        user_id: str = "default_user",
        lambda_param: Optional[float] = None,
    ) -> RecommendationResponse:
        """
        Convenience method loading catalog, user profile, and watch history
        from the persistent StorageManager before running recommendation generation.
        """
        if self.storage_manager is None:
            raise ValueError(
                "RecommendationEngine was initialized without a StorageManager. "
                "Pass storage_manager in __init__ or use generate_recommendations() directly."
            )

        catalog = self.storage_manager.load_catalog()
        profile = self.storage_manager.load_user_profile()
        history = self.storage_manager.load_watch_history()

        return self.generate_recommendations(
            profile=profile,
            catalog=catalog,
            history=history,
            top_k=top_k,
            lambda_param=lambda_param,
        )

    # ------------------------------------------------------------------------
    # Analytical & Diversity Metrics
    # ------------------------------------------------------------------------

    @staticmethod
    def compute_intra_list_diversity(vectors: Sequence[np.ndarray]) -> float:
        """
        Computes the Intra-List Diversity (ILD) for a set of unit vectors.
        ILD = 1.0 - mean(pairwise cosine similarities).

        Returns:
            float in [0.0, 2.0] (higher represents greater diversity).
        """
        n = len(vectors)
        if n < 2:
            return 0.0

        mat = np.array(vectors, dtype=np.float64)  # Shape: (n, D)
        # Pairwise dot products
        sim_mat = np.dot(mat, mat.T)  # Shape: (n, n)

        # Extract upper triangle without diagonal
        upper_indices = np.triu_indices(n, k=1)
        pairwise_sims = sim_mat[upper_indices]

        mean_sim = float(np.mean(pairwise_sims))
        return 1.0 - mean_sim
