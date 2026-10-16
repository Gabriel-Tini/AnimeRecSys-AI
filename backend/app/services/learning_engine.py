"""
backend/app/services/learning_engine.py
Continuous Online Learning Engine for AnimeRecSys AI.

Implements:
- Online gradient delta learning: u'_{t+1} = gamma * u_t + eta * (R - R0) * v_hat_i + Delta u_gemini
- Mandatory L2 spherical projection: u_{t+1} = u' / (||u'||_2 + epsilon)
- Mathematical invariant: ||u_t||_2 = 1.0 +- 1e-6 across >= 100 updates
- Maximal Marginal Relevance (MMR) diversity reranker (lambda = 0.50)
- Thread-safe and process-safe persistence via StorageManager
"""

from __future__ import annotations

import logging
import math
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, Sequence, Set, Tuple, Union

import numpy as np

try:
    from app.core.storage_manager import StorageManager
    from app.models.anime import AnimeEntity, AnimeTag
    from app.models.recommendation import SemanticTagDelta, SemanticTuningResponse
    from app.models.user import Hyperparameters, UserProfileData, WatchHistoryEntry
except ImportError:
    try:
        from backend.app.core.storage_manager import StorageManager
        from backend.app.models.anime import AnimeEntity, AnimeTag
        from backend.app.models.recommendation import SemanticTagDelta, SemanticTuningResponse
        from backend.app.models.user import Hyperparameters, UserProfileData, WatchHistoryEntry
    except ImportError:
        from ..core.storage_manager import StorageManager
        from ..models.anime import AnimeEntity, AnimeTag
        from ..models.recommendation import SemanticTagDelta, SemanticTuningResponse
        from ..models.user import Hyperparameters, UserProfileData, WatchHistoryEntry

logger = logging.getLogger("animerecsys.learning")


class LearningEngine:
    """
    Mathematical Online Learning Engine managing real-time preference adaptation,
    weight decay, spherical projection invariants, and storage persistence.
    """

    EPSILON: float = 1e-9
    DEFAULT_ETA: float = 0.15
    DEFAULT_GAMMA: float = 0.98
    DEFAULT_ALPHA: float = 0.25
    NEUTRAL_RATING: float = 5.5
    DEFAULT_LAMBDA_MMR: float = 0.50

    def __init__(
        self,
        storage_manager: Optional[StorageManager] = None,
        default_eta: float = DEFAULT_ETA,
        default_gamma: float = DEFAULT_GAMMA,
        default_alpha: float = DEFAULT_ALPHA,
        neutral_pivot: float = NEUTRAL_RATING,
        epsilon: float = EPSILON,
    ) -> None:
        """
        Initializes the learning engine with optional storage manager and hyperparameter overrides.
        """
        self.storage_manager = storage_manager
        self.default_eta = float(default_eta)
        self.default_gamma = float(default_gamma)
        self.default_alpha = float(default_alpha)
        self.neutral_pivot = float(neutral_pivot)
        self.epsilon = float(epsilon)

    # =========================================================================
    # 1. Low-Level Vector Mathematics (Stateless & NumPy-Based)
    # =========================================================================

    @classmethod
    def rating_gradient_delta(
        cls,
        rating: float,
        item_vec: np.ndarray,
        eta: float = DEFAULT_ETA,
        neutral_pivot: float = NEUTRAL_RATING,
    ) -> np.ndarray:
        """
        Computes Delta u_nota = eta * (R - 5.5) * hat{v}_i.
        Defensively clamps rating to [1.0, 10.0].
        """
        try:
            r = float(rating)
            if math.isnan(r) or np.isnan(r):
                r = neutral_pivot
            else:
                r = max(1.0, min(10.0, r))
        except (ValueError, TypeError):
            r = neutral_pivot

        return eta * (r - neutral_pivot) * item_vec

    @classmethod
    def apply_temporal_decay(
        cls,
        user_vec: np.ndarray,
        gamma: float = DEFAULT_GAMMA,
    ) -> np.ndarray:
        """
        Computes u_decayed = gamma * u_{t-1}.
        """
        return gamma * user_vec

    @classmethod
    def project_l2_sphere(
        cls,
        vec: np.ndarray,
        epsilon: float = EPSILON,
    ) -> np.ndarray:
        """
        Projects vector onto unit hypersphere: hat{u} = u / (||u||_2 + eps).
        If norm is below 1e-12 (cold start or total cancellation), returns zero vector.
        Guarantees | ||u||_2 - 1.0 | < 1e-6 for all non-zero inputs.
        """
        norm = float(np.linalg.norm(vec))
        if norm < 1e-12:
            return np.zeros_like(vec)
        return vec / (norm + epsilon)

    @classmethod
    def semantic_boost_delta(
        cls,
        identified_tags: Sequence[Union[SemanticTagDelta, Dict[str, Any]]],
        vocabulary: List[str],
        alpha: float = DEFAULT_ALPHA,
    ) -> np.ndarray:
        """
        Computes Delta u_gemini = sum_k (delta_k * c_k * alpha) over dense vocabulary.
        Clamps delta_k in [-1.0, 1.0], c_k in [0.0, 1.0]. Unknown tags are ignored.
        """
        boost = np.zeros(len(vocabulary), dtype=np.float64)
        vocab_index = {term: idx for idx, term in enumerate(vocabulary)}

        if hasattr(identified_tags, "identified_tags"):
            deltas_list = identified_tags.identified_tags
        elif hasattr(identified_tags, "affected_tags"):
            deltas_list = identified_tags.affected_tags
        else:
            deltas_list = identified_tags

        for item in deltas_list:
            if isinstance(item, dict):
                tag_name = item.get("tag_name")
                delta = float(item.get("impact_multiplier", 0.0))
                conf = float(item.get("confidence", 1.0))
            else:
                tag_name = getattr(item, "tag_name", "")
                delta = float(getattr(item, "impact_multiplier", 0.0))
                conf = float(getattr(item, "confidence", 1.0))

            if tag_name and tag_name in vocab_index:
                # Security Layer 4: Defensive clamping
                delta_clamped = max(-1.0, min(1.0, delta))
                conf_clamped = max(0.0, min(1.0, conf))
                boost[vocab_index[tag_name]] += delta_clamped * conf_clamped * alpha

        return boost

    @classmethod
    def update_vector(
        cls,
        user_vec: np.ndarray,
        item_vec: np.ndarray,
        rating: float,
        eta: float = DEFAULT_ETA,
        gamma: float = DEFAULT_GAMMA,
        neutral_pivot: float = NEUTRAL_RATING,
        gemini_boost: Optional[np.ndarray] = None,
        epsilon: float = EPSILON,
    ) -> np.ndarray:
        """
        Executes complete 3-step vector update cycle:
        1. u_decayed = gamma * u_t
        2. u' = u_decayed + Delta u_nota + Delta u_gemini
        3. u_{t+1} = u' / (||u'||_2 + eps)
        """
        u_decayed = cls.apply_temporal_decay(user_vec, gamma=gamma)
        delta_nota = cls.rating_gradient_delta(rating, item_vec, eta=eta, neutral_pivot=neutral_pivot)

        u_prime = u_decayed + delta_nota
        if gemini_boost is not None:
            u_prime = u_prime + gemini_boost

        return cls.project_l2_sphere(u_prime, epsilon=epsilon)

    @classmethod
    def cosine_similarity(
        cls,
        vec_a: np.ndarray,
        vec_b: np.ndarray,
    ) -> float:
        """
        Computes cosine similarity Sim(a, b) = dot(a, b) for unit-normalized vectors.
        Strictly clamped to [-1.0, 1.0].
        """
        val = float(np.dot(vec_a, vec_b))
        if np.isnan(val) or math.isnan(val):
            return 0.0
        return max(-1.0, min(1.0, val))

    @classmethod
    def compute_mmr(
        cls,
        user_vec: np.ndarray,
        candidate_vecs: Sequence[np.ndarray],
        already_selected_vecs: Union[Sequence[np.ndarray], Sequence[int]],
        lambda_param: float = DEFAULT_LAMBDA_MMR,
    ) -> int:
        """
        Computes Maximal Marginal Relevance selection index:
        MMR(i) = argmax [ lambda * Sim(u, v_i) - (1 - lambda) * max_{j in S} Sim(v_i, v_j) ]
        Returns index in candidate_vecs.
        Supports already_selected_vecs as either vectors or list of integer indices.
        """
        best_score = -float("inf")
        best_idx = 0

        # Handle selected items as either vectors or indices
        selected_vectors: List[np.ndarray] = []
        if already_selected_vecs:
            first_item = already_selected_vecs[0]
            if isinstance(first_item, (int, np.integer)):
                selected_vectors = [candidate_vecs[i] for i in already_selected_vecs]  # type: ignore
            else:
                selected_vectors = list(already_selected_vecs)  # type: ignore

        for idx, cand_vec in enumerate(candidate_vecs):
            sim_user = cls.cosine_similarity(user_vec, cand_vec)
            if not selected_vectors:
                max_sim_item = 0.0
            else:
                max_sim_item = max(
                    cls.cosine_similarity(cand_vec, s_vec) for s_vec in selected_vectors
                )

            score = lambda_param * sim_user - (1.0 - lambda_param) * max_sim_item
            if score > best_score:
                best_score = score
                best_idx = idx

        return best_idx

    # =========================================================================
    # 2. Sparse Dictionary Mathematics (Metadata-Agnostic & Zero-Pad Free)
    # =========================================================================

    @classmethod
    def extract_item_feature_weights(cls, anime: AnimeEntity) -> Dict[str, float]:
        """
        Extracts raw feature weights from AnimeEntity and projects to unit L2 norm:
        - Genres: 1.0
        - Tags (including spoiler tags per R3): rank / 100.0
        Returns dictionary of normalized weights mapping feature_name -> weight.
        """
        raw_weights: Dict[str, float] = {}

        # 1. Genres (weight 1.0)
        for g in anime.genres:
            g_clean = str(g).strip()
            if g_clean:
                raw_weights[g_clean] = 1.0

        # 2. Tags (weight rank / 100.0, including is_media_spoiler: True tags per R3)
        for tag in anime.tags:
            tag_name = tag.name.strip() if hasattr(tag, "name") else str(tag.get("name", "")).strip()
            rank = tag.rank if hasattr(tag, "rank") else tag.get("rank", 0)
            if tag_name:
                raw_weights[tag_name] = float(rank) / 100.0

        # L2 spherical projection
        norm = math.sqrt(sum(w * w for w in raw_weights.values()))
        if norm < 1e-12:
            return {}

        return {k: v / (norm + cls.EPSILON) for k, v in raw_weights.items()}

    @classmethod
    def update_feature_weights(
        cls,
        current_weights: Dict[str, float],
        item_weights: Dict[str, float],
        rating: float,
        eta: float = DEFAULT_ETA,
        gamma: float = DEFAULT_GAMMA,
        neutral_pivot: float = NEUTRAL_RATING,
        gemini_deltas: Optional[Sequence[Union[SemanticTagDelta, Dict[str, Any]]]] = None,
        alpha: float = DEFAULT_ALPHA,
        epsilon: float = EPSILON,
    ) -> Dict[str, float]:
        """
        Performs continuous online preference update directly over sparse dictionary.
        Mathematically equivalent to dense vector update:
        1. u_decayed = gamma * current_weights
        2. delta_nota = eta * (R - 5.5) * item_weights
        3. delta_gemini = sum_k (delta_k * conf_k * alpha)
        4. u_{t+1} = u' / (||u'||_2 + eps)
        Taste inversion: negative weights are NOT clamped to 0.0.
        """
        # Defensive rating clamping
        try:
            r = float(rating)
            if math.isnan(r) or np.isnan(r):
                r = neutral_pivot
            else:
                r = max(1.0, min(10.0, r))
        except (ValueError, TypeError):
            r = neutral_pivot

        rating_factor = eta * (r - neutral_pivot)

        # 1. Apply decay to current weights
        decayed: Dict[str, float] = {k: v * gamma for k, v in current_weights.items()}

        # 2. Add rating gradient delta
        for k, v in item_weights.items():
            decayed[k] = decayed.get(k, 0.0) + rating_factor * v

        # 3. Add Gemini semantic boost delta (if provided)
        if gemini_deltas:
            if hasattr(gemini_deltas, "identified_tags"):
                deltas_list = gemini_deltas.identified_tags
            elif hasattr(gemini_deltas, "affected_tags"):
                deltas_list = gemini_deltas.affected_tags
            else:
                deltas_list = gemini_deltas

            for item in deltas_list:
                if isinstance(item, dict):
                    tag_name = item.get("tag_name")
                    delta = float(item.get("impact_multiplier", 0.0))
                    conf = float(item.get("confidence", 1.0))
                else:
                    tag_name = getattr(item, "tag_name", "")
                    delta = float(getattr(item, "impact_multiplier", 0.0))
                    conf = float(getattr(item, "confidence", 1.0))

                if tag_name:
                    delta_clamped = max(-1.0, min(1.0, delta))
                    conf_clamped = max(0.0, min(1.0, conf))
                    boost = delta_clamped * conf_clamped * alpha
                    decayed[tag_name] = decayed.get(tag_name, 0.0) + boost

        # 4. Mandatory L2 spherical projection
        sq_sum = sum(w * w for w in decayed.values())
        norm = math.sqrt(sq_sum)

        if norm < 1e-12:
            return {}

        return {k: float(w / (norm + epsilon)) for k, w in decayed.items() if abs(w) > 1e-12}

    # =========================================================================
    # 3. Domain Model Operations (UserProfileData Functional Transformations)
    # =========================================================================

    def update_profile_rating(
        self,
        profile: UserProfileData,
        rated_anime: AnimeEntity,
        rating: float,
        qualitative_tuning: Optional[SemanticTuningResponse] = None,
    ) -> UserProfileData:
        """
        Pure-functional transformation: updates a UserProfileData model based on a rating.
        Respects custom hyperparameters stored in profile.hyperparameters.
        Increments total_ratings and updates updated_at timestamp.
        """
        hp = profile.hyperparameters or Hyperparameters()
        eta = float(getattr(hp, "eta", self.default_eta))
        gamma = float(getattr(hp, "gamma", self.default_gamma))
        alpha = float(getattr(hp, "alpha", self.default_alpha))

        item_weights = self.extract_item_feature_weights(rated_anime)
        deltas = qualitative_tuning.identified_tags if qualitative_tuning else None

        new_weights = self.update_feature_weights(
            current_weights=profile.feature_weights,
            item_weights=item_weights,
            rating=rating,
            eta=eta,
            gamma=gamma,
            neutral_pivot=self.neutral_pivot,
            gemini_deltas=deltas,
            alpha=alpha,
            epsilon=self.epsilon,
        )

        return profile.model_copy(
            update={
                "updated_at": datetime.now(timezone.utc),
                "total_ratings": profile.total_ratings + 1,
                "feature_weights": new_weights,
                "hyperparameters": hp,
            }
        )

    def apply_semantic_boost(
        self,
        profile: UserProfileData,
        tuning: SemanticTuningResponse,
    ) -> UserProfileData:
        """
        Applies a post-rating qualitative Gemini tuning update to the user profile
        without applying temporal decay or rating delta.
        """
        hp = profile.hyperparameters or Hyperparameters()
        alpha = float(getattr(hp, "alpha", self.default_alpha))

        current_weights = dict(profile.feature_weights)
        if hasattr(tuning, "identified_tags"):
            deltas_list = tuning.identified_tags
        elif hasattr(tuning, "affected_tags"):
            deltas_list = tuning.affected_tags
        else:
            deltas_list = tuning

        for delta_item in deltas_list:
            if isinstance(delta_item, dict):
                tag_name = str(delta_item.get("tag_name", "")).strip()
                delta_val = delta_item.get("impact_multiplier", 0.0)
                conf_val = delta_item.get("confidence", 1.0)
            else:
                tag_name = getattr(delta_item, "tag_name", "").strip()
                delta_val = getattr(delta_item, "impact_multiplier", 0.0)
                conf_val = getattr(delta_item, "confidence", 1.0)

            if tag_name:
                delta = max(-1.0, min(1.0, float(delta_val)))
                conf = max(0.0, min(1.0, float(conf_val)))
                boost = delta * conf * alpha
                current_weights[tag_name] = current_weights.get(tag_name, 0.0) + boost

        # Spherical projection
        norm = math.sqrt(sum(w * w for w in current_weights.values()))
        if norm < 1e-12:
            new_weights = {}
        else:
            new_weights = {k: float(w / (norm + self.epsilon)) for k, w in current_weights.items()}

        return profile.model_copy(
            update={
                "updated_at": datetime.now(timezone.utc),
                "feature_weights": new_weights,
                "hyperparameters": hp,
            }
        )

    @classmethod
    def initialize_cold_start_profile(
        cls,
        user_id: str = "default_user",
    ) -> UserProfileData:
        """
        Returns a fresh default cold-start UserProfileData instance.
        """
        return UserProfileData(
            user_id=user_id,
            updated_at=datetime.now(timezone.utc),
            total_ratings=0,
            feature_weights={},
            hyperparameters=Hyperparameters(),
        )

    # =========================================================================
    # 4. Storage-Integrated Service Layer (StorageManager Orchestration)
    # =========================================================================

    def record_rating(
        self,
        anime_id: int,
        rating: float,
        origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = "direct_search",
        user_review: Optional[str] = None,
        qualitative_tuning: Optional[SemanticTuningResponse] = None,
    ) -> Tuple[UserProfileData, WatchHistoryEntry]:
        """
        Complete end-to-end rating workflow:
        1. Loads catalog from storage and locates anime by ID.
        2. Atomically reads, updates, and saves user profile via storage_manager.update_user_profile().
        3. Appends interaction record to watch history via storage_manager.append_watch_history().
        4. Returns updated (UserProfileData, WatchHistoryEntry).
        """
        if self.storage_manager is None:
            raise RuntimeError(
                "StorageManager must be configured on LearningEngine to execute persistent record_rating."
            )

        # 1. Locate anime in catalog
        catalog = self.storage_manager.load_catalog()
        anime = next((a for a in catalog if a.id == anime_id), None)
        if anime is None:
            raise ValueError(f"Anime ID {anime_id} not found in catalog cache.")

        # Check if this anime was already evaluated in watch history
        current_history = self.storage_manager.load_watch_history()
        existing_entry = next((h for h in current_history if h.anime_id == anime_id and h.interaction_status == "watched"), None)

        # 2. Create and upsert watch history entry
        anime_title = anime.title_romaji
        if not anime_title and anime.title:
            anime_title = anime.title.romaji
        if not anime_title:
            anime_title = f"Anime #{anime.id}"

        history_entry = WatchHistoryEntry(
            id=existing_entry.id if existing_entry else f"wh-{uuid.uuid4().hex[:8]}",
            anime_id=anime.id,
            anime_title=anime_title,
            rating=rating,
            origin_track=origin_track,
            interaction_status="watched",
            user_review_raw=user_review[:300] if user_review else (existing_entry.user_review_raw if existing_entry else None),
            qualitative_tuning=qualitative_tuning or (existing_entry.qualitative_tuning if existing_entry else None),
            timestamp=datetime.now(timezone.utc),
        )
        self.storage_manager.append_watch_history(history_entry)

        # 3. Update User Profile
        if existing_entry is not None:
            # Recalibrate cleanly from cold start across all unique ratings to eliminate vector drift and duplicate weights
            updated_history = self.storage_manager.load_watch_history()
            catalog_by_id = {a.id: a for a in catalog}

            def recalibrate_modifier(prof: UserProfileData) -> UserProfileData:
                temp_prof = prof.model_copy(
                    update={
                        "feature_weights": {},
                        "total_ratings": 0,
                        "updated_at": datetime.now(timezone.utc),
                    }
                )
                for h in updated_history:
                    if h.interaction_status == "watched" and h.anime_id in catalog_by_id:
                        temp_prof = self.update_profile_rating(
                            profile=temp_prof,
                            rated_anime=catalog_by_id[h.anime_id],
                            rating=h.rating,
                            qualitative_tuning=h.qualitative_tuning,
                        )
                return temp_prof

            updated_profile = self.storage_manager.update_user_profile(recalibrate_modifier)
        else:
            # Brand-new rating: standard single-step online learning
            def modifier(current_profile: UserProfileData) -> UserProfileData:
                return self.update_profile_rating(
                    profile=current_profile,
                    rated_anime=anime,
                    rating=rating,
                    qualitative_tuning=qualitative_tuning,
                )

            updated_profile = self.storage_manager.update_user_profile(modifier)

        logger.info(
            f"Successfully recorded rating {rating:.1f} for anime '{anime_title}' (id={anime_id}, is_recalibration={existing_entry is not None}). "
            f"Profile total ratings: {updated_profile.total_ratings}, norm: {updated_profile.compute_l2_norm():.6f}"
        )

        return updated_profile, history_entry

    def record_drop(
        self,
        anime_id: int,
        drop_reason: str,
        origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = "direct_search",
        gemini_service: Optional[Any] = None,
    ) -> Tuple[UserProfileData, WatchHistoryEntry, SemanticTuningResponse]:
        """
        Records a dropped anime with written user justification analyzed by Gemini.
        1. Invokes GeminiService.analyze_drop_reason() to extract negative SemanticTagDelta values.
        2. Updates the user profile vector with negative semantic adjustments and mild item penalty,
           strictly preserving unit L2 norm invariant ||u_{t+1}||_2 = 1.0.
        3. Adds anime_id to profile.dropped_anime_ids and appends WatchHistoryEntry(interaction_status='dropped').
        """
        if self.storage_manager is None:
            raise RuntimeError("StorageManager must be configured on LearningEngine.")

        catalog = self.storage_manager.load_catalog()
        anime = next((a for a in catalog if a.id == anime_id), None)
        if anime is None:
            raise ValueError(f"Anime ID {anime_id} not found in catalog cache.")

        if gemini_service is None:
            try:
                from app.services.gemini_service import get_gemini_service
                gemini_service = get_gemini_service()
            except ImportError:
                from backend.app.services.gemini_service import get_gemini_service
                gemini_service = get_gemini_service()

        sanitized_reason = gemini_service.sanitize_user_feedback(drop_reason, max_length=300)
        current_profile = self.storage_manager.load_user_profile()

        tuning_response = gemini_service.analyze_drop_reason(
            drop_reason=sanitized_reason,
            dropped_anime=anime,
            current_profile=current_profile,
        )

        # Apply mild negative rating delta (rating=3.0) + strong targeted Gemini negative semantic boost
        def drop_modifier(prof: UserProfileData) -> UserProfileData:
            updated = self.update_profile_rating(
                profile=prof,
                rated_anime=anime,
                rating=3.0,
                qualitative_tuning=tuning_response,
            )
            dropped_ids = list(getattr(updated, "dropped_anime_ids", []))
            if anime_id not in dropped_ids:
                dropped_ids.append(anime_id)
            watchlist_ids = [aid for aid in getattr(updated, "watchlist_anime_ids", []) if aid != anime_id]
            skipped_ids = [aid for aid in getattr(updated, "skipped_anime_ids", []) if aid != anime_id]
            return updated.model_copy(
                update={
                    "dropped_anime_ids": dropped_ids,
                    "watchlist_anime_ids": watchlist_ids,
                    "skipped_anime_ids": skipped_ids,
                }
            )

        updated_profile = self.storage_manager.update_user_profile(drop_modifier)

        anime_title = anime.title_romaji or (anime.title.romaji if anime.title else f"Anime #{anime.id}")
        history_entry = WatchHistoryEntry(
            id=f"wh-{uuid.uuid4().hex[:8]}",
            anime_id=anime.id,
            anime_title=anime_title,
            rating=2.0,
            origin_track=origin_track,
            interaction_status="dropped",
            drop_reason=sanitized_reason,
            user_review_raw=sanitized_reason,
            qualitative_tuning=tuning_response,
            timestamp=datetime.now(timezone.utc),
        )
        self.storage_manager.append_watch_history(history_entry)

        return updated_profile, history_entry, tuning_response

    def tune_and_record_rating(
        self,
        anime_id: int,
        rating: float,
        user_review: Optional[str] = None,
        origin_track: Literal["affinity", "discovery", "direct_search", "onboarding"] = "direct_search",
        gemini_service: Optional[Any] = None,
        spoiler_service: Optional[Any] = None,
    ) -> Tuple[UserProfileData, WatchHistoryEntry, Optional[SemanticTuningResponse]]:
        """
        End-to-end orchestration helper for simultaneous rating and qualitative semantic tuning.
        """
        if self.storage_manager is None:
            raise RuntimeError("StorageManager must be configured on LearningEngine.")

        catalog = self.storage_manager.load_catalog()
        anime = next((a for a in catalog if a.id == anime_id), None)
        if anime is None:
            raise ValueError(f"Anime ID {anime_id} not found in catalog cache.")

        tuning_response: Optional[SemanticTuningResponse] = None
        sanitized_review: Optional[str] = None

        if user_review and user_review.strip():
            sanitized_review = user_review.strip()[:300]

            if gemini_service is None:
                try:
                    from app.services.gemini_service import get_gemini_service
                    gemini_service = get_gemini_service()
                except ImportError:
                    from backend.app.services.gemini_service import get_gemini_service
                    gemini_service = get_gemini_service()

            current_profile = self.storage_manager.load_user_profile()

            try:
                tuning_response = gemini_service.tune_profile(
                    user_feedback=sanitized_review,
                    rated_anime=anime,
                    current_profile=current_profile,
                    rating=rating,
                )
            except Exception as exc:
                logger.warning(f"Gemini live tuning failed ({exc}); falling back to offline analysis.")
                if hasattr(gemini_service, "get_fallback_tuning_response"):
                    tuning_response = gemini_service.get_fallback_tuning_response(sanitized_review, anime)
                else:
                    tuning_response = None

            if tuning_response and spoiler_service is not None:
                tuning_response.public_safe_reason = spoiler_service.sanitize_reason(
                    tuning_response.public_safe_reason
                )

        def modifier(current_profile: UserProfileData) -> UserProfileData:
            return self.update_profile_rating(
                profile=current_profile,
                rated_anime=anime,
                rating=rating,
                qualitative_tuning=tuning_response,
            )

        updated_profile = self.storage_manager.update_user_profile(modifier)

        anime_title = anime.title_romaji
        if not anime_title and anime.title:
            anime_title = anime.title.romaji
        if not anime_title:
            anime_title = f"Anime #{anime.id}"

        history_entry = WatchHistoryEntry(
            id=f"wh-{uuid.uuid4().hex[:8]}",
            anime_id=anime.id,
            anime_title=anime_title,
            rating=max(1.0, min(10.0, float(rating))),
            origin_track=origin_track,
            user_review_raw=sanitized_review,
            qualitative_tuning=tuning_response,
            timestamp=datetime.now(timezone.utc),
        )
        self.storage_manager.append_watch_history(history_entry)

        return updated_profile, history_entry, tuning_response

    def tune_existing_interaction(
        self,
        anime_id: int,
        user_feedback: str,
        gemini_service: Optional[Any] = None,
        spoiler_service: Optional[Any] = None,
    ) -> Tuple[UserProfileData, Optional[WatchHistoryEntry], SemanticTuningResponse]:
        """
        Applies qualitative Gemini semantic tuning to an existing anime interaction,
        updating user preference weights without decay or rating delta, and enriching
        the most recent watch history entry.
        """
        if self.storage_manager is None:
            raise RuntimeError("StorageManager must be configured on LearningEngine.")

        catalog = self.storage_manager.load_catalog()
        anime = next((a for a in catalog if a.id == anime_id), None)
        if anime is None:
            raise ValueError(f"Anime ID {anime_id} not found in catalog cache.")

        sanitized_feedback = user_feedback.strip()[:300]

        if gemini_service is None:
            try:
                from app.services.gemini_service import get_gemini_service
                gemini_service = get_gemini_service()
            except ImportError:
                from backend.app.services.gemini_service import get_gemini_service
                gemini_service = get_gemini_service()

        current_profile = self.storage_manager.load_user_profile()

        try:
            tuning_response = gemini_service.tune_profile(
                user_feedback=sanitized_feedback,
                rated_anime=anime,
                current_profile=current_profile,
            )
        except Exception as exc:
            logger.warning(f"Gemini tuning error ({exc}); falling back to offline preset.")
            if hasattr(gemini_service, "get_fallback_tuning_response"):
                tuning_response = gemini_service.get_fallback_tuning_response(sanitized_feedback, anime)
            else:
                raise

        if spoiler_service is not None:
            tuning_response.public_safe_reason = spoiler_service.sanitize_reason(
                tuning_response.public_safe_reason
            )

        def boost_modifier(prof: UserProfileData) -> UserProfileData:
            return self.apply_semantic_boost(profile=prof, tuning=tuning_response)

        updated_profile = self.storage_manager.update_user_profile(boost_modifier)

        enriched_entry = self.storage_manager.enrich_latest_watch_history(
            anime_id=anime_id,
            user_review_raw=sanitized_feedback,
            qualitative_tuning=tuning_response,
        )

        return updated_profile, enriched_entry, tuning_response

    def get_user_profile(self) -> UserProfileData:
        """Retrieves user profile from persistent storage."""
        if self.storage_manager is None:
            raise RuntimeError("StorageManager is not configured.")
        return self.storage_manager.load_user_profile()

    def get_watch_history(self) -> List[WatchHistoryEntry]:
        """Retrieves watch history from persistent storage."""
        if self.storage_manager is None:
            raise RuntimeError("StorageManager is not configured.")
        return self.storage_manager.load_watch_history()

    def reset_profile(self) -> UserProfileData:
        """Resets user profile to cold start state atomically."""
        if self.storage_manager is None:
            raise RuntimeError("StorageManager is not configured.")
        return self.storage_manager.reset_user_profile()

    def reset_all_user_data(self) -> Tuple[UserProfileData, List[WatchHistoryEntry]]:
        """Resets both user profile and watch history atomically."""
        if self.storage_manager is None:
            raise RuntimeError("StorageManager is not configured.")
        return self.storage_manager.reset_all_user_data()
