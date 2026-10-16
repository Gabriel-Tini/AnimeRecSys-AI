"""
AnimeRecSys AI — Pytest Configuration, Shared Fixtures & Mathematical Oracles
Authoritative Reference: .antigravity/ARCHITECTURE.md, FIXTURES.json, SECURITY.md
"""

import json
import os
import shutil
import tempfile
import importlib
from typing import Any, Dict, List, Optional, Set
import numpy as np
import pytest

# Paths
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURES_PATH = os.path.join(ROOT_DIR, ".antigravity", "FIXTURES.json")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# ---------------------------------------------------------------------------
# Fixture: Canonical Specification Fixtures Data (.antigravity/FIXTURES.json)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def fixtures_data() -> Dict[str, Any]:
    """Loads canonical specification fixtures containing vocabulary, sample animes, mock responses."""
    assert os.path.exists(FIXTURES_PATH), f"FIXTURES.json missing at {FIXTURES_PATH}"
    with open(FIXTURES_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="session")
def tags_vocabulary(fixtures_data) -> List[Dict[str, Any]]:
    return fixtures_data["tags_vocabulary"]


@pytest.fixture(scope="session")
def sample_animes(fixtures_data) -> List[Dict[str, Any]]:
    return fixtures_data["sample_animes_full"]


@pytest.fixture(scope="session")
def mock_user_profile(fixtures_data) -> Dict[str, Any]:
    return fixtures_data["mock_user_profile"]


@pytest.fixture(scope="session")
def mock_watch_history(fixtures_data) -> Dict[str, Any]:
    return fixtures_data["mock_watch_history"]


@pytest.fixture(scope="session")
def mock_gemini_response(fixtures_data) -> Dict[str, Any]:
    return fixtures_data["mock_gemini_response"]


@pytest.fixture(scope="session")
def mock_recommendation_response(fixtures_data) -> Dict[str, Any]:
    return fixtures_data["mock_recommendation_response"]


# ---------------------------------------------------------------------------
# Fixture: Isolated Temporary Data Directory (Concurrency & Isolation)
# ---------------------------------------------------------------------------
@pytest.fixture
def temp_data_dir():
    """
    Creates an isolated sandbox copy of the data directory for each test run.
    Ensures tests are self-contained and never mutate workspace data files.
    """
    tmp_dir = tempfile.mkdtemp(prefix="animerecsys_test_data_")
    
    # Copy template data files if they exist
    for fname in ["cache_catalog.json", "user_profile.json", "watch_history.json"]:
        src = os.path.join(DATA_DIR, fname)
        dst = os.path.join(tmp_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
        else:
            # Seed defaults
            if fname == "cache_catalog.json":
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump([], f)
            elif fname == "user_profile.json":
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump({
                        "user_id": "default_user",
                        "updated_at": "2026-10-15T00:00:00Z",
                        "total_ratings": 0,
                        "feature_weights": {},
                        "hyperparameters": {"eta": 0.15, "alpha": 0.25, "gamma": 0.98, "lambda_mmr": 0.5}
                    }, f)
            elif fname == "watch_history.json":
                with open(dst, "w", encoding="utf-8") as f:
                    json.dump([], f)

    yield tmp_dir
    shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Module Resolver Helper: Dynamic Backend Component Discovery
# ---------------------------------------------------------------------------
def resolve_backend_module(module_name: str) -> Optional[Any]:
    """
    Attempts to import a backend module across valid project layouts
    (PROJECT.md services/ vs ARCHITECTURE.md core/ vs root backend/).
    """
    search_paths = [
        f"backend.app.services.{module_name}",
        f"backend.app.core.{module_name}",
        f"backend.app.models.{module_name}",
        f"backend.app.schemas.{module_name}",
        f"backend.app.security.{module_name}",
        f"backend.app.integrations.{module_name}",
        f"backend.app.api.{module_name}",
        f"backend.app.{module_name}",
        f"backend.{module_name}",
    ]
    for path in search_paths:
        try:
            return importlib.import_module(path)
        except ImportError:
            continue
    return None


@pytest.fixture
def module_resolver():
    return resolve_backend_module


# ---------------------------------------------------------------------------
# Mathematical Oracle Reference Implementations (Authoritative Specifications)
# Directly derived from ARCHITECTURE.md § 2 and ORIGINAL_REQUEST.md
# ---------------------------------------------------------------------------
class MathematicalOracle:
    """
    Authoritative reference implementation of the system's mathematical laws.
    Used to verify implementation correctness and provide expected values.
    """
    EPSILON = 1e-9
    ETA = 0.15          # Quantitative rating learning rate
    GAMMA = 0.98        # Temporal weight decay
    ALPHA = 0.25        # Qualitative Gemini semantic boost sensitivity
    NEUTRAL_RATING = 5.5
    LAMBDA_MMR = 0.50

    @classmethod
    def build_feature_vector(
        cls, 
        genres: List[str], 
        tags: List[Dict[str, Any]], 
        vocabulary: List[str]
    ) -> np.ndarray:
        """
        Calculates raw feature vector v_i and normalizes to unit sphere:
        v_{i, t} = 1.0 (if genre), rank/100.0 (if tag), 0.0 otherwise.
        hat{v}_i = v_i / (||v_i||_2 + eps)
        """
        v = np.zeros(len(vocabulary), dtype=np.float64)
        vocab_index = {term: idx for idx, term in enumerate(vocabulary)}

        for g in genres:
            if g in vocab_index:
                v[vocab_index[g]] = 1.0

        for t in tags:
            tag_name = t.get("name") if isinstance(t, dict) else getattr(t, "name", "")
            rank = t.get("rank", 0) if isinstance(t, dict) else getattr(t, "rank", 0)
            if tag_name in vocab_index:
                # Tags contribute rank / 100.0
                v[vocab_index[tag_name]] = float(rank) / 100.0

        norm = np.linalg.norm(v)
        return v / (norm + cls.EPSILON)

    @classmethod
    def l2_normalize(cls, vec: np.ndarray) -> np.ndarray:
        """Projects vector onto unit hypersphere with epsilon protection."""
        norm = np.linalg.norm(vec)
        return vec / (norm + cls.EPSILON)

    @classmethod
    def rating_gradient_delta(cls, rating: float, item_vec: np.ndarray) -> np.ndarray:
        """Delta u_nota = eta * (R - 5.5) * hat{v}_i"""
        return cls.ETA * (rating - cls.NEUTRAL_RATING) * item_vec

    @classmethod
    def temporal_decay(cls, profile_vec: np.ndarray) -> np.ndarray:
        """u_decayed = gamma * u_{t-1}"""
        return cls.GAMMA * profile_vec

    @classmethod
    def semantic_boost_delta(
        cls, 
        identified_tags: List[Dict[str, Any]], 
        vocabulary: List[str]
    ) -> np.ndarray:
        """
        Delta u_gemini, t = sum_{k | tag_k = t} (delta_k * c_k * alpha)
        Clamped to delta in [-1.0, 1.0], c in [0.0, 1.0]
        """
        boost = np.zeros(len(vocabulary), dtype=np.float64)
        vocab_index = {term: idx for idx, term in enumerate(vocabulary)}

        for item in identified_tags:
            tag_name = item.get("tag_name")
            if tag_name in vocab_index:
                delta = float(item.get("impact_multiplier", 0.0))
                delta = max(-1.0, min(1.0, delta))
                conf = float(item.get("confidence", 1.0))
                conf = max(0.0, min(1.0, conf))
                boost[vocab_index[tag_name]] += delta * conf * cls.ALPHA

        return boost

    @classmethod
    def update_user_profile(
        cls,
        user_vec: np.ndarray,
        rated_item_vec: np.ndarray,
        rating: float,
        identified_tags: Optional[List[Dict[str, Any]]] = None,
        vocabulary: Optional[List[str]] = None
    ) -> np.ndarray:
        """
        Full 3-step mathematical update cycle:
        1. u_decayed = gamma * u_{t-1}
        2. u' = u_decayed + Delta u_nota + Delta u_gemini
        3. u_t = u' / (||u'||_2 + eps)
        """
        # 1. Decay
        u_decayed = cls.temporal_decay(user_vec)

        # 2. Delta nota
        delta_nota = cls.rating_gradient_delta(rating, rated_item_vec)

        # Delta gemini
        delta_gemini = np.zeros_like(user_vec)
        if identified_tags and vocabulary:
            delta_gemini = cls.semantic_boost_delta(identified_tags, vocabulary)

        u_prime = u_decayed + delta_nota + delta_gemini

        # 3. L2 Unit Normalization
        return cls.l2_normalize(u_prime)

    @classmethod
    def cosine_similarity(cls, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Sim(u, v) = u . v (for L2-normalized unit vectors)."""
        return float(np.dot(vec_a, vec_b))

    @classmethod
    def compute_mmr(
        cls,
        user_vec: np.ndarray,
        candidate_vecs: List[np.ndarray],
        already_selected_vecs: List[np.ndarray],
        lambda_param: float = 0.50
    ) -> int:
        """
        MMR(i) = argmax [ lambda * Sim(u, v_i) - (1 - lambda) * max_{j in S} Sim(v_i, v_j) ]
        Returns index in candidate_vecs.
        """
        best_score = -float("inf")
        best_idx = 0

        for idx, cand_vec in enumerate(candidate_vecs):
            sim_user = cls.cosine_similarity(user_vec, cand_vec)
            if not already_selected_vecs:
                max_sim_item = 0.0
            else:
                max_sim_item = max(
                    cls.cosine_similarity(cand_vec, s_vec) for s_vec in already_selected_vecs
                )

            mmr_score = lambda_param * sim_user - (1.0 - lambda_param) * max_sim_item
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        return best_idx

    @classmethod
    def dual_track_partition(
        cls,
        user_vec: np.ndarray,
        candidates: List[Dict[str, Any]],
        candidate_vecs: List[np.ndarray],
        top_n: int = 10
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        80% Affinity (Exploitation) vs 20% Discovery (MMR Exploration).
        Guarantees S_aff intersection S_disc is empty.
        """
        k_aff = int(np.floor(0.8 * top_n))
        k_disc = top_n - k_aff

        # Rank candidates by cosine similarity
        similarities = [cls.cosine_similarity(user_vec, v) for v in candidate_vecs]
        ranked_indices = sorted(range(len(candidates)), key=lambda i: similarities[i], reverse=True)

        affinity_indices = ranked_indices[:k_aff]
        affinity_items = [candidates[i] for i in affinity_indices]
        affinity_vecs = [candidate_vecs[i] for i in affinity_indices]

        # Remaining candidates for discovery track
        remaining_indices = ranked_indices[k_aff:]
        discovery_indices: List[int] = []
        selected_for_mmr_vecs: List[np.ndarray] = list(affinity_vecs)

        for _ in range(min(k_disc, len(remaining_indices))):
            cand_pool_vecs = [candidate_vecs[i] for i in remaining_indices if i not in discovery_indices]
            cand_pool_actual_indices = [i for i in remaining_indices if i not in discovery_indices]
            if not cand_pool_actual_indices:
                break
            best_pool_idx = cls.compute_mmr(
                user_vec, cand_pool_vecs, selected_for_mmr_vecs, cls.LAMBDA_MMR
            )
            selected_actual_idx = cand_pool_actual_indices[best_pool_idx]
            discovery_indices.append(selected_actual_idx)
            selected_for_mmr_vecs.append(candidate_vecs[selected_actual_idx])

        discovery_items = [candidates[i] for i in discovery_indices]

        return {
            "affinity_track": affinity_items,
            "discovery_track": discovery_items
        }

    @classmethod
    def filter_spoilers_from_tags(cls, tags: List[Dict[str, Any]]) -> List[str]:
        """Filters out tags where is_media_spoiler == True."""
        safe_tags = []
        for t in tags:
            is_spoiler = t.get("is_media_spoiler", False) if isinstance(t, dict) else getattr(t, "is_media_spoiler", False)
            if not is_spoiler:
                name = t.get("name") if isinstance(t, dict) else getattr(t, "name", "")
                if name:
                    safe_tags.append(name)
        return safe_tags


@pytest.fixture(scope="session")
def oracle():
    return MathematicalOracle
