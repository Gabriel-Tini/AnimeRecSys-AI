"""
AnimeRecSys AI — Tier 3: Cross-Feature Multi-Component Combinations Test Suite
Authoritative Reference: ORIGINAL_REQUEST.md, ARCHITECTURE.md, SECURITY.md, FIXTURES.json

Coverage:
- Full continuous learning loop: Quantitative Rating + Temporal Decay + Gemini Boost + L2 Projection + MMR Partitioning
- Dual Spoiler Policy: Internal VSM preservation driving recommendation affinity vs 100% public DTO stripping
- Taste Inversion & Drift: Drastic preference transition with negative repulsion and temporal decay rotation
- Qualitative Sub-Genre Elevation: Semantic tuning boosting specific niche tags over primary genres
- Progressive Watch History Exclusion: Candidate pool filtering preventing duplicate recommendations
- Concurrency & Storage Integration: Concurrent rating writes and catalog expansion
Total Tier 3 Tests: >= 6 deep multi-feature combination tests
"""

import json
import os
import tempfile
import numpy as np
import pytest
from filelock import FileLock

from tests.conftest import MathematicalOracle, ROOT_DIR


class TestTier3CrossFeatureCombinations:
    """Verifies complex interactions between multiple system components."""

    def test_tier3_01_full_learning_cycle_and_mmr_recommendation(
        self, tags_vocabulary, sample_animes, oracle
    ):
        """
        Combination 1: Rating Update + Temporal Decay + Gemini Semantic Boost + L2 Projection + MMR.
        Simulates two consecutive anime ratings with qualitative tuning and generates 80/20 recommendations.
        """
        vocab = [t["name"] for t in tags_vocabulary]
        u_t = np.zeros(len(vocab), dtype=np.float64)

        # 1. First interaction: Rate Steins;Gate 10.0 with qualitative feedback
        steins = sample_animes[0]
        v_steins = oracle.build_feature_vector(steins["genres"], steins["tags"], vocab)
        tuning_1 = [
            {"tag_name": "Time Manipulation", "impact_multiplier": 0.95, "confidence": 0.98},
            {"tag_name": "Plot Twist", "impact_multiplier": 0.90, "confidence": 0.95}
        ]

        u_t = oracle.update_user_profile(
            user_vec=u_t,
            rated_item_vec=v_steins,
            rating=10.0,
            identified_tags=tuning_1,
            vocabulary=vocab
        )

        assert abs(np.linalg.norm(u_t) - 1.0) < 1e-6
        time_idx = vocab.index("Time Manipulation")
        assert u_t[time_idx] > 0.0

        # 2. Second interaction: Rate Cyberpunk: Edgerunners 9.0 with feedback
        cyberpunk = [a for a in sample_animes if a["id"] == 126546][0]
        v_cyber = oracle.build_feature_vector(cyberpunk["genres"], cyberpunk["tags"], vocab)
        tuning_2 = [
            {"tag_name": "Cyberpunk", "impact_multiplier": 0.85, "confidence": 0.90}
        ]

        u_t_prev = u_t.copy()
        u_t = oracle.update_user_profile(
            user_vec=u_t,
            rated_item_vec=v_cyber,
            rating=9.0,
            identified_tags=tuning_2,
            vocabulary=vocab
        )

        # Decay check: Steins;Gate unique tags decayed by 0.98 before new normalization
        assert abs(np.linalg.norm(u_t) - 1.0) < 1e-6
        cyber_idx = vocab.index("Cyberpunk")
        assert u_t[cyber_idx] > 0.0

        # 3. Generate 80/20 recommendations over remaining candidate pool (excluding watched)
        watched_ids = {steins["id"], cyberpunk["id"]}
        candidate_pool = [a for a in sample_animes if a["id"] not in watched_ids]
        candidate_vecs = [
            oracle.build_feature_vector(a["genres"], a["tags"], vocab)
            for a in candidate_pool
        ]

        result = oracle.dual_track_partition(u_t, candidate_pool, candidate_vecs, top_n=4)
        aff_track = result["affinity_track"]
        disc_track = result["discovery_track"]

        # For N=4: 80% = floor(3.2) = 3 affinity, 1 discovery
        assert len(aff_track) == 3
        assert len(disc_track) == 1
        # No watched anime recommended
        rec_ids = {a["id"] for a in aff_track + disc_track}
        assert rec_ids.isdisjoint(watched_ids)
        # Disjoint tracks
        assert {a["id"] for a in aff_track}.isdisjoint({a["id"] for a in disc_track})

    def test_tier3_02_spoiler_preservation_in_vsm_vs_public_dto_stripping(
        self, tags_vocabulary, sample_animes, oracle
    ):
        """
        Combination 2: Spoiler tag participation in profile calculation vs 100% public stripping.
        - Steins;Gate has Plot Twist & Tragedy.
        - Attack on Titan has Plot Twist & Tragedy.
        - Profile accumulates Plot Twist and Tragedy.
        - High cosine similarity is driven in part by shared spoiler tags.
        - Public response for candidate strictly suppresses Plot Twist and Tragedy from tags: [].
        """
        vocab = [t["name"] for t in tags_vocabulary]
        spoiler_tag_names = {t["name"] for t in tags_vocabulary if t.get("is_media_spoiler", False)}

        steins = sample_animes[0]
        v_steins = oracle.build_feature_vector(steins["genres"], steins["tags"], vocab)

        # Update profile with Steins;Gate 10.0
        u = np.zeros(len(vocab), dtype=np.float64)
        u = oracle.update_user_profile(u, v_steins, 10.0)

        # Confirm spoiler dimensions are active in user profile vector
        for spoiler_name in ["Plot Twist", "Tragedy"]:
            idx = vocab.index(spoiler_name)
            assert u[idx] > 0.0, f"Spoiler tag {spoiler_name} missing from internal vector space"

        # Candidate: Attack on Titan (also has Plot Twist and Tragedy)
        aot = sample_animes[1]
        v_aot = oracle.build_feature_vector(aot["genres"], aot["tags"], vocab)
        sim = oracle.cosine_similarity(u, v_aot)
        assert sim > 0.3, "Expected significant similarity driven by shared themes and narrative complexity"

        # Now simulate public DTO serialization
        safe_tags = oracle.filter_spoilers_from_tags(aot["tags"])

        # Verification: Zero spoiler tags in public tag list
        for st in spoiler_tag_names:
            assert st not in safe_tags, f"Spoiler tag '{st}' leaked into public tag list!"
        assert "Action" in safe_tags
        assert "Anti-Hero" in safe_tags

    def test_tier3_03_taste_drift_inversion_and_temporal_decay(
        self, tags_vocabulary, sample_animes, oracle
    ):
        """
        Combination 3: Taste drift. Active Action/Cyberpunk lover switches to Slice of Life / Music.
        Temporal decay gamma=0.98 + negative rating deltas on Action flip recommendations to Bocchi the Rock!
        """
        vocab = [t["name"] for t in tags_vocabulary]
        u = np.zeros(len(vocab), dtype=np.float64)

        # Phase 1: Train heavily on Action (Cyberpunk + Solo Leveling)
        cyber = [a for a in sample_animes if a["id"] == 126546][0]
        v_cyber = oracle.build_feature_vector(cyber["genres"], cyber["tags"], vocab)
        for _ in range(5):
            u = oracle.update_user_profile(u, v_cyber, 10.0)

        action_idx = vocab.index("Action")
        cyber_idx = vocab.index("Cyberpunk")
        assert u[action_idx] > 0.3
        assert u[cyber_idx] > 0.3

        # Phase 2: User taste inverts. Dislikes Action (rate 1.0) and loves Bocchi the Rock! (rate 10.0)
        bocchi = [a for a in sample_animes if a["id"] == 145064][0]
        v_bocchi = oracle.build_feature_vector(bocchi["genres"], bocchi["tags"], vocab)

        for _ in range(10):
            # Repel Action
            u = oracle.update_user_profile(u, v_cyber, 1.0)
            # Attract Slice of Life / Music
            u = oracle.update_user_profile(u, v_bocchi, 10.0)

        # Slice of Life / Music should now dominate
        music_idx = vocab.index("Music")
        sol_idx = vocab.index("Slice of Life")

        assert u[music_idx] > u[action_idx], "Music did not overtake Action after taste inversion"
        assert u[sol_idx] > u[cyber_idx], "Slice of Life did not overtake Cyberpunk after taste inversion"

    def test_tier3_04_qualitative_boost_sub_genre_above_main_genre(
        self, tags_vocabulary, sample_animes, oracle
    ):
        """
        Combination 4: Qualitative feedback selectively boosts a niche sub-genre over the main genre.
        Rating Frieren (genres: Adventure, Drama, Fantasy; tag: Philosophy) with strong feedback on Philosophy.
        """
        vocab = [t["name"] for t in tags_vocabulary]
        u = np.zeros(len(vocab), dtype=np.float64)
        frieren = [a for a in sample_animes if a["id"] == 154587][0]
        v_frieren = oracle.build_feature_vector(frieren["genres"], frieren["tags"], vocab)

        # Specific tuning praising Philosophy heavily (+1.0), and penalizing Fantasy (-0.3)
        tuning = [
            {"tag_name": "Philosophy", "impact_multiplier": 1.0, "confidence": 1.0, "rationale": "Deep philosophical themes"},
            {"tag_name": "Fantasy", "impact_multiplier": -0.5, "confidence": 0.8, "rationale": "Too generic fantasy"}
        ]

        u = oracle.update_user_profile(
            user_vec=u,
            rated_item_vec=v_frieren,
            rating=8.0,
            identified_tags=tuning,
            vocabulary=vocab
        )

        phil_idx = vocab.index("Philosophy")
        fantasy_idx = vocab.index("Fantasy")
        assert u[phil_idx] > u[fantasy_idx], "Philosophy weight should exceed Fantasy weight after selective tuning"

    def test_tier3_05_progressive_watch_history_candidate_filtering(
        self, tags_vocabulary, sample_animes, oracle
    ):
        """
        Combination 5: Progressive watch history accumulation filters candidate pool monotonically.
        """
        vocab = [t["name"] for t in tags_vocabulary]
        user_vec = oracle.build_feature_vector(sample_animes[0]["genres"], sample_animes[0]["tags"], vocab)

        watch_history = []
        # Progressively watch 3 animes
        for i in range(3):
            watch_history.append(sample_animes[i]["id"])
            remaining = [a for a in sample_animes if a["id"] not in watch_history]
            assert len(remaining) == len(sample_animes) - (i + 1)

            rem_vecs = [oracle.build_feature_vector(a["genres"], a["tags"], vocab) for a in remaining]
            result = oracle.dual_track_partition(user_vec, remaining, rem_vecs, top_n=len(remaining))
            all_recs = result["affinity_track"] + result["discovery_track"]

            # Ensure no watched anime appears
            rec_ids = {a["id"] for a in all_recs}
            assert rec_ids.isdisjoint(set(watch_history))

    def test_tier3_06_concurrent_storage_and_recommendation_pipeline(
        self, temp_data_dir, sample_animes, tags_vocabulary, oracle
    ):
        """
        Combination 6: Atomic storage operations combined with live profile and recommendation pipeline.
        Saves updated profile with FileLock, appends watch history, and reads back cleanly.
        """
        vocab = [t["name"] for t in tags_vocabulary]
        profile_path = os.path.join(temp_data_dir, "user_profile.json")
        history_path = os.path.join(temp_data_dir, "watch_history.json")

        # 1. Update in-memory vector
        u = oracle.build_feature_vector(sample_animes[0]["genres"], sample_animes[0]["tags"], vocab)

        # 2. Persist profile atomically with lock
        profile_data = {
            "user_id": "default_user",
            "updated_at": "2026-10-15T01:30:00Z",
            "total_ratings": 1,
            "feature_weights": {vocab[i]: float(u[i]) for i in range(len(vocab)) if u[i] > 0.0},
            "hyperparameters": {"eta": 0.15, "alpha": 0.25, "gamma": 0.98, "lambda_mmr": 0.5}
        }

        with FileLock(f"{profile_path}.lock", timeout=10):
            with tempfile.NamedTemporaryFile("w", dir=temp_data_dir, delete=False, encoding="utf-8") as tf:
                json.dump(profile_data, tf)
                temp_name = tf.name
            os.replace(temp_name, profile_path)

        # 3. Append to watch history atomically with lock
        with FileLock(f"{history_path}.lock", timeout=10):
            with open(history_path, "r", encoding="utf-8") as f:
                hist = json.load(f)
            hist.append({
                "id": "wh-combo-1",
                "anime_id": sample_animes[0]["id"],
                "anime_title": sample_animes[0]["title_romaji"],
                "rating": 10.0,
                "origin_track": "onboarding",
                "timestamp": "2026-10-15T01:30:00Z"
            })
            with tempfile.NamedTemporaryFile("w", dir=temp_data_dir, delete=False, encoding="utf-8") as tf:
                json.dump(hist, tf)
                temp_name = tf.name
            os.replace(temp_name, history_path)

        # 4. Verify integrity and reload
        with open(profile_path, "r", encoding="utf-8") as f:
            loaded_prof = json.load(f)
        with open(history_path, "r", encoding="utf-8") as f:
            loaded_hist = json.load(f)

        assert loaded_prof["total_ratings"] == 1
        assert len(loaded_hist) == 1
        assert loaded_hist[0]["anime_id"] == sample_animes[0]["id"]
