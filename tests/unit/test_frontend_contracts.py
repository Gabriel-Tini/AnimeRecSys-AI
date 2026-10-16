"""
tests/unit/test_frontend_contracts.py
Milestone 5 & 6 Verification: High-Density Minimalist Web UI & Decoupling Contracts.
Verifies:
- Presence and completeness of React + Vite + TypeScript + Tailwind CSS architecture.
- Mandatory Decoupling Rule: UI components never import fixtures directly; only services/api.ts does.
- Security Isolation: Zero references to GEMINI_API_KEY or VITE_GEMINI in frontend source code.
- Design System & R5 Requirements: #0a0c10 Deep Obsidian theme, 300ms search debounce, Ctrl+K command shortcut,
  80% Affinity vs 20% Discovery MMR tracks, 300-char qualitative modal cap, and client-side spoiler filtering.
"""

from pathlib import Path
import json
import re
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
FRONTEND_DIR = ROOT_DIR / "frontend"
SRC_DIR = FRONTEND_DIR / "src"
COMPONENTS_DIR = SRC_DIR / "components"


class TestFrontendArchitectureAndContracts:
    """Verifies Milestone 5 frontend files, decoupling rules, and security guardrails."""

    def test_frontend_project_files_exist(self):
        """All core configuration and source files exist in frontend/."""
        required_files = [
            FRONTEND_DIR / "package.json",
            FRONTEND_DIR / "tsconfig.json",
            FRONTEND_DIR / "vite.config.ts",
            FRONTEND_DIR / "tailwind.config.js",
            FRONTEND_DIR / "index.html",
            SRC_DIR / "main.tsx",
            SRC_DIR / "App.tsx",
            SRC_DIR / "index.css",
            SRC_DIR / "types" / "index.ts",
            SRC_DIR / "mocks" / "fixtures.ts",
            SRC_DIR / "services" / "api.ts",
            COMPONENTS_DIR / "Navbar.tsx",
            COMPONENTS_DIR / "SearchBar.tsx",
            COMPONENTS_DIR / "OnboardingShelf.tsx",
            COMPONENTS_DIR / "TrackAffinity.tsx",
            COMPONENTS_DIR / "TrackDiscovery.tsx",
            COMPONENTS_DIR / "AnimeCard.tsx",
            COMPONENTS_DIR / "QualitativeModal.tsx",
            COMPONENTS_DIR / "ProfileMetrics.tsx",
        ]
        for fpath in required_files:
            assert fpath.exists(), f"Missing required frontend file: {fpath}"
            assert fpath.stat().st_size > 50, f"Frontend file is unexpectedly empty: {fpath}"

    def test_package_json_dependencies_and_scripts(self):
        """package.json specifies React 18, Vite, TypeScript, Tailwind CSS, and Lucide React."""
        pkg = json.loads((FRONTEND_DIR / "package.json").read_text(encoding="utf-8"))
        deps = pkg.get("dependencies", {})
        dev_deps = pkg.get("devDependencies", {})
        assert "react" in deps
        assert "react-dom" in deps
        assert "lucide-react" in deps
        assert "vite" in dev_deps
        assert "typescript" in dev_deps
        assert "tailwindcss" in dev_deps

    def test_strict_decoupling_rule_no_direct_fixture_imports_in_components(self):
        """
        Mandatory Rule: UI components and App.tsx must NEVER import mocks/fixtures directly.
        All data access must pass through frontend/src/services/api.ts.
        """
        ui_files = list(COMPONENTS_DIR.glob("*.tsx")) + [SRC_DIR / "App.tsx"]
        assert len(ui_files) >= 8

        for tsx_file in ui_files:
            content = tsx_file.read_text(encoding="utf-8")
            assert "mocks/fixtures" not in content, (
                f"VIOLATION: {tsx_file.name} imports mocks/fixtures directly instead of services/api.ts"
            )
            assert "FIXTURES.json" not in content, (
                f"VIOLATION: {tsx_file.name} references FIXTURES.json directly"
            )

    def test_zero_gemini_api_key_leakage_in_frontend(self):
        """R6 Security: Frontend codebase must contain zero references to GEMINI_API_KEY or VITE_GEMINI."""
        for src_file in FRONTEND_DIR.rglob("*"):
            if src_file.is_file() and "node_modules" not in src_file.parts and "dist" not in src_file.parts:
                text = src_file.read_text(encoding="utf-8", errors="ignore")
                assert "GEMINI_API_KEY" not in text, f"Secret key identifier leaked in {src_file}"
                assert "VITE_GEMINI" not in text, f"VITE_GEMINI variable found in {src_file}"
                assert "AIzaSy" not in text, f"Hardcoded Google API key pattern found in {src_file}"

    def test_search_bar_has_300ms_debounce_and_ctrl_k_shortcut(self):
        """R5: SearchBar implements 300ms debounce and Ctrl+K / Cmd+K command shortcut."""
        content = (COMPONENTS_DIR / "SearchBar.tsx").read_text(encoding="utf-8")
        assert "300" in content, "SearchBar must use 300ms debounce timer"
        assert "ctrlKey" in content or "metaKey" in content, "SearchBar must support Ctrl+K / Cmd+K shortcut"
        assert "fetchCatalog" in content, "SearchBar must query catalog via api.ts"

    def test_dual_tracks_80_20_demarcation(self):
        """R5: TrackAffinity (80%) and TrackDiscovery (20% MMR) are visually demarcated."""
        aff_content = (COMPONENTS_DIR / "TrackAffinity.tsx").read_text(encoding="utf-8")
        disc_content = (COMPONENTS_DIR / "TrackDiscovery.tsx").read_text(encoding="utf-8")

        assert "80%" in aff_content
        assert "emerald" in aff_content
        assert "20%" in disc_content
        assert "MMR" in disc_content
        assert "purple" in disc_content

    def test_qualitative_modal_300_char_limit_and_safe_reason(self):
        """R2 & R5: QualitativeModal enforces 300-character limit and renders public_safe_reason."""
        modal_content = (COMPONENTS_DIR / "QualitativeModal.tsx").read_text(encoding="utf-8")
        assert "maxLength={300}" in modal_content or "slice(0, 300)" in modal_content
        assert "public_safe_reason" in modal_content
        assert "O que mais você gostou nesse anime?" in modal_content

    def test_api_service_fallback_and_spoiler_sanitization(self):
        """R3 & R5: services/api.ts sanitizes spoiler tags and provides transparent offline fallback."""
        api_content = (SRC_DIR / "services" / "api.ts").read_text(encoding="utf-8")
        assert "sanitizePublicAnime" in api_content
        assert "is_media_spoiler" in api_content
        assert "/health" in api_content
        assert "/recommendations" in api_content
        assert "/rate" in api_content
        assert "/tune" in api_content
        assert "/skip" in api_content
        assert "/drop" in api_content
        assert "/favorite" in api_content
        assert "/watchlist" in api_content
        assert "/history" in api_content
        assert "/reset" in api_content

    def test_all_sidebar_tabs_and_skip_drop_features_in_frontend(self):
        """Verifies Explorar, Minha Lista, Favoritos, Pular (Skip), and Dropar com Gemini are wired in UI."""
        app_content = (SRC_DIR / "App.tsx").read_text(encoding="utf-8")
        card_content = (COMPONENTS_DIR / "AnimeCard.tsx").read_text(encoding="utf-8")
        modal_content = (COMPONENTS_DIR / "QualitativeModal.tsx").read_text(encoding="utf-8")

        assert "activeTab === 'home'" in app_content
        assert "activeTab === 'explore'" in app_content
        assert "activeTab === 'list'" in app_content
        assert "activeTab === 'favorites'" in app_content
        assert "activeTab === 'profile'" in app_content

        assert "Pular" in card_content
        assert "Dropar" in card_content
        assert "Por que você decidiu dropar este anime?" in modal_content


class TestSkipDropFavoriteWatchlistEndpoints:
    """Integration tests for /api/skip, /api/drop, /api/favorite, /api/watchlist, and /api/history."""

    def test_skip_and_unskip_anime_without_vector_distortion(self, tmp_path):
        from fastapi.testclient import TestClient
        from backend.main import create_app
        from backend.app.core.storage_manager import StorageManager
        from backend.app.services.learning_engine import LearningEngine
        from backend.app.services.recommendation_engine import RecommendationEngine

        storage = StorageManager(workspace_root=ROOT_DIR, data_dir=tmp_path / "data")
        learning = LearningEngine(storage_manager=storage)
        rec_eng = RecommendationEngine(storage_manager=storage)

        app = create_app()
        app.state.storage_manager = storage
        app.state.learning_engine = learning
        app.state.recommendation_engine = rec_eng

        client = TestClient(app)

        # Rate one anime first so feature_weights is non-empty
        rate_res = client.post("/api/rate", json={"anime_id": 9253, "rating": 9.5, "origin_track": "onboarding"})
        assert rate_res.status_code == 200
        weights_before = rate_res.json()["new_profile"]["feature_weights"]

        # Skip another anime from fixtures (154587 - Sousou no Frieren)
        skip_res = client.post("/api/skip", json={"anime_id": 154587, "unskip": False})
        assert skip_res.status_code == 200
        skip_data = skip_res.json()
        assert 154587 in skip_data["skipped_anime_ids"]
        # Verify zero vector distortion (feature_weights unchanged)
        assert skip_data["new_profile"]["feature_weights"] == weights_before
        # Verify 154587 is excluded from recommendations
        aff_ids = [a["id"] for a in skip_data["updated_recommendations"]["affinity_track"]]
        disc_ids = [a["id"] for a in skip_data["updated_recommendations"]["discovery_track"]]
        assert 154587 not in aff_ids
        assert 154587 not in disc_ids

        # Unskip 154587
        unskip_res = client.post("/api/skip", json={"anime_id": 154587, "unskip": True})
        assert unskip_res.status_code == 200
        assert 154587 not in unskip_res.json()["skipped_anime_ids"]

    def test_drop_anime_with_gemini_negative_justification(self, tmp_path):
        from fastapi.testclient import TestClient
        from backend.main import create_app
        from backend.app.core.storage_manager import StorageManager
        from backend.app.services.learning_engine import LearningEngine
        from backend.app.services.recommendation_engine import RecommendationEngine

        storage = StorageManager(workspace_root=ROOT_DIR, data_dir=tmp_path / "data")
        learning = LearningEngine(storage_manager=storage)
        rec_eng = RecommendationEngine(storage_manager=storage)

        app = create_app()
        app.state.storage_manager = storage
        app.state.learning_engine = learning
        app.state.recommendation_engine = rec_eng

        client = TestClient(app)

        drop_res = client.post(
            "/api/drop",
            json={
                "anime_id": 151807,
                "drop_reason": "Achei o ritmo muito lento e o protagonista muito clichê shounen.",
                "origin_track": "onboarding",
            },
        )
        assert drop_res.status_code == 200
        data = drop_res.json()
        assert data["success"] is True
        assert 151807 in data["new_profile"]["dropped_anime_ids"]
        assert data["history_entry"]["interaction_status"] == "dropped"
        assert "lento" in data["history_entry"]["drop_reason"]

        # Verify Gemini extracted negative multipliers
        deltas = data["tuning_response"]["identified_tags"]
        assert len(deltas) >= 1
        for d in deltas:
            assert d["impact_multiplier"] < 0

        # Verify dropped anime is excluded from recommendations
        aff_ids = [a["id"] for a in data["updated_recommendations"]["affinity_track"]]
        disc_ids = [a["id"] for a in data["updated_recommendations"]["discovery_track"]]
        assert 151807 not in aff_ids
        assert 151807 not in disc_ids

    def test_favorite_watchlist_and_history_endpoints(self, tmp_path):
        from fastapi.testclient import TestClient
        from backend.main import create_app
        from backend.app.core.storage_manager import StorageManager

        storage = StorageManager(workspace_root=ROOT_DIR, data_dir=tmp_path / "data")
        app = create_app()
        app.state.storage_manager = storage
        client = TestClient(app)

        fav_res = client.post("/api/favorite", json={"anime_id": 154587})
        assert fav_res.status_code == 200
        assert fav_res.json()["active"] is True
        assert 154587 in fav_res.json()["anime_ids"]

        wl_res = client.post("/api/watchlist", json={"anime_id": 113415})
        assert wl_res.status_code == 200
        assert wl_res.json()["active"] is True
        assert 113415 in wl_res.json()["anime_ids"]

        hist_res = client.get("/api/history")
        assert hist_res.status_code == 200
        hist_data = hist_res.json()
        assert 154587 in hist_data["favorite_anime_ids"]
        assert 113415 in hist_data["watchlist_anime_ids"]
        # Verify resolved anime public items are present in history response
        assert len(hist_data["favorite_animes"]) >= 1
        assert any(a["id"] == 154587 for a in hist_data["favorite_animes"])
        assert len(hist_data["watchlist_animes"]) >= 1
        assert any(a["id"] == 113415 for a in hist_data["watchlist_animes"])

    def test_recommendation_filters_block_formats_and_genres(self, tmp_path):
        from fastapi.testclient import TestClient
        from backend.main import create_app
        from backend.app.core.storage_manager import StorageManager
        from backend.app.services.learning_engine import LearningEngine
        from backend.app.services.recommendation_engine import RecommendationEngine

        storage = StorageManager(workspace_root=ROOT_DIR, data_dir=tmp_path / "data")
        learning = LearningEngine(storage_manager=storage)
        rec_eng = RecommendationEngine(storage_manager=storage)

        app = create_app()
        app.state.storage_manager = storage
        app.state.learning_engine = learning
        app.state.recommendation_engine = rec_eng

        client = TestClient(app)

        # Apply filter: block MOVIE and SPECIAL, block 'Romance' and 'Mecha'
        res = client.post(
            "/api/filters",
            json={
                "excluded_formats": ["MOVIE", "SPECIAL"],
                "blocked_genres": ["Romance", "Mecha"],
            },
        )
        assert res.status_code == 200
        data = res.json()
        assert data["success"] is True
        assert "MOVIE" in data["new_profile"]["excluded_formats"]
        assert "Romance" in data["new_profile"]["blocked_genres"]

        # Ensure no MOVIE or SPECIAL in recommendations
        all_recs = (
            data["updated_recommendations"]["affinity_track"]
            + data["updated_recommendations"]["discovery_track"]
        )
        for anime in all_recs:
            assert anime["format"] not in ["MOVIE", "SPECIAL"]
            for g in anime["genres"]:
                assert g.lower() not in ["romance", "mecha"]

    def test_google_auth_login_sync_and_logout(self, tmp_path):
        from fastapi.testclient import TestClient
        from backend.main import create_app
        from backend.app.core.storage_manager import StorageManager

        storage = StorageManager(workspace_root=ROOT_DIR, data_dir=tmp_path / "data")
        app = create_app()
        app.state.storage_manager = storage
        client = TestClient(app)

        # Login with Google
        login_res = client.post(
            "/api/auth/google",
            json={
                "email": "test.user@gmail.com",
                "name": "Test User",
            },
        )
        assert login_res.status_code == 200
        login_data = login_res.json()
        assert login_data["success"] is True
        assert login_data["new_profile"]["google_account"]["email"] == "test.user@gmail.com"

        # Explicit Sync
        sync_res = client.post("/api/auth/sync")
        assert sync_res.status_code == 200
        assert sync_res.json()["success"] is True

        # Logout
        logout_res = client.post("/api/auth/logout")
        assert logout_res.status_code == 200
        assert logout_res.json()["new_profile"]["google_account"] is None

