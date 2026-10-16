"""
backend/scripts/clean_and_deduplicate.py
Deduplicates watch_history.json keeping the latest entry per anime_id,
recalculates user_profile.json feature_weights and total_ratings from cold start
strictly preserving unit L2 norm, and syncs account snapshots.
"""
import sys
from pathlib import Path
from datetime import datetime, timezone

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.core.storage_manager import StorageManager
from app.services.learning_engine import LearningEngine

def main():
    storage = StorageManager()
    learning = LearningEngine(storage_manager=storage)

    # 1. Load current history
    history = storage.load_watch_history()
    print(f"Current history length: {len(history)}")

    # Group by anime_id, keeping the latest entry
    deduped_by_id = {}
    for h in history:
        aid = h.anime_id
        if aid not in deduped_by_id:
            deduped_by_id[aid] = h
        else:
            # If already exists, keep the one that was updated last or has qualitative review
            prev = deduped_by_id[aid]
            if (h.user_review_raw and not prev.user_review_raw) or (h.timestamp >= prev.timestamp):
                deduped_by_id[aid] = h

    cleaned_history = list(deduped_by_id.values())
    print(f"Cleaned history length: {len(cleaned_history)}")

    # Save cleaned history
    storage.save_watch_history(cleaned_history)

    # 2. Recalibrate user profile from cold-start state
    catalog = storage.load_catalog()
    catalog_by_id = {a.id: a for a in catalog}

    now_utc = datetime.now(timezone.utc)

    def recalibrate(prof):
        # Reset weights and ratings count
        temp_prof = prof.model_copy(
            update={
                "feature_weights": {},
                "total_ratings": 0,
                "updated_at": now_utc,
            }
        )
        for h in cleaned_history:
            if h.interaction_status == "watched" and h.anime_id in catalog_by_id:
                ent = catalog_by_id[h.anime_id]
                temp_prof = learning.update_profile_rating(
                    profile=temp_prof,
                    rated_anime=ent,
                    rating=h.rating,
                    qualitative_tuning=h.qualitative_tuning,
                )
        return temp_prof

    updated_profile = storage.update_user_profile(recalibrate)
    print(f"Recalibrated user profile: total_ratings={updated_profile.total_ratings}, norm={updated_profile.compute_l2_norm():.6f}")

    # 3. Update account files
    accounts_dir = Path("data/accounts")
    if accounts_dir.exists():
        import json
        for acc_path in accounts_dir.glob("*.json"):
            try:
                acc_data = json.loads(acc_path.read_text(encoding="utf-8"))
                # Filter watch history in account
                acc_hist = acc_data.get("watch_history", [])
                acc_deduped = {}
                for h in acc_hist:
                    aid = h.get("anime_id")
                    if aid:
                        acc_deduped[aid] = h
                acc_data["watch_history"] = list(acc_deduped.values())
                acc_data["profile"] = updated_profile.model_dump(mode="json")
                acc_path.write_text(json.dumps(acc_data, indent=2, ensure_ascii=False), encoding="utf-8")
                print(f"Updated account snapshot in {acc_path.name}")
            except Exception as e:
                print(f"Error updating account {acc_path}: {e}")

    print("Deduplication and algorithm recalibration completed successfully!")

if __name__ == "__main__":
    main()
