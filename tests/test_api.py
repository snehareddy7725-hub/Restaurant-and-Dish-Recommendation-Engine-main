import json
import time

from fastapi.testclient import TestClient

from api.main import app
from recommender import config


def test_api_end_to_end():
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"

        r = client.get("/recommend/0", params={"n": 5})
        assert r.status_code == 200 and r.json()["mode"] == "hybrid"
        assert len(r.json()["recommendations"]) == 5

        assert client.get("/recommend/123456789").json()["mode"] == "popularity"

        # bad input is rejected
        assert client.post("/interactions", json={"user_id": 1, "restaurant_id": 1, "interaction_type": "like"}).status_code == 422
        assert client.post("/interactions", json={"user_id": 1, "restaurant_id": 99999, "interaction_type": "view"}).status_code == 404

        # a brand-new user: popularity -> content_only after first interaction
        new_user = 555_555
        r = client.post("/interactions", json={"user_id": new_user, "restaurant_id": 24, "interaction_type": "order"})
        assert r.status_code == 201
        assert client.get(f"/recommend/{new_user}").json()["mode"] == "content_only"

        # threshold is 5 in tests -> 4 more interactions trigger a retrain
        triggered = False
        for rid in (10, 11, 12, 13):
            triggered |= client.post("/interactions", json={"user_id": new_user, "restaurant_id": rid, "interaction_type": "view"}).json()["retrain_triggered"]
        assert triggered

        from api.main import state
        for _ in range(120):
            if not state["retrain"].running:
                break
            time.sleep(0.5)
        assert state["retrain"].last_result["status"] == "ok"

        # after retraining the new user is part of the ALS model -> hybrid
        assert client.get(f"/recommend/{new_user}").json()["mode"] == "hybrid"
        assert client.get("/health").json()["pending_interactions"] == 0

    # every request was logged as one JSON line
    lines = (config.LOG_DIR / "requests.jsonl").read_text().strip().splitlines()
    entries = [json.loads(line) for line in lines]
    assert any(e.get("mode") == "hybrid" and e["path"] == "/recommend/0" for e in entries)
    assert all({"request_id", "status", "latency_ms"} <= e.keys() for e in entries)
