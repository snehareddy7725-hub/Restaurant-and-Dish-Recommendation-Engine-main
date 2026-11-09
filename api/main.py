"""FastAPI service for the hybrid recommender.

Run locally:   uvicorn api.main:app --reload
Docs:          http://localhost:8000/docs
"""

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from logging.handlers import RotatingFileHandler
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Request
from pydantic import BaseModel

from recommender import config
from recommender.engine import MODES, Recommender
from recommender.retraining import RetrainManager

# ------------------------------------------------------------------ logging
config.LOG_DIR.mkdir(parents=True, exist_ok=True)
request_log = logging.getLogger("reco.requests")
request_log.setLevel(logging.INFO)
if not request_log.handlers:
    fh = RotatingFileHandler(config.LOG_DIR / "requests.jsonl", maxBytes=5_000_000, backupCount=3)
    fh.setFormatter(logging.Formatter("%(message)s"))
    request_log.addHandler(fh)
    request_log.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

state = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    reco = Recommender()
    state["reco"] = reco
    state["retrain"] = RetrainManager(reco)
    state["mode_counts"] = {m: 0 for m in MODES}
    yield
    state.clear()


app = FastAPI(title="Hybrid Restaurant Recommender", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """One JSON line per request: id, method, path, status, latency, and the
    recommendation mode when one was served."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:12])
    start = time.perf_counter()
    request.state.extra = {}
    response = await call_next(request)
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "query": str(request.url.query) or None,
        "status": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        **request.state.extra,
    }
    request_log.info(json.dumps(entry))
    response.headers["x-request-id"] = request_id
    return response


# ------------------------------------------------------------------ schemas
class InteractionIn(BaseModel):
    user_id: int
    restaurant_id: int
    interaction_type: Literal["view", "cart", "order"]


# ------------------------------------------------------------------ routes
@app.get("/health")
def health():
    reco, rt = state["reco"], state["retrain"]
    training = None
    if config.TRAINING_STATE_PATH.exists():
        training = json.loads(config.TRAINING_STATE_PATH.read_text())
    return {
        "status": "ok",
        "n_users_in_als": len(reco.user_mapper),
        "n_restaurants": len(reco.restaurants),
        "pending_interactions": rt.pending,
        "retrain_threshold": rt.threshold,
        "retrain_running": rt.running,
        "retrains_since_start": rt.retrain_count,
        "last_training": training,
        "mode_counts": state["mode_counts"],
    }


@app.get("/recommend/{user_id}")
def recommend(
    request: Request,
    user_id: int,
    n: int = Query(10, ge=1, le=50),
    cuisine: str | None = Query(None, description="Preferred cuisine, used by the popularity fallback"),
):
    rec = state["reco"].recommend(user_id, n=n, cuisine=cuisine)
    state["mode_counts"][rec.mode] += 1
    request.state.extra = {"user_id": user_id, "mode": rec.mode, "n_items": len(rec.items)}
    return {"user_id": rec.user_id, "mode": rec.mode, "recommendations": rec.items}


@app.post("/interactions", status_code=201)
def add_interaction(request: Request, body: InteractionIn):
    reco = state["reco"]
    if body.restaurant_id not in reco.restaurants.index:
        raise HTTPException(404, f"Unknown restaurant_id {body.restaurant_id}")
    result = state["retrain"].log_interaction(body.user_id, body.restaurant_id, body.interaction_type)
    request.state.extra = {"user_id": body.user_id, **result}
    return {"logged": body.model_dump(), **result}


@app.post("/admin/retrain", status_code=202)
def retrain():
    started = state["retrain"].trigger(reason="manual")
    return {"started": started, "message": "retraining in background" if started else "already running"}


@app.get("/admin/retrain/status")
def retrain_status():
    rt = state["retrain"]
    return {"running": rt.running, "pending_interactions": rt.pending, "last_result": rt.last_result}
