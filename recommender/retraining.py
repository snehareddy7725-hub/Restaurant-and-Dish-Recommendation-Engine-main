"""Interaction logging and the interaction-count retraining trigger.

New interactions posted to the API are appended to data/new_interactions.csv.
Once RETRAIN_THRESHOLD new interactions have accumulated, a background thread
retrains both models (training.retrain_and_save) and hot-reloads them into the
running Recommender. Only one retrain runs at a time.
"""

import csv
import logging
import threading
from datetime import datetime, timezone

from . import config
from .training import retrain_and_save

log = logging.getLogger("reco.retrain")
FIELDS = ["user_id", "restaurant_id", "interaction_type", "strength", "logged_at"]


class RetrainManager:
    def __init__(self, recommender, threshold: int = None, auto: bool = None):
        self.recommender = recommender
        self.threshold = config.RETRAIN_THRESHOLD if threshold is None else threshold
        self.auto = config.AUTO_RETRAIN if auto is None else auto
        self._lock = threading.Lock()
        self._running = threading.Event()
        self.pending = self._count_pending()
        self.last_result = None
        self.retrain_count = 0

    @staticmethod
    def _count_pending() -> int:
        if not config.NEW_INTERACTIONS_PATH.exists():
            return 0
        with open(config.NEW_INTERACTIONS_PATH) as f:
            return max(sum(1 for _ in f) - 1, 0)

    def log_interaction(self, user_id: int, restaurant_id: int, interaction_type: str) -> dict:
        strength = config.INTERACTION_STRENGTH[interaction_type]
        with self._lock:
            new_file = not config.NEW_INTERACTIONS_PATH.exists()
            with open(config.NEW_INTERACTIONS_PATH, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    w.writeheader()
                w.writerow({
                    "user_id": user_id, "restaurant_id": restaurant_id,
                    "interaction_type": interaction_type, "strength": strength,
                    "logged_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                })
            self.pending += 1
            pending = self.pending

        self.recommender.add_interaction(user_id, restaurant_id, strength)
        triggered = False
        if self.auto and pending >= self.threshold:
            triggered = self.trigger(reason=f"threshold reached ({pending}/{self.threshold})")
        return {"pending_interactions": pending, "retrain_triggered": triggered}

    def trigger(self, reason: str = "manual", blocking: bool = False) -> bool:
        """Start a retrain unless one is already running. Returns True if started."""
        if self._running.is_set():
            return False
        self._running.set()
        log.info("retrain started: %s", reason)
        if blocking:
            self._run()
        else:
            threading.Thread(target=self._run, daemon=True).start()
        return True

    def _run(self):
        try:
            with self._lock:  # block new writes while new_interactions.csv is merged
                state = retrain_and_save(merge_new_interactions=True)
                self.pending = 0
            self.recommender.reload()
            self.retrain_count += 1
            self.last_result = {"status": "ok", **state}
            log.info("retrain finished: %s", state)
        except Exception as exc:  # keep serving the old model if training fails
            self.last_result = {"status": "failed", "error": str(exc)}
            log.exception("retrain failed")
        finally:
            self._running.clear()

    @property
    def running(self) -> bool:
        return self._running.is_set()
