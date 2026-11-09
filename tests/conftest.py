"""Tests run against a temporary copy of data/ so the real files are never modified."""
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_tmp = Path(tempfile.mkdtemp(prefix="reco-test-"))
shutil.copytree(ROOT / "data", _tmp / "data")
os.environ["RECO_DATA_DIR"] = str(_tmp / "data")
os.environ["RECO_LOG_DIR"] = str(_tmp / "logs")
os.environ["RECO_RETRAIN_THRESHOLD"] = "5"
sys.path.insert(0, str(ROOT))
