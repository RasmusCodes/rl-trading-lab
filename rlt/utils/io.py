"""Run directories, JSON/JSONL logging, config hashing and provenance."""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"


def safe_name(name: str) -> str:
    """Filesystem-safe version of a universe/experiment name (Windows forbids ':')."""
    return str(name).replace(":", "-").replace("/", "-").replace("\\", "-")


def _default(o):
    if is_dataclass(o):
        return asdict(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and not np.isfinite(o):
        return None
    return str(o)


def _clean(o):
    """Replace non-finite floats by None so the JSON stays standard (browsers reject NaN)."""
    if isinstance(o, float):
        return o if np.isfinite(o) else None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


def dumps(obj, **kw) -> str:
    return json.dumps(_clean(json.loads(json.dumps(obj, default=_default))), **kw)


def save_json(path, obj, indent=2) -> None:
    """Atomic write. On Windows os.replace fails while another process (e.g. the dashboard) has the target open,
    so retry briefly instead of failing the run."""
    import os
    import time as _time

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    tmp.write_text(dumps(obj, indent=indent), encoding="utf-8")
    for attempt in range(50):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            _time.sleep(0.05 * (1 + attempt % 5))
    tmp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def config_hash(cfg) -> str:
    return hashlib.sha1(dumps(cfg, sort_keys=True).encode()).hexdigest()[:10]


def code_fingerprint() -> str:
    """Hash of every source file in the package: ties each result to the exact code that produced it."""
    h = hashlib.sha1()
    for p in sorted((PROJECT_ROOT / "rlt").rglob("*.py")):
        h.update(p.relative_to(PROJECT_ROOT).as_posix().encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:12]


def env_info() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "numpy": np.__version__}
    try:
        import torch

        info["torch"] = torch.__version__
    except ImportError:  # pragma: no cover
        pass
    try:
        import pandas

        info["pandas"] = pandas.__version__
    except ImportError:  # pragma: no cover
        pass
    return info


class JsonlLogger:
    """Append-only JSONL writer, flushed every row so the dashboard can tail it while a run trains."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")
        self.t0 = time.time()

    def log(self, row: dict) -> None:
        row = dict(row)
        row.setdefault("wall", round(time.time() - self.t0, 2))
        self._fh.write(dumps(row) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()


def read_jsonl(path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a row being written right now
    return rows
