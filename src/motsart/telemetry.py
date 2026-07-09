"""Lightweight per-reaction telemetry: timings + failure tallies to JSONL.

Each pipeline stage appends newline-delimited JSON records to
``results*/R{rxn_id}/telemetry/metrics.jsonl``. Two record kinds:

* timing -- how long an expensive component took::

    {"kind":"timing","rxn_id":74,"stage":"validator","component":"sp_opt",
     "seconds":42.1,"meta":{...}}

* event  -- a (total, failed) tally for a component, so failure rates can be
  aggregated across reactions::

    {"kind":"event","rxn_id":74,"stage":"validator","component":"sp_opt",
     "total":1,"failed":0,"meta":{...}}

Aggregate everything with ``experiments/summarize_telemetry.py``.

Design notes
------------
Best-effort: every operation is wrapped so telemetry can never raise into
  the pipeline. A failed write prints a warning and is dropped.
Per-reaction file, append-only: stages run as separate processes (one
  reaction per SLURM task), each appending to its own reaction's file, so there
  is no cross-process contention.
Set ``MOTSART_TELEMETRY=0`` to disable all recording.
"""

from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from motsart.common import PathHandler


class Telemetry:
    """Append timing/event records for one reaction + pipeline stage."""

    def __init__(self, metrics_jsonl: Optional[Path], rxn_id, stage: str):
        self.path = Path(metrics_jsonl) if metrics_jsonl else None
        self.rxn_id = rxn_id
        self.stage = stage
        self.enabled = (os.environ.get("MOTSART_TELEMETRY", "1") != "0") and self.path is not None
        if self.enabled:
            try:
                self.path.parent.mkdir(parents=True, exist_ok=True)
            except Exception:
                self.enabled = False

    # ----------------------------- constructors ----------------------------- #
    @classmethod
    def for_rxn(cls, rxn_id, results_folder: str, stage: str) -> "Telemetry":
        """Build from a reaction id + results folder (resolves the JSONL path)."""
        try:
            ph = PathHandler(rxn_id=rxn_id, r_or_p="r", ts_method="_telemetry",
                             results_folder=results_folder)
            return cls(ph.metrics_jsonl, rxn_id, stage)
        except Exception:
            return cls(None, rxn_id, stage)

    @classmethod
    def from_path_handler(cls, path_handler, rxn_id, stage: str) -> "Telemetry":
        return cls(getattr(path_handler, "metrics_jsonl", None), rxn_id, stage)

    @classmethod
    def disabled(cls, rxn_id=None, stage: str = "") -> "Telemetry":
        return cls(None, rxn_id, stage)

    # ------------------------------- recording ------------------------------ #
    def _write(self, record: dict) -> None:
        if not self.enabled:
            return
        try:
            record.setdefault("rxn_id", self.rxn_id)
            record.setdefault("stage", self.stage)
            record["wall_ts"] = time.time()
            with open(self.path, "a") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except Exception as e:  # never break the pipeline
            print(f"[telemetry] write failed: {e}")

    def tally(self, component: str, total: int, failed: int, **meta) -> None:
        """Record a (total attempts, failed attempts) tally for ``component``."""
        self._write({"kind": "event", "component": component, "total": int(total), "failed": int(failed), "meta": meta})

    def event(self, component: str, ok: bool, **meta) -> None:
        """Record a single attempt (failed iff ``not ok``)."""
        self.tally(component, total=1, failed=0 if ok else 1, **meta)

    @contextmanager
    def timer(self, component: str, **meta):
        """Time a code block. The yielded dict's ``meta`` may be mutated in-place
        to attach values discovered during the block (e.g. counts)."""
        t0 = time.perf_counter()
        handle = {"meta": dict(meta)}
        try:
            yield handle
        finally:
            self._write({"kind": "timing", "component": component, "seconds": time.perf_counter() - t0, "meta": handle.get("meta", meta)})
