"""Telephone-adapted emotion head on top of emotion2vec+ embeddings.

emotion2vec+ is trained mostly on wide-band (16 kHz) speech. Measured on CREMA-D, its accuracy drops sharply
once audio goes through a telephone channel (8 kHz, 300-3400 Hz, mu-law), and it stays over-confident while
wrong. This head is a small, calibrated linear classifier (4 classes: neutral, angry, fearful, sad) trained on
the model's own 768-d utterance embedding, using clean, telephone and noisy-telephone versions of speakers that
are never used for testing. It runs on the same forward pass, so it adds well under a millisecond per window.

The weights are a plain JSON file (no pickle): `anvesh/data/models/ser_head_telephone.json`, built by
`eval/train_ser_head.py`. Set ANVESH_SER_HEAD=off to use the raw emotion2vec+ output instead.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "models" / "ser_head_telephone.json"


class Head:
    def __init__(self, spec: dict):
        self.spec = spec
        self.classes: list[str] = list(spec["classes"])
        self.mean = np.asarray(spec["mean"], dtype=np.float32)
        self.scale = np.asarray(spec["scale"], dtype=np.float32)
        self.coef = np.asarray(spec["coef"], dtype=np.float32)
        self.intercept = np.asarray(spec["intercept"], dtype=np.float32)
        self.temperature = float(spec.get("temperature", 1.0))
        # log-prior added to "neutral" after calibration: the operating point chosen on validation speakers to cap
        # false alarms on calm telephone speech (see eval/train_ser_head.py)
        self.neutral_prior = float(spec.get("neutral_log_prior", 0.0))
        if self.coef.shape != (len(self.classes), len(self.mean)):
            raise ValueError("ser head: weight shape does not match classes / embedding size")

    @property
    def dim(self) -> int:
        return len(self.mean)

    def predict(self, feats) -> dict[str, float] | None:
        z = np.asarray(feats, dtype=np.float32).reshape(-1)
        if z.shape[0] != self.dim or not np.all(np.isfinite(z)):
            return None
        logits = (self.coef @ ((z - self.mean) / self.scale) + self.intercept) / self.temperature
        if "neutral" in self.classes:
            logits[self.classes.index("neutral")] += self.neutral_prior
        p = np.exp(logits - logits.max())
        p /= p.sum()
        return {c: round(float(v), 4) for c, v in zip(self.classes, p)}

    def info(self) -> dict:
        keys = ("name", "base_model", "trained_on", "train_speakers", "train_clips", "conditions", "temperature",
                "neutral_log_prior", "validation")
        return {k: self.spec.get(k) for k in keys if k in self.spec}


@lru_cache(maxsize=4)
def load(path: str | None = None) -> Head | None:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        return None
    try:
        return Head(json.loads(p.read_text(encoding="utf-8")))
    except (ValueError, KeyError, OSError):
        return None
