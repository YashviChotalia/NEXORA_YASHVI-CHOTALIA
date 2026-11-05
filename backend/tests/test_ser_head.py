import json

import numpy as np

from anvesh import live_ser
from anvesh.speech import ser_head


def _spec(dim=3):
    return {"name": "test head", "classes": ["neutral", "angry", "fearful", "sad"],
            "mean": [0.0] * dim, "scale": [1.0] * dim,
            "coef": [[0.0] * dim, [2.0] + [0.0] * (dim - 1), [0.0, 2.0] + [0.0] * (dim - 2), [0.0] * dim],
            "intercept": [0.0, 0.0, 0.0, 0.0], "temperature": 1.0}


def test_head_returns_a_calibrated_distribution():
    h = ser_head.Head(_spec())
    p = h.predict([3.0, 0.0, 0.0])
    assert abs(sum(p.values()) - 1) < 1e-3 and max(p, key=p.get) == "angry"
    hot = ser_head.Head({**_spec(), "temperature": 4.0}).predict([3.0, 0.0, 0.0])
    assert hot["angry"] < p["angry"]  # a higher temperature means less confident, never a different answer
    assert max(hot, key=hot.get) == "angry"


def test_head_rejects_wrong_or_broken_embeddings():
    h = ser_head.Head(_spec())
    assert h.predict([1.0, 2.0]) is None
    assert h.predict([np.nan, 0.0, 0.0]) is None


def test_head_file_is_plain_json_and_missing_file_is_not_an_error(tmp_path):
    f = tmp_path / "head.json"
    f.write_text(json.dumps(_spec()), encoding="utf-8")
    assert ser_head.load(str(f)).predict([0.0, 3.0, 0.0])["fearful"] > 0.5
    assert ser_head.load(str(tmp_path / "missing.json")) is None


def test_shipped_head_matches_the_emotion_model_embedding():
    h = ser_head.load()
    if h is None:  # weights not built in this checkout
        return
    assert h.dim == 768 and h.classes == ["neutral", "angry", "fearful", "sad"]
    assert h.spec["split"].startswith("speaker-disjoint")


def test_emotion_adapter_prefers_the_head_and_falls_back_to_raw(monkeypatch):
    from anvesh.speech import models as M
    raw = {"neutral": 0.9, "angry": 0.1}
    monkeypatch.setattr(M.emotion, "predict_full", lambda x, block=False: (raw, np.array([3.0, 0.0, 0.0])))
    monkeypatch.setattr(M.emotion, "head", lambda: ser_head.Head(_spec()))
    assert max(M.emotion.predict(np.zeros(16000)).items(), key=lambda kv: kv[1])[0] == "angry"
    monkeypatch.setattr(M.emotion, "head", lambda: None)
    assert M.emotion.predict(np.zeros(16000)) == raw


def test_latency_summary_reports_median_and_p95():
    s = live_ser.LatencyStats()
    for v in range(1, 101):
        s.add("emotion_ms", v)
    out = s.summary()
    assert out["emotion_ms"]["n"] == 100 and out["emotion_ms"]["median"] == 50.5 and out["emotion_ms"]["p95"] == 95
    assert "transcript_s" not in out
