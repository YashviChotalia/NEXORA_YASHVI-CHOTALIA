"""Call-level distress evaluation: does a whole call from a distressed speaker score higher than a calm one?

The clip-level evaluation (run_ser_eval.py) scores single utterances. The officer, however, sees the
*distress track* for a whole call: 3-second windows, compared with the caller's own baseline, sustained over
time (anvesh/distress.py). This script builds one simulated phone call per speaker and emotion by joining
that speaker's CREMA-D sentences (calm = neutral, distressed = fear or sadness, plus anger), sends it through
the telephone channel, cuts it into windows with the app's own `pipeline.compute_windows` rules and scores it
with `distress.assess`, exactly as an uploaded call is scored.

Speaker-disjoint like the head: thresholds and formula choices are made on the VALIDATION speakers, and the
TEST speakers (id divisible by 3) are only reported.

  python eval/run_call_eval.py --crema-d eval/audio/crema-d               # runs the model, caches windows
  python eval/run_call_eval.py --from-cache                                # re-score from cached windows

Writes eval/results/call_results.md and eval/results/call_windows_<split>.json (window features and model
outputs, no audio).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel import degrade, trim_silence  # noqa: E402

from anvesh import distress  # noqa: E402
from anvesh.live_ser import adjust_for_role  # noqa: E402
from anvesh.speech import audio as A  # noqa: E402

OUT = Path(__file__).with_name("results")
CALLS = {"NEU": "calm", "FEA": "fearful", "SAD": "sad", "ANG": "angry"}
DISTRESSED = ("fearful", "sad")
CLIPS_PER_CALL = 10
GAP_S = 0.35


def split_of(speaker: int) -> str:
    if speaker % 3 == 0:
        return "test"
    return "val" if speaker % 9 in (1, 2) else "train"


def build_call(folder: Path, speaker: int, code: str) -> np.ndarray | None:
    files = sorted(folder.glob(f"{speaker}_*_{code}_*.wav"))[:CLIPS_PER_CALL]
    if len(files) < 6:
        return None
    gap = np.zeros(int(GAP_S * A.SR), np.float32)
    parts = []
    for f in files:
        parts += [trim_silence(A.decode(f.read_bytes())), gap]
    return np.concatenate(parts)


def windows_for(x: np.ndarray, emotion, head) -> list[dict]:
    """pipeline.compute_windows without a transcript (3 s windows), keeping raw and head outputs side by side."""
    from anvesh import pipeline
    out = []
    for t0, t1, chunk in A.windows(x, 3.0):
        f = A.features(chunk)
        voiced = 1 - f.get("pause_ratio", 1.0) if f else 0.0
        silent = not f or f.get("rms", 0.0) < pipeline.MIN_RMS or voiced < pipeline.MIN_VOICED
        raw = adapted = None
        if not silent:
            raw, feats = emotion.predict_full(chunk, block=True)
            adapted = head.predict(feats) if (head is not None and feats is not None) else None
        out.append({"t_start": round(t0, 2), "t_end": round(t1, 2), "silent": silent, "acoustic": f,
                    "raw": raw, "head": adapted})
    return out


def collect(folder: Path, split: str, condition: str, budget_s: float) -> dict:
    from anvesh.speech.models import emotion
    if emotion.get(block=True) is None:
        raise SystemExit(f"Voice emotion model not available: {emotion.status().error}")
    head = emotion.head()
    cache = OUT / f"call_windows_{split}.json"
    data = json.loads(cache.read_text()) if cache.exists() else {}
    speakers = sorted({int(p.stem.split("_")[0]) for p in folder.glob("*.wav")})
    t0 = time.time()
    for s in speakers:
        if split_of(s) != split:
            continue
        for code, kind in CALLS.items():
            key = f"{condition}|{s}|{kind}"
            if key in data:
                continue
            x = build_call(folder, s, code)
            if x is None:
                continue
            data[key] = windows_for(degrade(x, condition, seed=s * 10 + len(kind)), emotion, head)
            if budget_s and time.time() - t0 > budget_s:
                cache.write_text(json.dumps(data))
                print(f"  paused ({len(data)} calls cached); run again to continue", flush=True)
                return data
        cache.write_text(json.dumps(data))
        print(f"  {split} speaker {s} done ({len(data)} calls)", flush=True)
    cache.write_text(json.dumps(data))
    return data


def score_calls(data: dict, system: str, condition: str) -> list[dict]:
    rows = []
    for key, wins in data.items():
        cond, spk, kind = key.split("|")
        if cond != condition:
            continue
        ws = [distress.Window(w["t_start"], w["t_end"], adjust_for_role(w[system], "victim"), w["acoustic"],
                              w["silent"]) for w in wins]
        r = distress.assess("victim", ws, [], [], language="en", validated_languages={"en"})
        rows.append({"speaker": spk, "kind": kind, "score": r.score, "level": r.level})
    return rows


def auc(pos: list[float], neg: list[float]) -> float | None:
    if not pos or not neg:
        return None
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def summarise(rows: list[dict]) -> dict:
    calm = [r["score"] for r in rows if r["kind"] == "calm" and r["score"] is not None]
    dist = [r["score"] for r in rows if r["kind"] in DISTRESSED and r["score"] is not None]
    by = defaultdict(Counter)
    for r in rows:
        by[r["kind"]][r["level"]] += 1
    # paired: for each speaker, is their distressed call scored above their calm call?
    per = defaultdict(dict)
    for r in rows:
        per[r["speaker"]][r["kind"]] = r["score"]
    pairs = [(v[k], v["calm"]) for v in per.values() if "calm" in v and v["calm"] is not None
             for k in DISTRESSED if v.get(k) is not None]
    return {"calls": len(rows), "auc": auc(dist, calm),
            "paired": sum(a > b for a, b in pairs) / len(pairs) if pairs else None,
            "levels": {k: dict(v) for k, v in by.items()},
            "calm_flagged": (sum(r["level"] in ("elevated", "high") for r in rows if r["kind"] == "calm")
                             / max(1, sum(r["kind"] == "calm" for r in rows))),
            "distressed_flagged": (sum(r["level"] in ("elevated", "high") for r in rows if r["kind"] in DISTRESSED)
                                   / max(1, sum(r["kind"] in DISTRESSED for r in rows))),
            "distressed_high": (sum(r["level"] == "high" for r in rows if r["kind"] in DISTRESSED)
                                / max(1, sum(r["kind"] in DISTRESSED for r in rows))),
            "calm_high": (sum(r["level"] == "high" for r in rows if r["kind"] == "calm")
                          / max(1, sum(r["kind"] == "calm" for r in rows)))}


def pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.0f}%"


def report(data_by_split: dict, condition: str) -> tuple[str, dict]:
    L = ["# Call-level distress results\n",
         f"One simulated call per speaker and emotion ({CLIPS_PER_CALL} sentences, telephone channel), cut into 3 s "
         "windows and scored by `distress.assess` exactly as an uploaded victim call. Generated by "
         "`eval/run_call_eval.py`.\n",
         "| Speakers | System | Calls | AUC distressed vs calm | Distressed > calm (same speaker) | "
         "Calm calls flagged elevated+ | Distressed calls flagged elevated+ | Calm high | Distressed high |",
         "|---|---|---|---|---|---|---|---|---|"]
    out: dict = {}
    for split, data in data_by_split.items():
        for system, name in (("raw", "raw emotion2vec+"), ("head", "telephone head")):
            m = summarise(score_calls(data, system, condition))
            out[f"{split}/{system}"] = m
            L.append(f"| {split} | {name} | {m['calls']} | {m['auc']:.2f} | {pct(m['paired'])} | "
                     f"{pct(m['calm_flagged'])} | {pct(m['distressed_flagged'])} | {pct(m['calm_high'])} | "
                     f"{pct(m['distressed_high'])} |")
    L.append("\n*Distressed* = fear or sadness. *Elevated+* = level elevated or high. Only *high* can raise a "
             "call's priority (by one level); *elevated* is a tie-break inside a level.\n")
    test = out.get("test/head")
    if test:
        L += ["Levels per call type (test speakers, telephone head):\n", "| Call | low | elevated | high | abstain |",
              "|---|---|---|---|---|"]
        for kind, lv in test["levels"].items():
            L.append(f"| {kind} | {lv.get('low', 0)} | {lv.get('elevated', 0)} | {lv.get('high', 0)} "
                     f"| {lv.get('abstain', 0)} |")
    return "\n".join(L) + "\n", out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crema-d", help="folder with CREMA-D wav files (all four classes)")
    ap.add_argument("--condition", default="phone", choices=["clean", "phone", "phone_snr20", "phone_snr10"])
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (resume on the next run)")
    ap.add_argument("--from-cache", action="store_true")
    a = ap.parse_args(argv)
    OUT.mkdir(exist_ok=True)
    data = {}
    for split in a.splits.split(","):
        cache = OUT / f"call_windows_{split}.json"
        if a.from_cache:
            if cache.exists():
                data[split] = json.loads(cache.read_text())
        else:
            if not a.crema_d:
                ap.error("give --crema-d (or --from-cache)")
            data[split] = collect(Path(a.crema_d), split, a.condition, a.budget)
    md, out = report(data, a.condition)
    (OUT / "call_results.md").write_text(md, encoding="utf-8")
    (OUT / "call_results.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
