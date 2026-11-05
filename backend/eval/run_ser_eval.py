"""Speech-emotion evaluation on labelled audio clips (needs the voice-emotion model).

Measures the emotion step that feeds the live fear / anger / distress / nervousness signals:
  * 4-class accuracy and per-class recall (neutral, anger, fear, sadness) on standard SER data,
  * "needs attention" detection (fear, anger or sadness vs neutral): precision / recall / F1,
  * how often the system abstains ("uncertain", silent or unusable audio),
  * model time per clip on this machine.

Input, either:
  --crema-d DIR      a CREMA-D AudioWAV folder (labels read from file names, e.g. 1001_DFA_ANG_XX.wav)
  --manifest FILE    a CSV with columns: path,label[,source,lang]
                     label one of: neutral, angry, fearful, sad (also: anger, fear, sadness, distress)
                     source groups the report, e.g. crema-d / roleplay / youtube

Examples:
  python eval/run_ser_eval.py --crema-d D:/data/CREMA-D/AudioWAV --limit 600
  python eval/run_ser_eval.py --manifest eval/own_samples.csv
Writes a Markdown summary to eval/ser_results.md (paste it into EVALUATION.md).

Audio clips are NOT stored in this repository (licences and privacy). Keep them outside the repo.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anvesh.live_ser import UNCERTAIN_P, adjust_for_role
from anvesh.speech import audio as A

CLASSES = ("neutral", "angry", "fearful", "sad")
ALIASES = {"anger": "angry", "ang": "angry", "fear": "fearful", "fea": "fearful", "sadness": "sad",
           "distress": "sad", "neu": "neutral", "calm": "neutral"}
CREMA = {"NEU": "neutral", "ANG": "angry", "FEA": "fearful", "SAD": "sad"}  # HAP / DIS are not used
ATTENTION = {"angry", "fearful", "sad"}


def norm_label(s: str) -> str | None:
    s = s.strip().lower()
    s = ALIASES.get(s, s)
    return s if s in CLASSES else None


def load_items(args) -> list[dict]:
    items = []
    if args.crema_d:
        for p in sorted(Path(args.crema_d).glob("*.wav")):
            parts = p.stem.split("_")
            if len(parts) >= 3 and parts[2] in CREMA:
                items.append({"path": p, "label": CREMA[parts[2]], "source": "crema-d", "lang": "en"})
    if args.manifest:
        base = Path(args.manifest).parent
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = norm_label(r.get("label", ""))
                if lab:
                    p = Path(r["path"])
                    items.append({"path": p if p.is_absolute() else base / p, "label": lab,
                                  "source": r.get("source") or "manifest", "lang": r.get("lang") or "?"})
    if args.limit and len(items) > args.limit:  # balanced random sample
        random.seed(7)
        by = defaultdict(list)
        for it in items:
            by[it["label"]].append(it)
        per = max(1, args.limit // len(by))
        items = [x for v in by.values() for x in random.sample(v, min(per, len(v)))]
    return items


def predict_label(predict, x, role: str) -> tuple[str, float | None]:
    """Same decision rules as the live tracker: silent audio and low confidence abstain."""
    f = A.features(x) or {}
    if not f or f.get("rms", 0.0) < 0.006 or 1 - f.get("pause_ratio", 1.0) < 0.35:
        return "abstain", None
    emo = adjust_for_role(predict(x), role)
    if not emo:
        return "abstain", None
    e = {k: v for k, v in emo.items() if k in CLASSES or k == "disgusted"}
    if "disgusted" in e:  # the live mapping counts half of disgust as anger
        e["angry"] = e.get("angry", 0.0) + 0.5 * e.pop("disgusted")
    if not e:
        return "abstain", None
    label, p = max(e.items(), key=lambda kv: kv[1])
    return (label if p >= UNCERTAIN_P else "uncertain"), p


def evaluate(items: list[dict], predict, role: str) -> dict:
    rows, times = [], []
    for it in items:
        x = A.decode(Path(it["path"]).read_bytes())
        t = time.perf_counter()
        pred, p = predict_label(predict, x, role)
        times.append(time.perf_counter() - t)
        rows.append({**it, "pred": pred, "p": p})
    return {"rows": rows, "times": times}


def summarise(res: dict, role: str) -> str:
    rows = res["rows"]
    mean_ms = 1000 * sum(res["times"]) / max(1, len(res["times"]))
    median_ms = 1000 * sorted(res["times"])[len(res["times"]) // 2]
    head = (f"Role prior: `{role}` · clips: {len(rows)} · model time per clip: {mean_ms:.0f} ms "
            f"(median {median_ms:.0f} ms)\n")
    out = [head]
    out.append("| Source | Clips | Abstained | 4-class accuracy (answered) | Attention P / R / F1 |")
    out.append("|---|---|---|---|---|")
    groups = defaultdict(list)
    for r in rows:
        groups[r["source"]].append(r)
    groups["ALL"] = rows
    for src, rs in groups.items():
        ans = [r for r in rs if r["pred"] in CLASSES]
        acc = sum(r["pred"] == r["label"] for r in ans) / len(ans) if ans else 0.0
        tp = sum(r["pred"] in ATTENTION and r["label"] in ATTENTION for r in ans)
        fp = sum(r["pred"] in ATTENTION and r["label"] not in ATTENTION for r in ans)
        fn = sum(r["pred"] not in ATTENTION and r["label"] in ATTENTION for r in ans)
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * pr * rc / (pr + rc) if pr + rc else 0.0
        out.append(f"| {src} | {len(rs)} | {100 * (len(rs) - len(ans)) / len(rs):.0f}% | {100 * acc:.1f}% "
                   f"| {pr:.2f} / {rc:.2f} / {f1:.2f} |")
    out.append("\nPer-class recall (all sources, answered clips):\n")
    out.append("| True label | Clips | Recall | Most common prediction |")
    out.append("|---|---|---|---|")
    for c in CLASSES:
        rs = [r for r in rows if r["label"] == c and r["pred"] in CLASSES]
        if rs:
            common = Counter(r["pred"] for r in rs).most_common(1)[0][0]
            out.append(f"| {c} | {len(rs)} | {100 * sum(r['pred'] == c for r in rs) / len(rs):.1f}% | {common} |")
    return "\n".join(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crema-d", help="CREMA-D AudioWAV folder")
    ap.add_argument("--manifest", help="CSV: path,label[,source,lang]")
    ap.add_argument("--limit", type=int, default=0, help="balanced sample size (0 = all)")
    ap.add_argument("--role", default="none", choices=["none", "victim"],
                    help="'none' = raw model output; 'victim' = with the helpline prior used on victim calls")
    ap.add_argument("--out", default=str(Path(__file__).with_name("ser_results.md")))
    args = ap.parse_args(argv)
    if not (args.crema_d or args.manifest):
        ap.error("give --crema-d and/or --manifest")
    items = load_items(args)
    if not items:
        print("No labelled clips found.")
        return 1
    from anvesh.speech.models import emotion
    if emotion.get(block=True) is None:
        print(f"Voice emotion model not available: {emotion.status().error}")
        print("Install backend/requirements-ml.txt and run: python -m anvesh.download_models --only emotion")
        return 1
    print(f"Evaluating {len(items)} clips with {emotion.model_id} ...", flush=True)
    res = evaluate(items, emotion.predict, args.role)
    md = f"# Speech-emotion results ({emotion.model_id})\n\n" + summarise(res, args.role) + "\n"
    Path(args.out).write_text(md, encoding="utf-8")
    print(md)
    print(f"Saved to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
