"""Multilingual telephone head: train on many languages, test on languages it has never heard.

  python eval/multilingual_head.py extract --manifest eval/audio/multilingual/cameo/manifest.csv
  python eval/multilingual_head.py extract --manifest eval/audio/multilingual/rasa/manifest.csv
  python eval/multilingual_head.py lolo          # leave-one-language-out report (no model needed)
  python eval/multilingual_head.py build         # write anvesh/data/models/ser_head_telephone.json (multilingual)

Embeddings (emotion2vec+, 768-d) are computed once per clip for the clean audio and for one simulated phone-line
version, and cached in eval/audio/.cache/ml_feats.npz. English comes from the CREMA-D training and validation
speakers already cached by train_ser_head.py; CREMA-D test speakers (id divisible by 3) are never used here.

Leave-one-language-out: for each language L, a head is trained on every other language (and English) and scored
on L only. That is the honest estimate for a language Anvesh has no emotion data for.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel import degrade, trim_silence  # noqa: E402

from anvesh.speech import audio as A  # noqa: E402

CLASSES = ("neutral", "angry", "fearful", "sad")
CACHE = Path(__file__).with_name("audio") / ".cache"
ML = CACHE / "ml_feats.npz"
CREMA = CACHE / "train_feats_c1_s7.npz"
OUT = Path(__file__).resolve().parents[1] / "anvesh" / "data" / "models" / "ser_head_telephone.json"
RESULTS = Path(__file__).with_name("results") / "multilingual"
PHONE_MIX = ("phone", "phone_snr20", "phone_snr10")


def _load_ml() -> dict:
    if not ML.exists():
        return {k: [] for k in ("X", "y", "lang", "speaker", "cond", "key", "raw")}
    z = np.load(ML, allow_pickle=False)
    return {k: list(z[k]) for k in z.files}


def _save_ml(d: dict) -> None:
    CACHE.mkdir(parents=True, exist_ok=True)
    np.savez(ML, **{k: (np.array(v, dtype=np.float32) if k == "X" else np.array(v)) for k, v in d.items()})


def extract(manifest: Path, budget: float) -> None:
    from anvesh.speech.models import emotion
    if emotion.get(block=True) is None:
        raise SystemExit(f"Voice emotion model not available: {emotion.status().error}")
    d = _load_ml()
    done = set(d["key"])
    base = manifest.parent
    rows = list(csv.DictReader(open(manifest, newline="", encoding="utf-8")))
    t0, n = time.time(), 0
    for i, r in enumerate(rows):
        if r["label"] not in CLASSES or r["lang"].lower() == "english":
            continue
        x0 = None
        for cond in ("clean", PHONE_MIX[i % len(PHONE_MIX)]):
            key = f"{r['source']}|{r['path']}|{cond}"
            if key in done:
                continue
            if x0 is None:
                x0 = trim_silence(A.decode((base / r["path"]).read_bytes()))
            x = degrade(x0, cond, seed=7000 + i)
            f = A.features(x) or {}
            if not f or f.get("rms", 0.0) < 0.006:
                continue
            raw, feats = emotion.predict_full(x, block=True)
            if feats is None:
                continue
            d["X"].append(feats)
            d["y"].append(CLASSES.index(r["label"]))
            d["lang"].append(r["lang"].lower())
            d["speaker"].append(r.get("speaker") or "?")
            d["cond"].append(cond)
            d["key"].append(key)
            d["raw"].append(json.dumps(raw or {}))
            done.add(key)
            n += 1
        if n and n % 100 == 0:
            _save_ml(d)
            print(f"  {i + 1}/{len(rows)} clips, {n} new embeddings, {time.time() - t0:.0f}s", flush=True)
        if budget and time.time() - t0 > budget:
            break
    _save_ml(d)
    print(f"{len(d['key'])} embeddings cached; languages: {dict(Counter(d['lang']))}")


def _crema() -> dict:
    z = np.load(CREMA, allow_pickle=False)
    spk = z["speaker"]
    keep = spk % 3 != 0  # never the CREMA-D test speakers
    return {"X": z["X"][keep], "y": z["y"][keep], "lang": np.array(["english"] * int(keep.sum())),
            "cond": z["cond"][keep], "speaker": spk[keep].astype(str),
            "val": (spk[keep] % 9 == 1) | (spk[keep] % 9 == 2)}


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def _fit(X, y, C=0.3):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    return sc, clf


def _probs(sc, clf, X, T=1.0, prior=0.0):
    z = (sc.transform(X) @ clf.coef_.T + clf.intercept_) / T
    z[:, 0] += prior
    return _softmax(z)


def _metrics(P, y) -> dict:
    pred = P.argmax(1)
    att = (y != 0)
    return {"n": int(len(y)), "accuracy": float(np.mean(pred == y)),
            "attention_recall": float(np.mean(pred[att] != 0)) if att.any() else None,
            "false_alarm": float(np.mean(pred[~att] != 0)) if (~att).any() else None,
            "balanced_accuracy": float(np.mean([np.mean(pred[y == c] == c) for c in range(4) if (y == c).any()]))}


def _raw_probs(raw_json: list[str]) -> np.ndarray:
    out = []
    for s in raw_json:
        e = json.loads(s) if s else {}
        e = {k: v for k, v in e.items() if k in CLASSES or k == "disgusted"}
        if "disgusted" in e:
            e["angry"] = e.get("angry", 0.0) + 0.5 * e.pop("disgusted")
        t = sum(e.values()) or 1.0
        out.append([e.get(c, 0.0) / t for c in CLASSES])
    return np.array(out)


def _calibrate(sc, clf, X, y):
    temps = np.linspace(0.5, 4.0, 36)
    nll = [float(-np.mean(np.log(np.clip(_probs(sc, clf, X, t)[np.arange(len(y)), y], 1e-9, 1)))) for t in temps]
    return float(temps[int(np.argmin(nll))])


def _prior(sc, clf, X, y, cond, T, max_fa=0.20) -> float:
    phone = np.isin(cond, list(PHONE_MIX) + ["phone_snr5"])
    base = _metrics(_probs(sc, clf, X, T), y)["balanced_accuracy"]
    for b in np.round(np.arange(0.0, 2.01, 0.1), 2):
        m_all = _metrics(_probs(sc, clf, X, T, b), y)
        m_ph = _metrics(_probs(sc, clf, X[phone], T, b), y[phone]) if phone.any() else m_all
        if (m_ph["false_alarm"] or 0) <= max_fa and m_all["balanced_accuracy"] >= base - 1e-9:
            return float(b)
    return 0.0


def lolo() -> dict:
    """Leave-one-language-out over the extra languages; English is always in training."""
    from anvesh.speech import ser_head
    cr, ml = _crema(), _load_ml()
    X = np.array(ml["X"], dtype=np.float32)
    y, lang, cond = np.array(ml["y"]), np.array(ml["lang"]), np.array(ml["cond"])
    raw = _raw_probs(ml["raw"])
    english_only = ser_head.load()  # the shipped head before this step (may already be multilingual)
    langs = sorted(set(lang))
    rows, out = [], {}
    for L in langs:
        tr, te = lang != L, lang == L
        Xtr = np.concatenate([cr["X"][~cr["val"]], X[tr]])
        ytr = np.concatenate([cr["y"][~cr["val"]], y[tr]])
        sc, clf = _fit(Xtr, ytr)
        Xv = np.concatenate([cr["X"][cr["val"]], X[tr]])
        yv = np.concatenate([cr["y"][cr["val"]], y[tr]])
        cv = np.concatenate([cr["cond"][cr["val"]], cond[tr]])
        T = _calibrate(sc, clf, Xv, yv)
        b = _prior(sc, clf, Xv, yv, cv, T)
        res = {}
        for cname, m in (("clean", cond == "clean"), ("phone", np.isin(cond, PHONE_MIX))):
            sel = te & m
            if not sel.any():
                continue
            res[cname] = {"raw": _metrics(raw[sel], y[sel]),
                          "multilingual": _metrics(_probs(sc, clf, X[sel], T, b), y[sel])}
            if english_only is not None:
                Pe = np.array([[english_only.predict(v)[c] for c in CLASSES] for v in X[sel]])
                res[cname]["english_only"] = _metrics(Pe, y[sel])
        out[L] = res
        spk = len(set(np.array(ml["speaker"])[te]))
        for cname, r in res.items():
            e = r.get("english_only", {})
            rows.append(f"| {L} | {spk} | {cname} | {100 * r['raw']['accuracy']:.0f}% | "
                        f"{100 * e.get('accuracy', float('nan')):.0f}% | **{100 * r['multilingual']['accuracy']:.0f}%** | "
                        f"{r['raw']['attention_recall']:.2f} / {e.get('attention_recall', float('nan')):.2f} / "
                        f"**{r['multilingual']['attention_recall']:.2f}** | "
                        f"{100 * r['raw']['false_alarm']:.0f}% / {100 * e.get('false_alarm', float('nan')):.0f}% / "
                        f"{100 * r['multilingual']['false_alarm']:.0f}% |")
    avg = {}
    for cname in ("clean", "phone"):
        for sysk in ("raw", "english_only", "multilingual"):
            vals = [out[L][cname][sysk] for L in out if cname in out[L] and sysk in out[L][cname]]
            if vals:
                avg[(cname, sysk)] = {k: float(np.mean([v[k] for v in vals if v[k] is not None]))
                                      for k in ("accuracy", "attention_recall", "false_alarm", "balanced_accuracy")}
    md = ["# Leave-one-language-out: emotion on languages the head has never heard\n",
          "Each row: a head trained on English (CREMA-D train speakers) plus every OTHER language, scored only on this "
          "language. Columns compare the raw emotion2vec+ model / the English-only head / the multilingual head. "
          "Every clip answered (no abstention). Generated by `eval/multilingual_head.py lolo`.\n",
          "| Held-out language | Speakers | Audio | Accuracy raw | English-only head | **Multilingual head** | "
          "Attention recall raw / EN / ML | False alarms raw / EN / ML |",
          "|---|---|---|---|---|---|---|---|", *rows, "",
          "Average over held-out languages:\n", "| Audio | System | Accuracy | Balanced accuracy | Attention recall | "
          "False alarms |", "|---|---|---|---|---|---|"]
    for (cname, sysk), v in avg.items():
        md.append(f"| {cname} | {sysk} | {100 * v['accuracy']:.1f}% | {100 * v['balanced_accuracy']:.1f}% | "
                  f"{v['attention_recall']:.2f} | {100 * v['false_alarm']:.1f}% |")
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "lolo_results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    (RESULTS / "lolo_results.json").write_text(json.dumps({"per_language": out,
                                                           "average": {f"{a}/{b}": v for (a, b), v in avg.items()}},
                                                          indent=1), encoding="utf-8")
    print("\n".join(md))
    return out


def build() -> None:
    cr, ml = _crema(), _load_ml()
    X = np.array(ml["X"], dtype=np.float32)
    y, lang = np.array(ml["y"]), np.array(ml["lang"])
    Xtr, ytr = np.concatenate([cr["X"][~cr["val"]], X]), np.concatenate([cr["y"][~cr["val"]], y])
    sc, clf = _fit(Xtr, ytr)
    Xv, yv, cv = cr["X"][cr["val"]], cr["y"][cr["val"]], cr["cond"][cr["val"]]
    T = _calibrate(sc, clf, Xv, yv)
    b = _prior(sc, clf, Xv, yv, cv, T)
    langs = dict(Counter(lang))
    spec = {"name": "telephone-adapted head v2 (multilingual)", "base_model": "iic/emotion2vec_plus_base",
            "trained_on": "CREMA-D (English, ODbL) + " + ", ".join(sorted(langs)) + " (AI4Bharat Rasa CC-BY-4.0; "
                          "CAMEO corpora, research licences)",
            "languages": ["english", *sorted(langs)],
            "split": "speaker-disjoint for English (CREMA-D test speakers id % 3 == 0 never used); other languages "
                     "evaluated leave-one-language-out (eval/results/multilingual/lolo_results.md)",
            "train_clips": int(len(ytr)), "conditions": "clean + simulated phone line (+ noise)",
            "C": 0.3, "temperature": round(T, 3), "neutral_log_prior": b,
            "operating_point": "smallest neutral log-prior with phone false alarms <= 20% on CREMA-D validation "
                               "speakers and no loss of balanced accuracy",
            "created": date.today().isoformat(), "classes": list(CLASSES),
            "mean": [round(float(v), 6) for v in sc.mean_], "scale": [round(float(v), 6) for v in sc.scale_],
            "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
            "intercept": [round(float(v), 6) for v in clf.intercept_]}
    OUT.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    print(f"Saved {OUT}: {len(spec['languages'])} languages, {spec['train_clips']} clips, T={T}, prior={b}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["extract", "lolo", "build"])
    ap.add_argument("--manifest")
    ap.add_argument("--budget", type=float, default=0)
    a = ap.parse_args(argv)
    if a.cmd == "extract":
        if not a.manifest:
            ap.error("extract needs --manifest")
        extract(Path(a.manifest), a.budget)
    elif a.cmd == "lolo":
        lolo()
    else:
        build()
    return 0


if __name__ == "__main__":
    sys.exit(main())
