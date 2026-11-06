"""Train the telephone-adapted emotion head (anvesh/data/models/ser_head_telephone.json).

Why: emotion2vec+ is accurate on studio audio but loses a lot of accuracy on telephone-band audio and stays
over-confident there (see EVALUATION.md). Helpline calls are telephone audio, so the live system uses a small
linear head on emotion2vec+'s own 768-d embedding, trained on clean, telephone and noisy-telephone versions of
CREMA-D speech.

Speaker-disjoint by construction (no speaker appears in more than one split):
  test        speakers whose id is divisible by 3  -> never touched here; used by run_ser_eval.py --holdout
  validation  other speakers with id % 9 in (1, 2) -> choose regularisation and the temperature (calibration)
  train       the remaining speakers

  python eval/get_eval_audio.py --crema-all          # all CREMA-D clips of the four classes (~4,900)
  python eval/train_ser_head.py --crema-d eval/audio/crema-d
  python eval/run_ser_eval.py --crema-d eval/audio/crema-d --holdout --limit 800

Embeddings are cached in eval/audio/.cache/, so re-fitting takes seconds. The output is plain JSON (no pickle).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel import CONDITIONS, degrade, trim_silence  # noqa: E402

from anvesh.speech import audio as A  # noqa: E402

CLASSES = ("neutral", "angry", "fearful", "sad")
CREMA = {"NEU": "neutral", "ANG": "angry", "FEA": "fearful", "SAD": "sad"}
# how often each condition is used for a training clip (helpline audio is mostly telephone audio)
MIX = {"clean": 0.25, "phone": 0.35, "phone_snr20": 0.15, "phone_snr10": 0.15, "phone_snr5": 0.10}
MAX_FALSE_ALARM = 0.20  # calm telephone speech flagged as needing attention, on validation speakers
OUT = Path(__file__).resolve().parents[1] / "anvesh" / "data" / "models" / "ser_head_telephone.json"
CACHE = Path(__file__).with_name("audio") / ".cache"


def split_of(speaker: int) -> str:
    if speaker % 3 == 0:
        return "test"
    return "val" if speaker % 9 in (1, 2) else "train"


def extract(folder: Path, copies: int, seed: int, partial: Path, budget_s: float = 0) -> dict | None:
    """Embeddings for every train / validation clip. Resumable: progress is saved to `partial`."""
    from anvesh.speech.models import emotion
    if emotion.get(block=True) is None:
        raise SystemExit(f"Voice emotion model not available: {emotion.status().error}")
    files = sorted(p for p in folder.glob("*.wav")
                   if len(p.stem.split("_")) >= 3 and p.stem.split("_")[2] in CREMA
                   and split_of(int(p.stem.split("_")[0])) != "test")
    conds, probs = list(MIX), np.array(list(MIX.values()))
    rows: dict[str, list] = {"X": [], "y": [], "speaker": [], "cond": [], "fname": []}
    if partial.exists():
        z = np.load(partial, allow_pickle=False)
        rows = {k: list(z[k]) for k in rows}
    done = set(rows["fname"]) | set(json.loads(partial.with_suffix(".skipped.json").read_text())
                                   if partial.with_suffix(".skipped.json").exists() else [])
    skipped = []
    t0 = time.time()

    def save() -> None:
        np.savez(partial, **{k: (np.array(v, dtype=np.float32) if k == "X" else np.array(v)) for k, v in rows.items()})
        prev = (json.loads(partial.with_suffix(".skipped.json").read_text())
                if partial.with_suffix(".skipped.json").exists() else [])
        partial.with_suffix(".skipped.json").write_text(json.dumps(sorted(set(prev) | set(skipped))))

    todo = [(i, p) for i, p in enumerate(files) if p.name not in done]
    for n, (i, p) in enumerate(todo):
        rng = np.random.default_rng([seed, i])  # same condition for a clip on every run
        x0 = trim_silence(A.decode(p.read_bytes()))
        added = False
        for k in range(copies):
            c = "clean" if (copies > 1 and k == 0) else str(rng.choice(conds, p=probs / probs.sum()))
            x = degrade(x0, c, seed=seed * 100000 + i * 10 + k)
            f = A.features(x) or {}
            if not f or f.get("rms", 0.0) < 0.006:
                continue
            _, feats = emotion.predict_full(x, block=True)
            if feats is None:
                continue
            rows["X"].append(feats)
            rows["y"].append(CLASSES.index(CREMA[p.stem.split("_")[2]]))
            rows["speaker"].append(int(p.stem.split("_")[0]))
            rows["cond"].append(c)
            rows["fname"].append(p.name)
            added = True
        if not added:
            skipped.append(p.name)
        if (n + 1) % 100 == 0:
            save()
            print(f"  {len(done) + n + 1}/{len(files)} clips, {time.time() - t0:.0f}s", flush=True)
        if budget_s and time.time() - t0 > budget_s:
            save()
            print(f"  paused at {len(done) + n + 1}/{len(files)}; run again to continue", flush=True)
            return None
    save()
    return {"X": np.array(rows["X"], dtype=np.float32), "y": np.array(rows["y"]), "speaker": np.array(rows["speaker"]),
            "cond": np.array(rows["cond"]), "model": emotion.model_id}


def _nll(p: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def fit(d: dict) -> tuple[dict, str]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    split = np.array([split_of(int(s)) for s in d["speaker"]])
    tr, va = split == "train", split == "val"
    sc = StandardScaler().fit(d["X"][tr])
    Xtr, Xva = sc.transform(d["X"][tr]), sc.transform(d["X"][va])
    best = None
    lines = ["| C | validation accuracy | validation NLL |", "|---|---|---|"]
    for C in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced").fit(Xtr, d["y"][tr])
        acc = float(np.mean(clf.predict(Xva) == d["y"][va]))
        nll = _nll(clf.predict_proba(Xva), d["y"][va])
        lines.append(f"| {C} | {100 * acc:.1f}% | {nll:.3f} |")
        if best is None or acc > best[0] + 1e-9 or (abs(acc - best[0]) < 1e-9 and nll < best[1]):
            best = (acc, nll, C, clf)
    acc, _, C, clf = best
    logits = Xva @ clf.coef_.T + clf.intercept_
    temps = np.linspace(0.5, 4.0, 36)
    T = float(temps[int(np.argmin([_nll(_softmax(logits / t), d["y"][va]) for t in temps]))])
    # Operating point (validation only): the smallest log-prior on "neutral" that keeps false alarms on calm
    # telephone speech at or below MAX_FALSE_ALARM without lowering balanced accuracy.
    P = _softmax(logits / T)
    phone = va & (d["cond"] == "phone")
    yv, yp = d["y"][va], d["y"][phone]
    Pp = _softmax(sc.transform(d["X"][phone]) @ clf.coef_.T / T + clf.intercept_ / T)

    def op(b: float) -> tuple[float, float, float]:
        q = P.copy()
        q[:, 0] *= np.exp(b)
        pred = q.argmax(1)
        bal = float(np.mean([np.mean(pred[yv == c] == c) for c in range(len(CLASSES))]))
        qp = Pp.copy()
        qp[:, 0] *= np.exp(b)
        pp = qp.argmax(1)
        fa = float(np.mean(pp[yp == 0] != 0)) if (yp == 0).any() else 0.0
        rec = float(np.mean(pp[yp != 0] != 0)) if (yp != 0).any() else 0.0
        return bal, fa, rec
    bal0 = op(0.0)[0]
    prior, sweep_rows = 0.0, []
    for b in np.round(np.arange(0.0, 2.01, 0.1), 2):
        bal, fa, rec = op(float(b))
        sweep_rows.append(f"| {b:.1f} | {100 * bal:.1f}% | {100 * fa:.1f}% | {rec:.2f} |")
        if fa <= MAX_FALSE_ALARM and bal >= bal0 - 1e-9:
            prior = float(b)
            break
    lines += ["", "| neutral log-prior | validation balanced accuracy | phone false alarms on neutral | phone attention recall |",
              "|---|---|---|---|", *sweep_rows]
    per_cond = {}
    for c in CONDITIONS:
        m = va & (d["cond"] == c)
        if m.any():
            per_cond[c] = round(float(np.mean(clf.predict(sc.transform(d["X"][m])) == d["y"][m])), 4)
    spec = {
        "name": "telephone-adapted head v1.1",
        # the model id only, never a local cache path
        "base_model": "iic/emotion2vec_plus_base",
        "trained_on": "CREMA-D (acted English, ODbL), clean + simulated telephone channel + pink noise",
        "split": "speaker-disjoint: test = id % 3 == 0 (held out), validation = id % 9 in (1, 2), train = rest",
        "train_speakers": int(len(set(d["speaker"][tr]))), "train_clips": int(tr.sum()),
        "validation_speakers": int(len(set(d["speaker"][va]))), "validation_clips": int(va.sum()),
        "conditions": MIX, "C": C, "temperature": round(T, 3), "neutral_log_prior": prior,
        "operating_point": f"smallest neutral log-prior with phone false alarms <= {MAX_FALSE_ALARM:.0%} on validation "
                           "and no loss of balanced accuracy",
        "validation": {"accuracy": round(acc, 4), "accuracy_by_condition": per_cond},
        "created": date.today().isoformat(),
        "classes": list(CLASSES),
        "mean": [round(float(v), 6) for v in sc.mean_],
        "scale": [round(float(v), 6) for v in sc.scale_],
        "coef": [[round(float(v), 6) for v in row] for row in clf.coef_],
        "intercept": [round(float(v), 6) for v in clf.intercept_],
    }
    return spec, "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crema-d", required=True, help="folder with CREMA-D wav files (all four classes)")
    ap.add_argument("--copies", type=int, default=1, help="augmented copies per training clip")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--refit", action="store_true", help="reuse cached embeddings")
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (resume on the next run)")
    a = ap.parse_args(argv)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache = CACHE / f"train_feats_c{a.copies}_s{a.seed}.npz"
    if a.refit and cache.exists():
        z = np.load(cache, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["model"] = str(d["model"])
    else:
        d = extract(Path(a.crema_d), a.copies, a.seed, CACHE / f"partial_c{a.copies}_s{a.seed}.npz", a.budget)
        if d is None:
            return 2
        np.savez_compressed(cache, **{k: (np.array(v) if k == "model" else v) for k, v in d.items()})
    print(f"{len(d['y'])} embeddings from {len(set(d['speaker']))} speakers", flush=True)
    spec, table = fit(d)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
    print(table)
    print(f"chosen C={spec['C']}, temperature={spec['temperature']}, neutral prior={spec['neutral_log_prior']}, "
          f"validation={spec['validation']}")
    print(f"Saved {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
