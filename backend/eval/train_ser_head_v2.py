"""Telephone head v2: a small neural network on emotion2vec+ embeddings, built fast with parallel workers.

Same speaker-disjoint rules as train_ser_head.py (test = speaker id % 3 == 0, never used here except to score the
final choice; validation = id % 9 in (1, 2); train = the rest). Steps, all resumable:

  python eval/train_ser_head_v2.py download            # every CREMA-D clip of the four classes (~4,900)
  python eval/train_ser_head_v2.py extract --workers 4 # embeddings: train/val (2 copies each), and the exact
                                                       # 800 held-out test inputs of run_ser_eval.py x 5 conditions
  python eval/train_ser_head_v2.py fit                 # choose on validation, write the head only if it is better
  python eval/run_ser_eval.py --from-cache             # every table again, from the rewritten predictions

Rule fixed before looking at any result: the network replaces the linear head only if, on validation speakers,
its balanced accuracy on telephone + noisy-telephone audio is at least 1 point higher than a linear head fitted on
the same embeddings, with calm-speech false alarms on telephone audio still at or below 20%. Otherwise nothing
changes. Test speakers are scored once, after the choice, and both heads' test results are written side by side.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

CLASSES = ("neutral", "angry", "fearful", "sad")
CREMA = {"NEU": "neutral", "ANG": "angry", "FEA": "fearful", "SAD": "sad"}
NOISY = ("phone_snr20", "phone_snr10", "phone_snr5")
TEST_CONDS = ("clean", "phone", "phone_snr20", "phone_snr10", "phone_snr5")
AUDIO = HERE / "audio" / "crema-d"
CACHE = HERE / "audio" / ".cache" / "v2"
RESULTS = HERE / "results"
HEAD_OUT = HERE.parent / "anvesh" / "data" / "models" / "ser_head_telephone.json"
MAX_FALSE_ALARM = 0.20


def split_of(speaker: int) -> str:
    if speaker % 3 == 0:
        return "test"
    return "val" if speaker % 9 in (1, 2) else "train"


def jobs() -> list[tuple]:
    """(key, file, condition, seed, label, speaker, split). Train/val: one clean-or-phone copy + one noisy copy.
    Test: exactly the inputs run_ser_eval.py used (file order and seed 1000 + i from predictions_clean.csv)."""
    out = []
    files = sorted(p.name for p in AUDIO.glob("*.wav") if len(p.stem.split("_")) >= 3 and p.stem.split("_")[2] in CREMA)
    for i, f in enumerate(files):
        spk = int(f.split("_")[0])
        sp = split_of(spk)
        if sp == "test":
            continue
        rng = np.random.default_rng([11, i])
        c0 = "clean" if rng.random() < 0.4 else "phone"
        c1 = str(rng.choice(NOISY))
        for k, c in enumerate((c0, c1)):
            out.append((f"{f}|{c}", f, c, 7 * 100000 + i * 10 + k, CREMA[f.split("_")[2]], spk, sp))
    test = list(csv.DictReader(open(RESULTS / "predictions_clean.csv", encoding="utf-8")))
    for c in TEST_CONDS:
        for i, r in enumerate(test):
            out.append((f"T|{r['file']}|{c}", r["file"], c, 1000 + i, r["label"], int(r["speaker"]), "test"))
    return out


def work(shard: int, of: int) -> None:
    import torch
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // of))
    from channel import degrade, trim_silence
    from anvesh.speech import audio as A
    from anvesh.speech.models import emotion
    if emotion.get(block=True) is None:
        raise SystemExit(f"voice emotion model not available: {emotion.status().error}")
    CACHE.mkdir(parents=True, exist_ok=True)
    part = CACHE / f"shard{shard}of{of}.npz"
    done: dict[str, np.ndarray] = {}
    if part.exists():
        z = np.load(part, allow_pickle=False)
        done = {k: v for k, v in zip(z["keys"], z["X"])}
    mine = [j for n, j in enumerate(jobs()) if n % of == shard and j[0] not in done]
    t0, dec = time.time(), {}

    def save():
        keys = list(done)
        np.savez(part, keys=np.array(keys), X=np.array([done[k] for k in keys], dtype=np.float32))

    for n, (key, f, c, seed, *_rest) in enumerate(mine):
        if f not in dec:
            dec = {f: trim_silence(A.decode((AUDIO / f).read_bytes()))}
        x = degrade(dec[f], c, seed=seed)
        feat = A.features(x) or {}
        silent = not feat or feat.get("rms", 0.0) < 0.006 or 1 - feat.get("pause_ratio", 1.0) < 0.35
        if silent:
            done[key] = np.full(768, np.nan, dtype=np.float32)
        else:
            _, emb = emotion.predict_full(x, block=True)
            done[key] = emb if emb is not None else np.full(768, np.nan, dtype=np.float32)
        if (n + 1) % 200 == 0:
            save()
            print(f"  worker {shard}: {n + 1}/{len(mine)}  {time.time() - t0:.0f}s", flush=True)
    save()
    print(f"  worker {shard}: done ({len(mine)} new) in {time.time() - t0:.0f}s", flush=True)


def extract(workers: int) -> None:
    import subprocess
    procs = [subprocess.Popen([sys.executable, __file__, "work", "--shard", str(k), "--of", str(workers)])
             for k in range(workers)]
    codes = [p.wait() for p in procs]
    if any(codes):
        raise SystemExit(f"a worker failed: exit codes {codes} (run extract again: it resumes)")


def load_all() -> dict:
    X = {}
    for f in CACHE.glob("shard*.npz"):
        z = np.load(f, allow_pickle=False)
        X.update(zip(z["keys"], z["X"]))
    return X


# --------------------------------------------------------------------------- fitting

def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def nll(p, y):
    return float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1))))


def bal_acc(pred, y):
    return float(np.mean([np.mean(pred[y == c] == c) for c in range(len(CLASSES)) if (y == c).any()]))


class Linear:
    kind = "linear"

    def __init__(self, C):
        from sklearn.linear_model import LogisticRegression
        self.C, self.m = C, LogisticRegression(C=C, max_iter=4000, class_weight="balanced")

    def fit(self, X, y):
        self.m.fit(X, y)
        return self

    def logits(self, X):
        return X @ self.m.coef_.T + self.m.intercept_

    def spec(self):
        return {"kind": "linear", "C": self.C, "coef": self.m.coef_.round(6).tolist(),
                "intercept": self.m.intercept_.round(6).tolist()}


class MLP:
    kind = "mlp"

    def __init__(self, hidden, alpha):
        from sklearn.neural_network import MLPClassifier
        self.hidden, self.alpha = hidden, alpha
        self.m = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=alpha, max_iter=400, early_stopping=True,
                               validation_fraction=0.1, n_iter_no_change=15, random_state=0)

    def fit(self, X, y):
        # class balance by resampling (MLPClassifier has no class_weight)
        rng = np.random.default_rng(0)
        n = np.bincount(y).max()
        idx = np.concatenate([rng.choice(np.where(y == c)[0], n, replace=True) for c in range(len(CLASSES))])
        self.m.fit(X[idx], y[idx])
        return self

    def logits(self, X):
        h = np.maximum(0, X @ self.m.coefs_[0] + self.m.intercepts_[0])
        return h @ self.m.coefs_[1] + self.m.intercepts_[1]

    def spec(self):
        return {"kind": "mlp", "hidden": self.hidden, "alpha": self.alpha,
                "layers": [{"W": self.m.coefs_[0].T.round(5).tolist(), "b": self.m.intercepts_[0].round(5).tolist()},
                           {"W": self.m.coefs_[1].T.round(5).tolist(), "b": self.m.intercepts_[1].round(5).tolist()}]}


def fit(write: bool) -> int:
    from sklearn.preprocessing import StandardScaler
    X = load_all()
    J = [j for j in jobs() if j[0] in X and np.all(np.isfinite(X[j[0]]))]
    print(f"usable embeddings: {len(J)} of {len(jobs())}")
    tr = [j for j in J if j[6] == "train"]
    va = [j for j in J if j[6] == "val"]
    M = lambda js: np.stack([X[j[0]] for j in js])  # noqa: E731
    Y = lambda js: np.array([CLASSES.index(j[4]) for j in js])  # noqa: E731
    sc = StandardScaler().fit(M(tr))
    Xtr, ytr, Xva, yva = sc.transform(M(tr)), Y(tr), sc.transform(M(va)), Y(va)
    va_cond = np.array([j[2] for j in va])
    tel = va_cond != "clean"
    phone = va_cond == "phone"

    def score(model):
        L = model.logits(Xva)
        pred = L.argmax(1)
        return bal_acc(pred[tel], yva[tel]), model

    lines = ["| head | setting | validation balanced accuracy (telephone + noisy) |", "|---|---|---|"]
    lin = [score(Linear(C).fit(Xtr, ytr)) for C in (0.003, 0.01, 0.03, 0.1)]
    for s, m in lin:
        lines.append(f"| linear | C={m.C} | {100 * s:.1f}% |")
    mlp = [score(MLP(h, a).fit(Xtr, ytr)) for h in (128, 256) for a in (1e-3, 1e-2, 1e-1)]
    for s, m in mlp:
        lines.append(f"| network | hidden={m.hidden}, alpha={m.alpha} | {100 * s:.1f}% |")
    best_lin, best_mlp = max(lin, key=lambda t: t[0]), max(mlp, key=lambda t: t[0])

    def calibrate(model):
        L = model.logits(Xva)
        temps = np.linspace(0.5, 4.0, 36)
        T = float(temps[int(np.argmin([nll(softmax(L / t), yva) for t in temps]))])
        P = softmax(L / T)

        def op(b):
            q = P.copy()
            q[:, 0] *= np.exp(b)
            pred = q.argmax(1)
            pp = pred[phone]
            yp = yva[phone]
            return bal_acc(pred, yva), float(np.mean(pp[yp == 0] != 0)), float(np.mean(pp[yp != 0] != 0))
        b0 = op(0.0)[0]
        prior = 0.0
        for b in np.round(np.arange(0.0, 2.01, 0.1), 2):
            bal, fa, _ = op(float(b))
            if fa <= MAX_FALSE_ALARM and bal >= b0 - 1e-9:
                prior = float(b)
                break
        return T, prior, op(prior)

    Tl, pl, opl = calibrate(best_lin[1])
    Tm, pm, opm = calibrate(best_mlp[1])
    choose_mlp = best_mlp[0] >= best_lin[0] + 0.01 and opm[1] <= MAX_FALSE_ALARM
    lines += ["", f"Best linear: {100 * best_lin[0]:.1f}% · best network: {100 * best_mlp[0]:.1f}% "
              f"(validation, telephone + noisy). Phone false alarms after the operating point: linear "
              f"{100 * opl[1]:.1f}%, network {100 * opm[1]:.1f}%.",
              f"**Choice (rule fixed in advance): {'network' if choose_mlp else 'linear'}.**"]
    model, T, prior = (best_mlp[1], Tm, pm) if choose_mlp else (best_lin[1], Tl, pl)
    spec = {"name": f"telephone-adapted head v2 ({'network' if choose_mlp else 'linear'})",
            "base_model": "iic/emotion2vec_plus_base",
            "trained_on": "CREMA-D (acted English, ODbL), clean + simulated telephone channel + pink noise",
            "split": "speaker-disjoint: test = id % 3 == 0 (held out), validation = id % 9 in (1, 2), train = rest",
            "train_speakers": len({j[5] for j in tr}), "train_clips": len(tr),
            "validation_speakers": len({j[5] for j in va}), "validation_clips": len(va),
            "temperature": round(T, 3), "neutral_log_prior": prior,
            "validation": {"balanced_accuracy_telephone": round(max(best_lin[0], best_mlp[0]) if choose_mlp else best_lin[0], 4)},
            "created": date.today().isoformat(), "classes": list(CLASSES),
            "mean": sc.mean_.round(6).tolist(), "scale": sc.scale_.round(6).tolist(), **model.spec()}

    # test speakers, scored once: rewrite the "head" column of the saved predictions for both heads
    test_rows = {c: list(csv.DictReader(open(RESULTS / f"predictions_{c}.csv", encoding="utf-8"))) for c in TEST_CONDS}
    report = ["", "| Test condition (held-out speakers) | v1.1 accuracy | v2 accuracy | v1.1 fear / sad recall | v2 fear / sad recall |",
              "|---|---|---|---|---|"]
    new_cols = {}
    for c in TEST_CONDS:
        old_pred, new_pred, ys, col = [], [], [], []
        for r in test_rows[c]:
            k = f"T|{r['file']}|{c}"
            v = X.get(k)
            if v is None or not np.all(np.isfinite(v)) or not r["head"]:
                col.append(r["head"])
                continue
            L = model.logits(sc.transform(v[None]))[0] / T
            L[0] += prior
            p = np.exp(L - L.max())
            p /= p.sum()
            d = {cl: round(float(x), 4) for cl, x in zip(CLASSES, p)}
            col.append(json.dumps(d))
            h = json.loads(r["head"])
            old_pred.append(CLASSES.index(max(h, key=h.get)))
            new_pred.append(int(p.argmax()))
            ys.append(CLASSES.index(r["label"]))
        new_cols[c] = col
        o, nw, ys = np.array(old_pred), np.array(new_pred), np.array(ys)
        rec = lambda pr, k: float(np.mean(pr[ys == k] == k))  # noqa: E731
        report.append(f"| {c} | {100 * np.mean(o == ys):.1f}% | {100 * np.mean(nw == ys):.1f}% | "
                      f"{rec(o, 2):.2f} / {rec(o, 3):.2f} | {rec(nw, 2):.2f} / {rec(nw, 3):.2f} |")
    md = "# Telephone head v2\n\nGenerated by `eval/train_ser_head_v2.py fit`.\n\n" + "\n".join(lines + report) + "\n"
    (RESULTS / "ser_head_v2.md").write_text(md, encoding="utf-8")
    print(md)
    if write:
        import shutil
        bak = RESULTS / "v1_1"
        bak.mkdir(exist_ok=True)
        for c in TEST_CONDS:
            shutil.copy(RESULTS / f"predictions_{c}.csv", bak / f"predictions_{c}.csv")
        shutil.copy(HEAD_OUT, bak / "ser_head_telephone_v1_1.json")
        HEAD_OUT.write_text(json.dumps(spec, separators=(",", ":")), encoding="utf-8")
        for c in TEST_CONDS:
            rows = test_rows[c]
            for r, v in zip(rows, new_cols[c]):
                r["head"] = v
            with open(RESULTS / f"predictions_{c}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        print(f"wrote {HEAD_OUT.name} (v1.1 and its predictions kept in results/v1_1/)")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["download", "extract", "work", "fit"])
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 4) // 2)))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--write", action="store_true", help="install the chosen head (keeps v1.1 in results/v1_1)")
    a = ap.parse_args(argv)
    if a.step == "download":
        from get_eval_audio import crema_all
        crema_all()
    elif a.step == "extract":
        print(f"{len(jobs())} embeddings to compute with {a.workers} workers ({os.cpu_count()} CPU threads)")
        extract(a.workers)
    elif a.step == "work":
        work(a.shard, a.of)
    else:
        return fit(a.write)
    return 0


if __name__ == "__main__":
    sys.exit(main())
