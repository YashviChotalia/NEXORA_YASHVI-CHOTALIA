"""Speech-emotion evaluation on labelled audio clips (needs the voice-emotion model).

Measures the emotion step that feeds the live fear / anger / distress / nervousness signals, using the
same decision rules as the live call (silence abstains, top probability < UNCERTAIN_P is "uncertain"):

  1. Accuracy, per-class recall, confusion matrix and "needs attention" P / R / F1
     (attention = fear, anger or sadness; the decision that matters for prioritising a call).
  2. Robustness on helpline audio: the same clips through a simulated telephone channel
     (8 kHz, 300-3400 Hz, G.711 mu-law) and with background noise at 20 / 10 / 5 dB SNR.
  3. Calibration: expected calibration error, and a sweep of the abstention threshold
     (coverage vs selective accuracy), so the threshold used live is chosen from data.
  4. Fairness: results per speaker sex, age band and race (CREMA-D demographics).
  5. The live indicators themselves: ROC-AUC of the fear, anger and distress indicators.
  6. Model time per clip on this machine.

Input, either:
  --crema-d DIR      a CREMA-D AudioWAV folder (labels from file names, e.g. 1001_DFA_ANG_XX.wav).
                     Put VideoDemographics.csv in the same folder for the fairness table
                     (eval/get_eval_audio.py downloads both).
  --manifest FILE    a CSV with columns: path,label[,source,lang,speaker,sex,age]
                     label one of: neutral, angry, fearful, sad (also: anger, fear, sadness, distress)

Examples:
  python eval/run_ser_eval.py --crema-d eval/audio/crema-d                       # clean + telephone + noise
  python eval/run_ser_eval.py --crema-d eval/audio/crema-d --conditions clean    # quick run
  python eval/run_ser_eval.py --manifest eval/audio/roleplay/manifest.csv --conditions clean,phone
  python eval/run_ser_eval.py --from-cache                                       # re-summarise, no model

Writes eval/results/ser_results.md (tables), eval/results/ser_results.json (numbers) and
eval/results/predictions_<condition>.csv (per-clip model output, no audio) so every number can be re-derived.
Audio clips are NOT stored in this repository (licences and privacy).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from anvesh.live_ser import UNCERTAIN_P, adjust_for_role, indicators  # noqa: E402
from anvesh.speech import audio as A  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from channel import CONDITION_LABEL, CONDITIONS, degrade, trim_silence  # noqa: E402

CLASSES = ("neutral", "angry", "fearful", "sad")
ALIASES = {"anger": "angry", "ang": "angry", "fear": "fearful", "fea": "fearful", "sadness": "sad",
           "distress": "sad", "neu": "neutral", "calm": "neutral"}
CREMA = {"NEU": "neutral", "ANG": "angry", "FEA": "fearful", "SAD": "sad"}  # HAP / DIS are not used
ATTENTION = {"angry", "fearful", "sad"}
HOLDOUT = "speaker id divisible by 3"  # CREMA-D speakers kept out of head training (see train_ser_head.py)


def is_holdout(speaker: str) -> bool:
    return speaker.isdigit() and int(speaker) % 3 == 0
SNR_GATE_DB = 8.0  # same audio-quality gate as the distress track (distress.assess)
OUT_DIR = Path(__file__).with_name("results")


# --------------------------------------------------------------------------- data

def norm_label(s: str) -> str | None:
    s = s.strip().lower()
    s = ALIASES.get(s, s)
    return s if s in CLASSES else None


def age_band(age) -> str:
    try:
        a = int(float(age))
    except (TypeError, ValueError):
        return "?"
    return "20-29" if a < 30 else "30-49" if a < 50 else "50+"


def _demographics(folder: Path) -> dict[str, dict]:
    for f in (folder / "VideoDemographics.csv", folder.parent / "VideoDemographics.csv"):
        if f.exists():
            with open(f, newline="", encoding="utf-8") as fh:
                return {r["ActorID"].strip(): r for r in csv.DictReader(fh)}
    return {}


def load_items(args) -> list[dict]:
    items = []
    if args.crema_d:
        folder = Path(args.crema_d)
        demo = _demographics(folder)
        for p in sorted(folder.glob("*.wav")):
            parts = p.stem.split("_")
            if len(parts) >= 3 and parts[2] in CREMA:
                d = demo.get(parts[0], {})
                items.append({"path": str(p), "label": CREMA[parts[2]], "source": "crema-d", "lang": "en",
                              "speaker": parts[0], "sex": (d.get("Sex") or "?").lower(),
                              "age": age_band(d.get("Age")), "race": d.get("Race") or "?"})
    if args.manifest:
        base = Path(args.manifest).parent
        with open(args.manifest, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                lab = norm_label(r.get("label", ""))
                if lab:
                    p = Path(r["path"])
                    items.append({"path": str(p if p.is_absolute() else base / p), "label": lab,
                                  "source": r.get("source") or "manifest", "lang": r.get("lang") or "?",
                                  "speaker": r.get("speaker") or "?", "sex": (r.get("sex") or "?").lower(),
                                  "age": age_band(r.get("age")), "race": "?"})
    if getattr(args, "holdout", False):
        items = [it for it in items if it["source"] != "crema-d" or is_holdout(it["speaker"])]
    if args.limit and len(items) > args.limit:  # balanced random sample
        random.seed(7)
        by = defaultdict(list)
        for it in items:
            by[it["label"]].append(it)
        per = max(1, args.limit // len(by))
        items = [x for v in by.values() for x in random.sample(v, min(per, len(v)))]
    return items


# --------------------------------------------------------------------------- decision rules

def collapse(emo: dict[str, float] | None) -> dict[str, float] | None:
    """Model classes -> the four evaluated classes, as the live mapping does (half of disgust is anger)."""
    if not emo:
        return None
    e = {k: v for k, v in emo.items() if k in CLASSES or k == "disgusted"}
    if "disgusted" in e:
        e["angry"] = e.get("angry", 0.0) + 0.5 * e.pop("disgusted")
    return e or None


SYSTEM = {"key": "emotions"}  # which column is scored: "emotions" (raw model) or "head" (telephone-adapted)


def decide(row: dict, role: str, threshold: float = UNCERTAIN_P, quality_gate: bool = False) -> tuple[str, float | None]:
    """Same rules as the live tracker. Returns (label | 'uncertain' | 'abstain', top probability)."""
    emo = row.get(SYSTEM["key"]) or (row["emotions"] if SYSTEM["key"] == "head" and not row.get("head") else None)
    if row["silent"] or not emo:
        return "abstain", None
    if quality_gate and row["snr_db"] is not None and row["snr_db"] < SNR_GATE_DB:
        return "abstain", None
    e = collapse(adjust_for_role(emo, role))
    if not e:
        return "abstain", None
    label, p = max(e.items(), key=lambda kv: kv[1])
    return (label if p >= threshold else "uncertain"), p


# --------------------------------------------------------------------------- run the model

def run_condition(items: list[dict], emotion, head, condition: str, feats_out: dict | None = None) -> list[dict]:
    rows = []
    for i, it in enumerate(items):
        x = trim_silence(A.decode(Path(it["path"]).read_bytes()))  # as the live voice-activity cut does
        x = degrade(x, condition, seed=1000 + i)
        f = A.features(x) or {}
        silent = not f or f.get("rms", 0.0) < 0.006 or 1 - f.get("pause_ratio", 1.0) < 0.35
        emo = adapted = None
        ms = None
        if not silent:
            t = time.perf_counter()
            emo, feats = emotion.predict_full(x, block=True)
            adapted = head.predict(feats) if (head is not None and feats is not None) else None
            ms = round(1000 * (time.perf_counter() - t), 1)
            if feats_out is not None and feats is not None:
                feats_out[Path(it["path"]).name] = feats
        rows.append({**{k: it[k] for k in ("label", "source", "lang", "speaker", "sex", "age", "race")},
                     "file": Path(it["path"]).name, "silent": silent, "snr_db": f.get("snr_db"),
                     "emotions": emo, "head": adapted, "ms": ms})
        if (i + 1) % 100 == 0:
            print(f"  {condition}: {i + 1}/{len(items)}", flush=True)
    return rows


def save_predictions(rows: list[dict], condition: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    with open(OUT_DIR / f"predictions_{condition}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "label", "source", "lang", "speaker", "sex", "age", "race", "silent", "snr_db",
                    "ms", "emotions", "head"])
        for r in rows:
            w.writerow([r["file"], r["label"], r["source"], r["lang"], r["speaker"], r["sex"], r["age"], r["race"],
                        int(r["silent"]), "" if r["snr_db"] is None else round(r["snr_db"], 2),
                        "" if r["ms"] is None else r["ms"], json.dumps(r["emotions"]) if r["emotions"] else "",
                        json.dumps(r["head"]) if r.get("head") else ""])


def load_predictions(condition: str) -> list[dict] | None:
    f = OUT_DIR / f"predictions_{condition}.csv"
    if not f.exists():
        return None
    rows = []
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({**r, "silent": r["silent"] == "1",
                         "snr_db": float(r["snr_db"]) if r["snr_db"] else None,
                         "ms": float(r["ms"]) if r["ms"] else None,
                         "emotions": json.loads(r["emotions"]) if r["emotions"] else None,
                         "head": json.loads(r["head"]) if r.get("head") else None})
    return rows


# --------------------------------------------------------------------------- metrics

def metrics(rows: list[dict], role: str, threshold: float = UNCERTAIN_P, quality_gate: bool = False) -> dict:
    preds = [(r, *decide(r, role, threshold, quality_gate)) for r in rows]
    ans = [(r, p) for r, p, _ in preds if p in CLASSES]
    n, na = len(rows), len(ans)
    acc = sum(p == r["label"] for r, p in ans) / na if na else 0.0
    tp = sum(p in ATTENTION and r["label"] in ATTENTION for r, p in ans)
    fp = sum(p in ATTENTION and r["label"] not in ATTENTION for r, p in ans)
    fn = sum(p not in ATTENTION and r["label"] in ATTENTION for r, p in ans)
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    neutral_ans = [(r, p) for r, p in ans if r["label"] == "neutral"]
    recalls = {}
    for c in CLASSES:
        rs = [p for r, p in ans if r["label"] == c]
        recalls[c] = sum(p == c for p in rs) / len(rs) if rs else None
    known = [v for v in recalls.values() if v is not None]
    return {"clips": n, "answered": na, "abstain_rate": (n - na) / n if n else 0.0, "accuracy": acc,
            "macro_recall": sum(known) / len(known) if known else 0.0,
            "attention_precision": pr, "attention_recall": rc,
            "attention_f1": 2 * pr * rc / (pr + rc) if pr + rc else 0.0,
            "false_alarm_on_neutral": (sum(p in ATTENTION for _, p in neutral_ans) / len(neutral_ans)
                                       if neutral_ans else None),
            "recall": recalls}


def confusion(rows: list[dict], role: str) -> dict[str, Counter]:
    m: dict[str, Counter] = {c: Counter() for c in CLASSES}
    for r in rows:
        p, _ = decide(r, role)
        m[r["label"]][p if p in CLASSES else "uncertain/abstain"] += 1
    return m


def calibration(rows: list[dict], role: str, bins: int = 10) -> dict:
    """ECE of the top-class probability (the value compared with UNCERTAIN_P live)."""
    pts = []
    for r in rows:
        p_lab, p = decide(r, role, threshold=0.0)
        if p is not None and p_lab in CLASSES:
            pts.append((p, p_lab == r["label"]))
    if not pts:
        return {"ece": None, "table": []}
    ece, table = 0.0, []
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        sel = [(p, ok) for p, ok in pts if lo <= p < hi or (b == bins - 1 and p == 1.0)]
        if not sel:
            continue
        conf = sum(p for p, _ in sel) / len(sel)
        acc = sum(ok for _, ok in sel) / len(sel)
        ece += len(sel) / len(pts) * abs(conf - acc)
        table.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": len(sel), "confidence": conf, "accuracy": acc})
    return {"ece": ece, "table": table}


def sweep(rows: list[dict], role: str) -> list[dict]:
    out = []
    for t in (0.0, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.8, 0.9):
        m = metrics(rows, role, threshold=t)
        out.append({"threshold": t, "coverage": 1 - m["abstain_rate"], "accuracy": m["accuracy"],
                    "attention_recall": m["attention_recall"]})
    return out


def fairness(rows: list[dict], role: str) -> dict[str, list[dict]]:
    out = {}
    for key in ("sex", "age", "race"):
        groups = defaultdict(list)
        for r in rows:
            if r.get(key) not in (None, "", "?", "unknown"):
                groups[r[key]].append(r)
        table = []
        for g, rs in sorted(groups.items()):
            m = metrics(rs, role)
            table.append({"group": g, "clips": len(rs), "speakers": len({r["speaker"] for r in rs}),
                          "accuracy": m["accuracy"], "attention_recall": m["attention_recall"],
                          "false_alarm_on_neutral": m["false_alarm_on_neutral"], "abstain_rate": m["abstain_rate"]})
        if table:
            out[key] = table
    return out


def _auc(y: list[int], s: list[float]) -> float | None:
    if len(set(y)) < 2:
        return None
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def indicator_auc(rows: list[dict], role: str) -> dict:
    """How well the live indicators (model part, no within-call baseline) separate the acted emotions."""
    use = [(r, indicators(adjust_for_role(r.get(SYSTEM["key"]) or r["emotions"], role), None))
           for r in rows if r["emotions"]]
    fear = _auc([int(r["label"] == "fearful") for r, _ in use], [i["fear"] for _, i in use])
    anger = _auc([int(r["label"] == "angry") for r, _ in use], [i["anger"] for _, i in use])
    sub = [(r, i) for r, i in use if r["label"] in ("fearful", "sad", "neutral")]
    distress = _auc([int(r["label"] != "neutral") for r, _ in sub], [i["distress"] for _, i in sub])
    att = _auc([int(r["label"] in ATTENTION) for r, _ in use],
               [max(i["fear"], i["anger"], i["distress"]) for _, i in use])
    return {"fear_vs_rest": fear, "anger_vs_rest": anger, "distress_fear_sad_vs_neutral": distress,
            "attention_any_vs_neutral": att}


# --------------------------------------------------------------------------- report

def pct(v) -> str:
    return "n/a" if v is None else f"{100 * v:.1f}%"


def f2(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def compare(results: dict[str, list[dict]], role: str) -> tuple[list[str], dict]:
    """Raw emotion2vec+ vs the telephone-adapted head on the same clips.

    Compared at EQUAL coverage (every non-silent clip answered, no confidence threshold), so a system cannot
    look better just by abstaining more; the live setting (threshold UNCERTAIN_P) is shown separately.
    """
    out: dict = {}
    keep = SYSTEM["key"]
    for key in ("emotions", "head"):
        SYSTEM["key"] = key
        out[key] = {c: {"all": metrics(rows, role, threshold=0.0), "live": metrics(rows, role),
                        "ece": calibration(rows, role)["ece"]} for c, rows in results.items()}
    SYSTEM["key"] = keep
    L = ["## 0. Raw model vs telephone-adapted head\n",
         "Same held-out clips. Left: both systems answer every non-silent clip (equal coverage). "
         "*Attention recall* = share of fear / anger / sadness clips flagged; *false alarms* = share of neutral "
         "clips flagged; ECE = calibration error (lower is better).\n",
         "| Condition | Accuracy raw → head | Attention recall raw → head | False alarms raw → head | ECE raw → head |",
         "|---|---|---|---|---|"]
    for c in results:
        a, b = out["emotions"][c], out["head"][c]
        L.append(f"| {CONDITION_LABEL.get(c, c)} | {pct(a['all']['accuracy'])} → **{pct(b['all']['accuracy'])}** "
                 f"| {f2(a['all']['attention_recall'])} → **{f2(b['all']['attention_recall'])}** "
                 f"| {pct(a['all']['false_alarm_on_neutral'])} → {pct(b['all']['false_alarm_on_neutral'])} "
                 f"| {f2(a['ece'])} → **{f2(b['ece'])}** |")
    L.append("\nThe raw model's few false alarms on telephone audio come from calling almost everything *neutral*, "
             "which is also why it misses most callers who need attention. With the live abstention threshold "
             f"(p < {UNCERTAIN_P} → *uncertain*), the head answers fewer clips but is right more often when it does "
             "(section 1).\n")
    return L, out


def report(results: dict[str, list[dict]], role: str, model_id: str) -> tuple[str, dict]:
    system = "telephone-adapted head" if SYSTEM["key"] == "head" else "raw model output"
    data: dict = {"model": model_id, "system": system, "role_prior": role, "uncertain_threshold": UNCERTAIN_P,
                  "conditions": {}}
    L = ["# Speech-emotion results\n",
         f"Model `{model_id}` ({system}) · decision rules as in the live call (role prior `{role}`, "
         f"uncertain below p = {UNCERTAIN_P}) · generated by `eval/run_ser_eval.py`.\n"]
    if any(r.get("head") for rows in results.values() for r in rows):
        cl, data["comparison"] = compare(results, role)
        L += cl
        L.append(f"Sections 1-6 below describe the **{system}**, which is what the live call uses.\n")

    L += ["## 1. Robustness: studio audio vs a telephone line\n",
          "| Condition | Clips | Abstained | Accuracy (answered) | Macro recall | Attention P / R / F1 | "
          "False alarms on neutral |", "|---|---|---|---|---|---|---|"]
    for c, rows in results.items():
        m = metrics(rows, role)
        g = metrics(rows, role, quality_gate=True)
        data["conditions"][c] = {"metrics": m, "with_quality_gate": g}
        L.append(f"| {CONDITION_LABEL.get(c, c)} | {m['clips']} | {pct(m['abstain_rate'])} | {pct(m['accuracy'])} "
                 f"| {pct(m['macro_recall'])} | {f2(m['attention_precision'])} / {f2(m['attention_recall'])} / "
                 f"{f2(m['attention_f1'])} | {pct(m['false_alarm_on_neutral'])} |")
    L.append("\nChance level for 4 balanced classes is 25%. *Attention* = fear, anger or sadness vs neutral.\n")
    if any("_snr" in c for c in results):
        L += ["With the distress track's audio-quality gate (windows under "
              f"{SNR_GATE_DB:.0f} dB estimated SNR abstain):\n",
              "| Condition | Abstained | Accuracy (answered) | Attention recall |", "|---|---|---|---|"]
        for c in results:
            g = data["conditions"][c]["with_quality_gate"]
            L.append(f"| {CONDITION_LABEL.get(c, c)} | {pct(g['abstain_rate'])} | {pct(g['accuracy'])} "
                     f"| {f2(g['attention_recall'])} |")
        L.append("")

    first = next(iter(results))
    base = results[first]
    L += [f"## 2. Per class ({CONDITION_LABEL.get(first, first)})\n",
          "| True \\ predicted | neutral | angry | fearful | sad | uncertain / abstain | Recall |",
          "|---|---|---|---|---|---|---|"]
    cm = confusion(base, role)
    rec = metrics(base, role)["recall"]
    for c in CLASSES:
        row = cm[c]
        L.append(f"| **{c}** | " + " | ".join(str(row[k]) for k in (*CLASSES, "uncertain/abstain"))
                 + f" | {pct(rec[c])} |")
    data["confusion"] = {c: dict(v) for c, v in cm.items()}

    L += ["\n## 3. Calibration and the abstention threshold\n"]
    for c in results:
        cal = calibration(results[c], role)
        data["conditions"][c]["ece"] = cal["ece"]
    cal = calibration(base, role)
    data["calibration_table"] = cal["table"]
    L.append("Expected calibration error (top-class probability vs being right, 10 bins): "
             + ", ".join(f"{CONDITION_LABEL.get(c, c)} **{f2(data['conditions'][c]['ece'])}**" for c in results) + "\n")
    L += ["| Confidence bin | Clips | Mean confidence | Actual accuracy |", "|---|---|---|---|"]
    for b in cal["table"]:
        L.append(f"| {b['bin']} | {b['n']} | {pct(b['confidence'])} | {pct(b['accuracy'])} |")
    for c in results:
        data["conditions"][c]["sweep"] = sweep(results[c], role)
    tel = "phone" if "phone" in results else first
    L += [f"\nThreshold sweep ({CONDITION_LABEL.get(tel, tel)}): what the officer would see at each setting.\n",
          "| Abstain below p | Coverage (answered) | Accuracy (answered) | Attention recall |", "|---|---|---|---|"]
    for s in data["conditions"][tel]["sweep"]:
        mark = " ← used live" if abs(s["threshold"] - UNCERTAIN_P) < 1e-9 else ""
        L.append(f"| {s['threshold']:.2f}{mark} | {pct(s['coverage'])} | {pct(s['accuracy'])} "
                 f"| {f2(s['attention_recall'])} |")

    fair = fairness(base, role)
    data["fairness"] = fair
    if fair:
        L += [f"\n## 4. Fairness across speakers ({CONDITION_LABEL.get(first, first)})\n",
              "| Group | Speakers | Clips | Accuracy | Attention recall | False alarms on neutral |",
              "|---|---|---|---|---|---|"]
        for key, table in fair.items():
            for t in table:
                L.append(f"| {key}: {t['group']} | {t['speakers']} | {t['clips']} | {pct(t['accuracy'])} "
                         f"| {f2(t['attention_recall'])} | {pct(t['false_alarm_on_neutral'])} |")
            accs = [t["accuracy"] for t in table if t["clips"] >= 20 and t["speakers"] >= 3]
            if len(accs) >= 2:
                L.append(f"| *{key}: largest gap* | | | *{100 * (max(accs) - min(accs)):.1f} pts* | | |")
        ok = (lambda t: t["clips"] >= 20 and t["speakers"] >= 3)  # noqa: E731
        data["fairness_gap"] = {k: max(t["accuracy"] for t in v if ok(t)) - min(t["accuracy"] for t in v if ok(t))
                                for k, v in fair.items() if sum(ok(t) for t in v) >= 2}
        L.append("\n*Largest gap is computed over groups with at least 3 speakers and 20 clips.*")

    L += ["\n## 5. The live indicators (model part)\n",
          "ROC-AUC of each indicator as a score (0.5 = chance, 1.0 = perfect). The within-call baseline part of "
          "distress and nervousness needs a whole call and is covered by unit tests, not by single clips.\n",
          "| Indicator | " + " | ".join(CONDITION_LABEL.get(c, c) for c in results) + " |",
          "|---|" + "---|" * len(results)]
    aucs = {c: indicator_auc(rows, role) for c, rows in results.items()}
    data["indicator_auc"] = aucs
    for key, name in (("fear_vs_rest", "Fear (fear vs others)"), ("anger_vs_rest", "Anger (anger vs others)"),
                      ("distress_fear_sad_vs_neutral", "Distress (fear or sadness vs neutral)"),
                      ("attention_any_vs_neutral", "Any of the three (attention vs neutral)")):
        L.append(f"| {name} | " + " | ".join(f2(aucs[c][key]) for c in results) + " |")

    ms = [r["ms"] for r in base if r["ms"]]
    if ms:
        ms.sort()
        data["latency_ms"] = {"median": ms[len(ms) // 2], "p95": ms[int(0.95 * (len(ms) - 1))],
                              "mean": sum(ms) / len(ms)}
        L.append(f"\n## 6. Model time\n\nVoice-emotion model per clip (≈ one 3 s live window) on this machine's CPU: "
                 f"median **{data['latency_ms']['median']:.0f} ms**, p95 {data['latency_ms']['p95']:.0f} ms.\n")
    sp = {r["speaker"] for r in base}
    held = all(is_holdout(r["speaker"]) for r in base if r["source"] == "crema-d")
    L.append(f"\n*Data: {len(base)} clips from {len(sp)} speakers, balanced over the four classes"
             + (f"; CREMA-D test speakers are held out from head training ({HOLDOUT})." if held else ".") + "*\n")
    return "\n".join(L), data


# --------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crema-d", help="CREMA-D AudioWAV folder")
    ap.add_argument("--manifest", help="CSV: path,label[,source,lang,speaker,sex,age]")
    ap.add_argument("--limit", type=int, default=0, help="balanced sample size (0 = all)")
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help=f"comma list from {', '.join(CONDITIONS)}")
    ap.add_argument("--role", default="victim", choices=["none", "victim"],
                    help="'victim' = the helpline prior used on victim calls (default); 'none' = raw model output")
    ap.add_argument("--from-cache", action="store_true", help="re-summarise saved predictions, no model needed")
    ap.add_argument("--holdout", action="store_true",
                    help=f"CREMA-D: only test speakers never used to train the head ({HOLDOUT})")
    ap.add_argument("--system", default="auto", choices=["auto", "raw", "head"],
                    help="what sections 1-6 describe: the head when it is available (auto), or the raw model")
    args = ap.parse_args(argv)
    conds = [c.strip() for c in args.conditions.split(",") if c.strip()]
    bad = [c for c in conds if c not in CONDITIONS]
    if bad:
        ap.error(f"unknown condition(s): {bad}")

    results: dict[str, list[dict]] = {}
    model_id = "emotion2vec+"
    if args.from_cache:
        for c in conds:
            rows = load_predictions(c)
            if rows:
                results[c] = rows
        if not results:
            print(f"No saved predictions in {OUT_DIR}")
            return 1
        meta = OUT_DIR / "ser_results.json"
        if meta.exists():
            model_id = json.loads(meta.read_text(encoding="utf-8")).get("model", model_id)
    else:
        if not (args.crema_d or args.manifest):
            ap.error("give --crema-d and/or --manifest (or --from-cache)")
        items = load_items(args)
        if not items:
            print("No labelled clips found.")
            return 1
        from anvesh.speech.models import emotion
        if emotion.get(block=True) is None:
            print(f"Voice emotion model not available: {emotion.status().error}")
            print("Install backend/requirements-ml.txt and run: python -m anvesh.download_models --only emotion")
            return 1
        model_id = Path(emotion.model_id).name if Path(emotion.model_id).exists() else emotion.model_id
        if model_id == "emotion2vec_plus_base":
            model_id = "iic/emotion2vec_plus_base"
        print(f"Evaluating {len(items)} clips x {len(conds)} condition(s) with {model_id} ...", flush=True)
        head = emotion.head()
        if head is not None:
            print(f"Telephone-adapted head: {head.spec.get('name')}", flush=True)
        for c in conds:
            results[c] = run_condition(items, emotion, head, c)
            save_predictions(results[c], c)
    has_head = any(r.get("head") for rows in results.values() for r in rows)
    SYSTEM["key"] = "head" if (args.system == "head" or (args.system == "auto" and has_head)) else "emotions"
    md, data = report(results, args.role, model_id)
    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / "ser_results.md").write_text(md, encoding="utf-8")
    (OUT_DIR / "ser_results.json").write_text(json.dumps(data, indent=1, default=float), encoding="utf-8")
    print(md)
    print(f"Saved to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
