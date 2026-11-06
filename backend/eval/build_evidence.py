"""Collect the measured results into one small file the app shows on its Evidence page.

  python eval/build_evidence.py      # after run_ser_eval.py / run_call_eval.py / run_lid_eval.py

Reads eval/results/*.json|csv (written by those scripts) and writes anvesh/data/evidence.json. Nothing is typed in by
hand: every number on the page comes from a results file in the repository, which is named next to it.
"""

from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE / "results"
OUT = HERE.parent / "anvesh" / "data" / "evidence.json"
COND = {"clean": "Studio (16 kHz)", "phone": "Phone line (8 kHz, G.711)", "phone_snr20": "Phone + noise 20 dB",
        "phone_snr10": "Phone + noise 10 dB", "phone_snr5": "Phone + noise 5 dB"}


def _r(v, n=3):
    return None if v is None else round(float(v), n)


def main() -> int:
    ev: dict = {"built": date.today().isoformat(), "sources": {}}
    f = R / "ser_results.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        rows = []
        for c, label in COND.items():
            x = d["conditions"].get(c)
            if not x:
                continue
            eq = next((s for s in x.get("sweep", []) if s["threshold"] == 0.0), {})
            raw = (d.get("comparison", {}).get("emotions", {}).get(c, {}).get("all") or {})
            m = x["metrics"]
            rows.append({"condition": label, "accuracy_all": _r(eq.get("accuracy")), "accuracy_answered": _r(m["accuracy"]),
                         "abstain": _r(m["abstain_rate"]), "attention_recall": _r(m["attention_recall"]),
                         "false_alarms": _r(m["false_alarm_on_neutral"]), "ece": _r(x.get("ece")),
                         "raw_accuracy": _r(raw.get("accuracy")), "recall": {k: _r(v) for k, v in m["recall"].items()}})
        ev["ser"] = {"model": d.get("model"), "system": d.get("system"), "clips": d["conditions"]["clean"]["metrics"]["clips"],
                     "rows": rows, "fairness_gap": {k: _r(v) for k, v in d.get("fairness_gap", {}).items()},
                     "latency_ms": {k: _r(v, 0) for k, v in d.get("latency_ms", {}).items()}}
        ev["sources"]["ser"] = "eval/results/ser_results.json (eval/run_ser_eval.py --holdout --limit 800)"
    f = R / "call_results.json"
    if f.exists():
        d = json.loads(f.read_text(encoding="utf-8"))
        t, raw = d.get("test/head", {}), d.get("test/raw", {})
        ev["calls"] = {"calls": t.get("calls"), "distressed_flagged": _r(t.get("distressed_flagged")),
                       "calm_flagged": _r(t.get("calm_flagged")), "calm_high": _r(t.get("calm_high")),
                       "distressed_high": _r(t.get("distressed_high")), "auc": _r(t.get("auc")),
                       "raw_distressed_flagged": _r(raw.get("distressed_flagged")), "raw_auc": _r(raw.get("auc"))}
        ev["sources"]["calls"] = "eval/results/call_results.json (eval/run_call_eval.py)"
    f = R / "multilingual" / "lid_predictions.csv"
    if f.exists():
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        acc = lambda k: sum(r[k] == r["true"] for r in rows) / len(rows)  # noqa: E731
        ev["languages"] = {"clips": len(rows), "languages": len({r["true"] for r in rows}),
                           "anvesh": _r(acc("anvesh")), "whisper": _r(acc("whisper")), "indic": _r(acc("indic"))}
        ev["sources"]["languages"] = "eval/results/multilingual/lid_predictions.csv (eval/run_lid_eval.py)"
    f = R / "ser_head_v2.md"
    if f.exists():
        ev["sources"]["head_v2"] = "eval/results/ser_head_v2.md (eval/train_ser_head_v2.py fit)"
    OUT.write_text(json.dumps(ev, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
