"""Download open speech-emotion test audio for run_ser_eval.py.

  python eval/get_eval_audio.py --crema 200          # CREMA-D (English, acted, ODbL): 200 clips per class
  python eval/get_eval_audio.py --crema-all          # every CREMA-D clip of the four classes (~4,900, ~500 MB),
                                                     # needed by train_ser_head.py and run_ser_eval.py --holdout
  python eval/get_eval_audio.py --rasa 50            # AI4Bharat Rasa Hindi test split (CC-BY-4.0): 50 per class

CREMA-D clips go to eval/audio/crema-d/. Rasa clips go to eval/audio/rasa-hi/ with a manifest.csv.
Rasa is gated: first accept its terms at https://huggingface.co/datasets/ai4bharat/Rasa, then run
`pip install datasets huggingface_hub` and `huggingface-cli login`.
Audio is for evaluation only. It is never committed (eval/audio/ is in .gitignore) and never used for training.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).with_name("audio")
CREMA_REPO = "CheyneyComputerScience/CREMA-D"
CREMA_CLASSES = ("ANG", "FEA", "NEU", "SAD")
# Rasa "style" -> evaluation class. Rasa has no neutral emotion, so the conversational reading style stands in.
RASA_MAP = {"anger": "angry", "angry": "angry", "fear": "fearful", "sad": "sad", "conversation": "neutral"}


def _json(url: str):
    req = urllib.request.Request(url, headers={"User-Agent": "anvesh-eval", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


def crema(per_class: int) -> None:
    dest = OUT / "crema-d"
    dest.mkdir(parents=True, exist_ok=True)
    root = _json(f"https://api.github.com/repos/{CREMA_REPO}/git/trees/master")
    sha = next(t["sha"] for t in root["tree"] if t["path"] == "AudioWAV")
    names = [t["path"] for t in _json(f"https://api.github.com/repos/{CREMA_REPO}/git/trees/{sha}")["tree"]
             if t["path"].endswith(".wav")]
    by = defaultdict(list)
    for n in names:
        p = n.split("_")
        if len(p) >= 3 and p[2] in CREMA_CLASSES:
            by[p[2]].append(n)
    random.seed(7)  # same balanced sample every time
    pick = []
    for k in sorted(by):
        v = by[k]
        random.shuffle(v)
        pick += v[:per_class]

    def get(n: str) -> bool:
        f = dest / n
        if f.exists() and f.read_bytes()[:4] == b"RIFF":
            return True
        url = f"https://media.githubusercontent.com/media/{CREMA_REPO}/master/AudioWAV/{n}"
        for _ in range(3):
            try:
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                if data[:4] == b"RIFF":
                    f.write_bytes(data)
                    return True
            except Exception:
                pass
        return False

    with ThreadPoolExecutor(6) as ex:
        ok = sum(ex.map(get, pick))
    demographics(dest)
    print(f"CREMA-D: {ok}/{len(pick)} clips in {dest} from {len({n[:4] for n in pick})} speakers")


CREMA_SENTENCES = ("IEO", "TIE", "IOM", "IWW", "TAI", "MTI", "IWL", "ITH", "DFA", "ITS", "TSI", "WSI")


def _download(n: str, dest: Path) -> bool:
    f = dest / n
    if f.exists() and f.read_bytes()[:4] == b"RIFF":
        return True
    url = f"https://media.githubusercontent.com/media/{CREMA_REPO}/master/AudioWAV/{n}"
    for _ in range(3):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read()
            if data[:4] == b"RIFF":
                f.write_bytes(data)
                return True
        except urllib.error.HTTPError as e:
            if e.code == 404:  # not every actor / sentence / level combination exists
                return False
        except Exception:
            pass
    return False


def demographics(dest: Path) -> None:
    """Speaker age / sex / race, used for the fairness table in run_ser_eval.py."""
    f = dest / "VideoDemographics.csv"
    if not f.exists():
        url = f"https://raw.githubusercontent.com/{CREMA_REPO}/master/VideoDemographics.csv"
        with urllib.request.urlopen(url, timeout=60) as r:
            f.write_bytes(r.read())


def crema_all() -> None:
    """All four-class clips, built from CREMA-D's file-name scheme (no API listing needed)."""
    dest = OUT / "crema-d"
    dest.mkdir(parents=True, exist_ok=True)
    names = []
    for actor in range(1001, 1092):
        for s in CREMA_SENTENCES:
            for e in CREMA_CLASSES:
                levels = ("LO", "MD", "HI", "XX") if (s == "IEO" and e != "NEU") else ("XX",)
                names += [f"{actor}_{s}_{e}_{lv}.wav" for lv in levels]
    with ThreadPoolExecutor(8) as ex:
        ok = sum(ex.map(lambda n: _download(n, dest), names))
    demographics(dest)
    print(f"CREMA-D: {ok} clips in {dest} (of {len(names)} possible names; missing combinations are normal)")


def rasa(per_class: int) -> None:
    from datasets import Audio, load_dataset
    dest = OUT / "rasa-hi"
    dest.mkdir(parents=True, exist_ok=True)
    ds = load_dataset("ai4bharat/Rasa", "Hindi", split="test", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    count, seen, rows = Counter(), Counter(), []
    quota = per_class // 2 or 1  # half female, half male
    for r in ds:
        style = str(r.get("style", "")).strip().lower()
        seen[style] += 1
        lab = RASA_MAP.get(style)
        g = str(r.get("gender", "?")).lower()
        if not lab or count[(lab, g)] >= quota:
            if all(count[(lab_, g_)] >= quota for lab_ in set(RASA_MAP.values()) for g_ in ("female", "male")):
                break
            continue
        a = r["audio"]
        ext = Path(a.get("path") or "x.wav").suffix or ".wav"
        f = dest / f"{lab}_{g}_{count[(lab, g)]:03d}{ext}"
        f.write_bytes(a["bytes"])
        count[(lab, g)] += 1
        rows.append({"path": f.name, "label": lab, "source": "rasa-hi", "lang": "hi"})
    with open(dest / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "source", "lang"])
        w.writeheader()
        w.writerows(rows)
    print("Rasa styles seen:", dict(seen))
    print(f"Rasa Hindi: {len(rows)} clips in {dest}:", dict(count))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--crema", type=int, default=0, help="CREMA-D clips per class (e.g. 200)")
    ap.add_argument("--crema-all", action="store_true", help="every CREMA-D clip of the four classes")
    ap.add_argument("--rasa", type=int, default=0, help="Rasa Hindi clips per class (e.g. 50)")
    a = ap.parse_args()
    if not (a.crema or a.crema_all or a.rasa):
        ap.error("give --crema N, --crema-all and/or --rasa N")
    if a.crema_all:
        crema_all()
    elif a.crema:
        crema(a.crema)
    if a.rasa:
        rasa(a.rasa)
