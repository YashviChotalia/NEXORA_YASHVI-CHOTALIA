"""Download the multilingual speech models and emotion data (run on the machine that can reach Hugging Face).

  python eval/get_multilingual_audio.py                 # log in if needed, then everything below
  python eval/get_multilingual_audio.py --only rasa     # one part: asr | rasa | cameo

What it fetches (each source is gated: accept its terms on huggingface.co first):
  asr    ai4bharat/indic-conformer-600m-multilingual  -> data/hf/indic-conformer   (22 Indian languages, MIT)
  rasa   ai4bharat/Rasa (Assamese, Bengali, Tamil; CC-BY-4.0) -> eval/audio/multilingual/rasa/
  cameo  amu-cai/CAMEO (13 corpora, 8 languages; licences per corpus) -> eval/audio/multilingual/cameo/

Datasets are STREAMED and only a balanced sample is kept: at most --per clips for each (language, emotion), for the
four classes Anvesh uses (neutral, angry, fearful, sad). Each folder gets a manifest.csv for run_ser_eval.py.
Audio stays out of git (eval/audio/ and data/ are in .gitignore) and is used for research evaluation only.
The Hugging Face token is never written to this repository; huggingface_hub keeps it in your user profile.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "eval" / "audio" / "multilingual"
CLASSES = {"neutral": "neutral", "neu": "neutral", "calm": "neutral", "conversation": "neutral", "normal": "neutral",
           "angry": "angry", "anger": "angry", "ang": "angry",
           "fear": "fearful", "fearful": "fearful", "fea": "fearful", "scared": "fearful",
           "sad": "sad", "sadness": "sad",
           # Rasa: plain read and conversational speech is the calm (neutral) reference
           "book": "neutral", "news": "neutral", "wiki": "neutral", "read": "neutral", "reading": "neutral",
           "narration": "neutral", "default": "neutral", "plain": "neutral", "conversational": "neutral",
           "neutral_conversation": "neutral", "conv": "neutral"}
EMOTION_COLS = ("emotion", "style", "label", "emotion_label", "category")
LANG_COLS = ("language", "lang", "locale")
SPEAKER_COLS = ("speaker", "speaker_id", "spk", "actor", "actor_id")


def login() -> None:
    from huggingface_hub import HfApi, login as hf_login
    try:
        who = HfApi().whoami()
        print(f"Hugging Face: signed in as {who.get('name')}")
    except Exception:
        print("Hugging Face: paste your READ token (input is hidden), then press Enter.")
        hf_login(add_to_git_credential=False)


def asr() -> None:
    from huggingface_hub import snapshot_download
    dest = ROOT / "data" / "hf" / "indic-conformer"
    if (dest / "config.json").exists() or any(dest.glob("*.onnx")) or len(list(dest.rglob("*"))) > 50:
        print(f"IndicConformer already in {dest}")
        return
    print("Downloading IndicConformer (about 2-3 GB) ...", flush=True)
    snapshot_download("ai4bharat/indic-conformer-600m-multilingual", local_dir=str(dest))
    print(f"IndicConformer ready in {dest}")


def _pick(row: dict, names: tuple[str, ...]) -> str | None:
    for n in names:
        if n in row and row[n] not in (None, ""):
            return str(row[n])
    return None


def _save(name: str, per: int, configs: list[str] | None, lang_from_config: bool) -> None:
    from datasets import Audio, get_dataset_config_names, load_dataset
    dest = OUT / name.split("/")[-1].lower()
    dest.mkdir(parents=True, exist_ok=True)
    man = dest / "manifest.csv"
    if man.exists() and sum(1 for _ in open(man, encoding="utf-8")) > 1:
        print(f"{name}: already downloaded ({man}); delete that folder to fetch again")
        return
    try:
        configs = configs or get_dataset_config_names(name)
    except Exception:
        configs = [None]
    print(f"{name}: configs {configs}", flush=True)
    count: Counter = Counter()
    rows_out, seen_labels, schema_shown = [], Counter(), False
    for cfg in configs:
        try:
            ds = load_dataset(name, cfg, streaming=True)
        except Exception as e:
            print(f"  skip config {cfg}: {type(e).__name__}: {str(e)[:200]}")
            continue
        for split, stream in ds.items():
            try:
                stream = stream.cast_column("audio", Audio(decode=False))
            except Exception:
                pass
            n_split = 0
            for r in stream:
                if not schema_shown:
                    print("  columns:", {k: type(v).__name__ for k, v in r.items()})
                    schema_shown = True
                raw = _pick(r, EMOTION_COLS)
                seen_labels[(raw or "?").lower()] += 1
                lab = CLASSES.get((raw or "").strip().lower())
                lang = (cfg if lang_from_config else None) or _pick(r, LANG_COLS) or (cfg or "?")
                lang = str(lang).lower()
                if not lab or count[(lang, lab)] >= per:
                    n_split += 1
                    if n_split > 200000:
                        break
                    continue
                a = r.get("audio") or {}
                data = a.get("bytes") if isinstance(a, dict) else None
                if not data and isinstance(a, dict) and a.get("path") and Path(a["path"]).exists():
                    data = Path(a["path"]).read_bytes()
                if not data:
                    continue
                ext = Path((a.get("path") if isinstance(a, dict) else None) or "x.wav").suffix or ".wav"
                fn = f"{lang}_{lab}_{count[(lang, lab)]:04d}{ext}"
                small = _to_16k(data)
                if not small:
                    continue
                fn = fn.rsplit(".", 1)[0] + ".wav"
                (dest / fn).write_bytes(small)
                count[(lang, lab)] += 1
                spk = _pick(r, SPEAKER_COLS) or "?"
                rows_out.append({"path": fn, "label": lab, "source": name.split("/")[-1].lower(), "lang": lang,
                                 "speaker": f"{lang}-{spk}"})
                n_split += 1
            print(f"  {cfg or '-'} / {split}: kept so far {sum(count.values())}", flush=True)
    with open(dest / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "source", "lang", "speaker"])
        w.writeheader()
        w.writerows(rows_out)
    (dest / "labels_seen.json").write_text(json.dumps(dict(seen_labels.most_common(50)), indent=1), encoding="utf-8")
    print(f"{name}: {len(rows_out)} clips in {dest}")
    for (lang, lab), n in sorted(count.items()):
        print(f"  {lang:>12} {lab:<8} {n}")


class _Tee:
    """Mirror everything printed to logs/multilingual.log (so progress can be checked from another session)."""

    def __init__(self, stream, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.stream, self.file = stream, open(path, "a", encoding="utf-8")

    def write(self, s: str) -> int:
        self.stream.write(s)
        self.file.write(s)
        self.file.flush()
        return len(s)

    def flush(self) -> None:
        self.stream.flush()
        self.file.flush()


def _to_16k(data: bytes) -> bytes | None:
    """Any audio bytes -> 16 kHz mono 16-bit WAV (what the models use; about 1/6 of the size of 48 kHz)."""
    sys.path.insert(0, str(ROOT))
    from anvesh.speech import audio as A
    try:
        x = A.decode(data)
    except Exception:
        return None
    return A.to_wav_bytes(x) if len(x) >= A.SR // 2 else None


def rasa(per: int, max_shards: int) -> None:
    """A small balanced sample of every Rasa language: up to `per` clips per emotion, reading a few parquet
    parcels spread across each language (streamed, nothing large is stored) and stopping once all 4 are full."""
    from datasets import Audio, load_dataset
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    dest = OUT / "rasa"
    dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):
        old.unlink()
    langs = sorted(p.split("/")[-1] for p in fs.ls("datasets/ai4bharat/Rasa", detail=False)
                   if fs.isdir(p) and not p.split("/")[-1].startswith("."))
    print(f"Rasa languages: {len(langs)} {langs}", flush=True)
    rows_out, styles = [], Counter()
    for lang in langs:
        shards = sorted(fs.glob(f"datasets/ai4bharat/Rasa/{lang}/*.parquet"))
        if not shards:
            continue
        order, seen = [], set()
        for k in (0, len(shards) // 2, len(shards) - 1, len(shards) // 4, 3 * len(shards) // 4, 1, len(shards) // 3):
            if 0 <= k < len(shards) and k not in seen:
                order.append(shards[k])
                seen.add(k)
        count: Counter = Counter()
        for shard in order[:max_shards]:
            if all(count[c] >= per for c in ("neutral", "angry", "fearful", "sad")):
                break
            try:
                stream = load_dataset("parquet", data_files=f"hf://{shard}", split="train", streaming=True)
                try:
                    stream = stream.cast_column("audio", Audio(decode=False))  # raw bytes; no torchcodec needed
                except Exception:
                    pass
            except Exception as e:
                print(f"  {lang}: could not read {shard.split('/')[-1]}: {str(e)[:150]}")
                continue
            for r in stream:
                st = str(r.get("style") or "").strip().lower()
                styles[st] += 1
                lab = CLASSES.get(st)
                if not lab or count[lab] >= per:
                    if all(count[c] >= per for c in ("neutral", "angry", "fearful", "sad")):
                        break
                    continue
                a = r.get("audio") or {}
                data = _to_16k(a.get("bytes")) if isinstance(a, dict) and a.get("bytes") else None
                if not data:
                    continue
                code = lang.lower()
                fn = f"{code}_{lab}_{count[lab]:03d}.wav"
                (dest / fn).write_bytes(data)
                count[lab] += 1
                rows_out.append({"path": fn, "label": lab, "source": "rasa", "lang": code,
                                 "speaker": f"{code}-{r.get('gender') or '?'}", "style": st,
                                 "text": str(r.get("text") or "").replace("\n", " ").strip()})
        print(f"  {lang:<10} " + "  ".join(f"{c} {count[c]}" for c in ("neutral", "angry", "fearful", "sad")), flush=True)
        _write_manifest(dest, rows_out)
        (dest / "labels_seen.json").write_text(json.dumps(dict(styles.most_common(50)), indent=1), encoding="utf-8")
        print("    styles seen so far: " + ", ".join(f"{k} {v}" for k, v in styles.most_common(12)), flush=True)
    (dest / "labels_seen.json").write_text(json.dumps(dict(styles.most_common(50)), indent=1), encoding="utf-8")
    print(f"Rasa: {len(rows_out)} clips from {len({r['lang'] for r in rows_out})} languages in {dest}")


def _write_manifest(dest: Path, rows: list[dict]) -> None:
    with open(dest / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["path", "label", "source", "lang", "speaker", "style", "text"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    log = ROOT.parent / "logs" / "multilingual.log"
    log.write_text("", encoding="utf-8")
    sys.stdout = _Tee(sys.stdout, log)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["asr", "rasa", "cameo"], action="append")
    ap.add_argument("--per", type=int, default=20, help="max clips per (language, emotion)")
    ap.add_argument("--max-shards", type=int, default=3, help="Rasa: parquet parcels read per language at most")
    a = ap.parse_args(argv)
    parts = a.only or ["asr", "rasa", "cameo"]
    login()
    failed = []
    for part in parts:
        try:
            if part == "asr":
                asr()
            elif part == "rasa":
                rasa(a.per, a.max_shards)
            elif part == "cameo":
                _save("amu-cai/CAMEO", a.per, None, lang_from_config=False)
        except Exception as e:
            failed.append(part)
            print(f"[FAIL] {part}: {type(e).__name__}: {str(e)[:400]}")
            if "gated" in str(e).lower() or "403" in str(e) or "401" in str(e):
                print("       -> open the page on huggingface.co while signed in and accept its terms, then rerun.")
    print("\nDone." if not failed else f"\nFinished with failures: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
