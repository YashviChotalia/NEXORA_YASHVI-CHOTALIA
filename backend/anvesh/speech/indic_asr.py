"""Speech-to-text for 22 Indian languages: AI4Bharat IndicConformer (600M, multilingual, MIT licence).

Whisper is strong for English, Hindi and many foreign languages but weak for most other Indian languages. When a
call is in one of the languages below, Anvesh transcribes it with IndicConformer (CTC decoding, word timestamps).

This is a small loader of its own: it reads only the model FILES (ONNX graphs, the TorchScript feature
preprocessor, vocab.json, language_masks.json). It never imports the Python file shipped with the download, so no
downloaded code runs. Weights: `python -m anvesh.download_models --only indic` (gated on Hugging Face: accept the
model's terms first), or eval/get_multilingual_audio.py. Location: ANVESH_INDIC_ASR_DIR.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# IndicConformer language codes -> display names
LANGUAGES = {
    "as": "Assamese", "bn": "Bengali", "brx": "Bodo", "doi": "Dogri", "gu": "Gujarati", "hi": "Hindi",
    "kn": "Kannada", "kok": "Konkani", "ks": "Kashmiri", "mai": "Maithili", "ml": "Malayalam", "mni": "Manipuri",
    "mr": "Marathi", "ne": "Nepali", "or": "Odia", "pa": "Punjabi", "sa": "Sanskrit", "sat": "Santali",
    "sd": "Sindhi", "ta": "Tamil", "te": "Telugu", "ur": "Urdu",
}
BLANK_ID = 256
FRAME_S = 0.08  # one encoder frame
MAX_CHUNK_S = 20.0  # long recordings are cut at pauses into chunks of at most this length
SR = 16000


class IndicConformer:
    def __init__(self, folder: str | Path):
        import onnxruntime as ort
        import torch
        root = Path(folder)
        assets = root / "assets" if (root / "assets").exists() else root
        need = [assets / f for f in ("preprocessor.ts", "encoder.onnx", "ctc_decoder.onnx", "vocab.json",
                                     "language_masks.json")]
        missing = [p.name for p in need if not p.exists()]
        if missing:
            raise FileNotFoundError(f"IndicConformer files missing in {assets}: {', '.join(missing)}")
        self.torch = torch
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prov = ["CPUExecutionProvider"]
        self.pre = torch.jit.load(str(assets / "preprocessor.ts"), map_location="cpu")
        self.encoder = ort.InferenceSession(str(assets / "encoder.onnx"), opts, providers=prov)
        self.ctc = ort.InferenceSession(str(assets / "ctc_decoder.onnx"), opts, providers=prov)
        self.vocab = json.loads((assets / "vocab.json").read_text(encoding="utf-8"))
        self.masks = json.loads((assets / "language_masks.json").read_text(encoding="utf-8"))

    def _logits(self, x: np.ndarray) -> np.ndarray:
        torch = self.torch
        wav = torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))[None, :]
        with torch.no_grad():
            feats, length = self.pre(input_signal=wav, length=torch.tensor([wav.shape[-1]]))
        enc, enc_len = self.encoder.run(["outputs", "encoded_lengths"],
                                        {"audio_signal": feats.numpy(), "length": length.numpy()})
        return self.ctc.run(["logprobs"], {"encoder_output": enc})[0][0][: int(enc_len[0])]

    def identify(self, x: np.ndarray, max_s: float = 15.0) -> list[tuple[str, float]]:
        """Which of the 22 languages is this? Scores the audio against each language's own vocabulary (mean
        best-path log-probability on speech frames) from ONE encoder pass; higher is more likely."""
        lg = self._logits(x[: int(max_s * SR)])
        out = []
        for lang, mask in self.masks.items():
            z = lg[:, mask]
            z = z - z.max(axis=1, keepdims=True)
            z = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
            speech = z.argmax(axis=1) != BLANK_ID
            out.append((lang, float(z.max(axis=1)[speech].mean()) if speech.any() else -99.0))
        return sorted(out, key=lambda kv: -kv[1])

    def _decode(self, x: np.ndarray, lang: str) -> list[tuple[str, float, float]]:
        """One chunk -> [(word, start_s, end_s)] by greedy CTC over this language's vocabulary."""
        lp = self._logits(x)[:, self.masks[lang]]
        path = lp.argmax(axis=-1)
        vocab = self.vocab[lang]
        tokens, cur, start = [], None, 0
        for f, tok in enumerate(path):
            tok = int(tok)
            if tok == BLANK_ID or tok != cur:
                if cur is not None:
                    tokens.append((vocab[cur], start * FRAME_S, f * FRAME_S))
                cur, start = (None, f) if tok == BLANK_ID else (tok, f)
        if cur is not None:
            tokens.append((vocab[cur], start * FRAME_S, len(path) * FRAME_S))
        words, w, ws, we = [], "", 0.0, 0.0
        for t, t0, t1 in tokens:
            if "▁" in t:
                if w:
                    words.append((w, ws, we))
                w, ws = t.replace("▁", ""), t0
            else:
                w += t
            we = t1
        if w:
            words.append((w, ws, we))
        return [(a, round(b, 2), round(c, 2)) for a, b, c in words if a.strip()]

    def transcribe(self, x: np.ndarray, lang: str, offset: float = 0.0) -> list[dict]:
        """Segments in the same shape as the Whisper adapter: start, end, text, words[]."""
        if lang not in self.masks:
            raise ValueError(f"IndicConformer has no language '{lang}'")
        words: list[tuple[str, float, float]] = []
        for c0, chunk in split_at_pauses(x):
            words += [(w, a + c0, b + c0) for w, a, b in self._decode(chunk, lang)]
        return words_to_segments(words, offset)


def split_at_pauses(x: np.ndarray, sr: int = SR, max_s: float = MAX_CHUNK_S) -> list[tuple[float, np.ndarray]]:
    """Cut long audio into <= max_s pieces at the quietest 100 ms between 60% and 100% of the limit."""
    out, i, n, hop = [], 0, len(x), int(0.1 * sr)
    limit = int(max_s * sr)
    while n - i > limit:
        lo, hi = i + int(0.6 * limit), i + limit
        energies = [float(np.mean(x[j:j + hop] ** 2)) for j in range(lo, hi - hop, hop)]
        cut = lo + hop * int(np.argmin(energies)) + hop // 2 if energies else hi
        out.append((i / sr, x[i:cut]))
        i = cut
    if n - i > int(0.2 * sr):
        out.append((i / sr, x[i:]))
    return out


def words_to_segments(words: list[tuple[str, float, float]], offset: float = 0.0, pause_s: float = 0.6,
                      max_s: float = 12.0) -> list[dict]:
    """Group words into transcript lines at pauses, so each line can be played back and cited as evidence."""
    segs: list[dict] = []
    cur: list[tuple[str, float, float]] = []

    def flush() -> None:
        if cur:
            segs.append({"start": round(cur[0][1] + offset, 2), "end": round(cur[-1][2] + offset, 2),
                         "text": " ".join(w for w, _, _ in cur), "avg_logprob": None, "no_speech_prob": None,
                         "words": [{"word": w, "start": round(a + offset, 2), "end": round(b + offset, 2), "p": None}
                                   for w, a, b in cur]})
            cur.clear()

    for w in words:
        if cur and (w[1] - cur[-1][2] > pause_s or w[2] - cur[0][1] > max_s):
            flush()
        cur.append(w)
    flush()
    return segs


def default_dir() -> Path:
    return Path(__file__).resolve().parents[2] / "data" / "hf" / "indic-conformer"
