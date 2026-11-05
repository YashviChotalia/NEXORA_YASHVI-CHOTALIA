"""Download every enabled speech model once, before the first call.

Model weights are not stored in the repository (they are several GB). Run:

    python -m anvesh.download_models            # all enabled models
    python -m anvesh.download_models --only asr emotion

Weights go to the normal Hugging Face / ModelScope caches (ECAPA to <data_dir>/models/ecapa), so the
backend finds them on start-up. Which models are used is set by the ANVESH_* variables in the README.
Exit code is 0 when every requested model loaded, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
import time

from .speech import models as M

ALL = {
    "asr": M.asr,                 # live transcription (faster-whisper, ANVESH_ASR_MODEL)
    "asr_final": M.asr_final,     # accurate re-transcription after a call (ANVESH_ASR_FINAL_MODEL)
    "emotion": M.emotion,         # voice emotion (emotion2vec+, ANVESH_EMOTION_MODEL)
    "speaker": M.speaker,         # speaker embeddings for suspect recordings (SpeechBrain ECAPA)
    "spoof": M.spoof,             # synthetic-voice detector (ANVESH_SPOOF_MODEL)
    "indic": M.indic,             # 22 Indian languages (AI4Bharat IndicConformer; gated: accept terms + log in)
}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Pre-download Anvesh speech models")
    ap.add_argument("--only", nargs="+", choices=sorted(ALL), help="download only these models")
    args = ap.parse_args(argv)
    names = args.only or list(ALL)
    failed = 0
    for name in names:
        m = ALL[name]
        if not m.enabled:
            print(f"[skip] {name:<10} disabled by configuration")
            continue
        print(f"[....] {name:<10} {m.model_id}", flush=True)
        t = time.time()
        m.get(block=True)
        st = m.status()
        if st.loaded:
            print(f"[ ok ] {name:<10} ready in {time.time() - t:.0f}s")
        else:
            failed += 1
            print(f"[FAIL] {name:<10} {st.error}")
    if failed:
        print(f"\n{failed} model(s) not available. The app still runs; those signals are reported as "
              "unavailable and the system abstains. Check requirements-ml.txt and your network.")
    else:
        print("\nAll requested models are ready.")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
