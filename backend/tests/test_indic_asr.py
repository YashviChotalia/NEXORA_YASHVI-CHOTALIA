import numpy as np

from anvesh.speech import indic_asr
from anvesh.speech import models as M


def test_words_become_lines_at_pauses():
    words = [("नमस्ते", 0.0, 0.5), ("जी", 0.6, 0.8), ("मेरे", 2.0, 2.3), ("पैसे", 2.4, 2.8)]
    segs = indic_asr.words_to_segments(words, offset=10.0)
    assert [s["text"] for s in segs] == ["नमस्ते जी", "मेरे पैसे"]
    assert segs[1]["start"] == 12.0 and segs[1]["words"][0]["word"] == "मेरे"


def test_long_audio_is_cut_at_the_quietest_moment():
    sr = 16000
    x = np.random.default_rng(0).standard_normal(45 * sr).astype(np.float32) * 0.1
    x[int(17 * sr):int(17.3 * sr)] = 0  # a pause near the 20 s limit
    chunks = indic_asr.split_at_pauses(x)
    assert all(len(c) <= 20 * sr for _, c in chunks)
    assert abs(chunks[1][0] - 17.15) < 0.2  # first cut lands in the pause
    assert sum(len(c) for _, c in chunks) == len(x)


class _FakeIndic:
    def identify(self, x):
        return [("bn", -0.1), ("as", -0.2)]

    def transcribe(self, x, lang, offset=0.0):
        return [{"start": offset, "end": offset + 1, "text": "எனக்கு உதவி", "words": []}]


def test_tamil_goes_to_indicconformer_and_hindi_stays_on_whisper(monkeypatch):
    monkeypatch.setattr(M.indic, "get", lambda block=False: _FakeIndic())
    assert M.ASR.use_indic("ta") and M.ASR.use_indic("bn")
    assert not M.ASR.use_indic("hi") and not M.ASR.use_indic("en") and not M.ASR.use_indic(None)
    asr = M.ASR("small", True)
    monkeypatch.setattr(asr, "get", lambda block=True: object())
    segs, lang = asr.transcribe(np.zeros(16000, np.float32), offset=3.0, language="ta")
    assert lang == "ta" and segs[0]["text_native"] == "எனக்கு உதவி" and segs[0]["text"] == "எனக்கு உதவி"
    assert asr.last_stats["engine"] == "indicconformer"


def test_no_indic_model_falls_back_to_whisper_routing(monkeypatch):
    monkeypatch.setattr(M.indic, "get", lambda block=False: None)
    assert not M.ASR.use_indic("ta")


def test_urdu_detection_is_read_as_hindi_for_hinglish_callers(monkeypatch):
    class W:
        def detect_language(self, audio):
            return "ur", 0.6, [("ur", 0.6), ("hi", 0.3), ("en", 0.1)]
    from dataclasses import replace
    monkeypatch.setattr(M, "settings", replace(M.settings, asr_language="auto"))
    assert M.ASR("small", True)._language(W(), np.zeros(16000)) == "hi"


def test_indicconformer_decides_which_indian_language(monkeypatch):
    from dataclasses import replace
    monkeypatch.setattr(M.indic, "get", lambda block=False: _FakeIndic())

    class W:
        def detect_language(self, audio):
            return "ne", 0.5, [("ne", 0.5), ("bn", 0.3), ("hi", 0.2)]
    monkeypatch.setattr(M, "settings", replace(M.settings, asr_language="auto"))
    asr = M.ASR("small", True)
    monkeypatch.setattr(asr, "get", lambda block=True: W())
    _, lang = asr.transcribe(np.zeros(16000, np.float32))
    assert lang == "bn"  # Whisper said Nepali; the Indian-language model's own check says Bengali


def test_routing_rule_for_each_kind_of_whisper_verdict():
    from anvesh.speech.indic_asr import route_language
    gu = [("gu", -0.05), ("hi", -0.20), ("pa", -0.25)]
    assert route_language("en", gu) == "en"                         # English stays English
    assert route_language("hi", gu) == "hi"                         # Hindi kept while Hindi is in the top 3
    assert route_language("hi", [("gu", -0.05), ("pa", -0.2), ("mr", -0.3)]) == "gu"  # Gujarati heard as Hindi
    assert route_language("bn", [("as", -0.04), ("bn", -0.09)]) == "as"  # Assamese heard as Bengali
    assert route_language("fr", [("brx", -0.06), ("as", -0.3)]) == "brx"  # Bodo heard as French: confident
    assert route_language("fr", [("mai", -0.45), ("hi", -0.5)]) == "fr"   # a real French caller stays French
    assert route_language("ta", None) == "ta"
