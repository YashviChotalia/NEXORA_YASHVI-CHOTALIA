"""Distress track: role-gated, uncertainty-aware fusion of voice and words.

Output is an *indicative assistance signal* for human prioritisation. It is
never computed for suspects and never used as evidence of honesty or guilt.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from statistics import mean, pstdev

DISTRESS_EMOTIONS = {"fearful": 1.0, "fear": 1.0, "sad": 0.85, "angry": 0.6, "anger": 0.6, "disgusted": 0.4}
ALLOWED_ROLES = {"victim", "informant"}
# Voice levels. Chosen on validation speakers with eval/run_call_eval.py (whole simulated phone calls), never on
# the test speakers: see EVALUATION.md, "Call-level distress".
VOICE_ELEVATED = 0.40
VOICE_HIGH = 0.55
MIN_WINDOWS_HIGH = 4  # a "high" voice reading (the only one that can change priority) needs this many windows
ACOUSTIC_RISE_ONLY = True  # only a RISE in arousal over the caller's own baseline adds distress


@dataclass
class Window:
    t_start: float
    t_end: float
    emotions: dict[str, float] | None = None  # label -> probability (from the emotion model)
    acoustic: dict[str, float] | None = None  # pitch_std, rms, speech_rate, pause_ratio, snr_db
    silent: bool = False


@dataclass
class DistressResult:
    enabled: bool
    level: str  # off | abstain | low | elevated | high
    score: float | None
    confidence: str  # none | low | medium
    timeline: list[dict] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    abstain_reasons: list[str] = field(default_factory=list)
    label: str = "Acoustic and lexical observation, indicative only, not evidence."
    voice_score: float | None = None
    recent: float | None = None
    trend: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def _emo_score(emotions: dict[str, float] | None) -> float | None:
    if not emotions:
        return None
    return min(sum(p * DISTRESS_EMOTIONS.get(k.lower(), 0.0) for k, p in emotions.items()), 1.0)


def _acoustic_scores(windows: list[Window]) -> list[float | None]:
    """Change against the speaker's own baseline (first 20% of windows)."""
    feats = [w.acoustic for w in windows]
    usable = [f for f in feats if f]
    if len(usable) < 3:
        return [None] * len(windows)
    k = max(2, len(usable) // 5)
    base = usable[:k]

    def stat(key):
        vals = [f.get(key, 0.0) for f in base]
        return mean(vals), (pstdev(vals) or 1e-6)

    stats = {key: stat(key) for key in ("pitch_std", "rms", "speech_rate", "pause_ratio")}
    out: list[float | None] = []
    for f in feats:
        if not f:
            out.append(None)
            continue
        z = 0.0
        for key, wgt in (("pitch_std", 0.35), ("rms", 0.25), ("speech_rate", 0.2), ("pause_ratio", 0.2)):
            mu, sd = stats[key]
            z += wgt * max(min((f.get(key, mu) - mu) / sd, 3.0), -3.0)
        out.append(max(min(0.5 + z / 6.0, 1.0), 0.0))
    return out


def assess(role: str, windows: list[Window], lexical_hits: list[dict], crisis_hits: list[dict],
           language: str | None = None, validated_languages: set[str] | None = None,
           coercion: bool = False, loss_reported: bool = False) -> DistressResult:
    """Fuse voice (emotion + change against the caller's own baseline), words and situation.

    Voice is weighted most; words and situation can raise the level (a caller who says "please help,
    I am scared" while under active coercion is distressed even if they sound flat), never lower it.
    """
    from .live_ser import adjust_for_role
    if role not in ALLOWED_ROLES:
        return DistressResult(False, "off", None, "none",
                              reasons=[f"Distress analysis is disabled for the '{role}' role by policy."])

    abstain: list[str] = []
    validated_languages = validated_languages or {"en", "hi"}
    if language and language not in validated_languages:
        abstain.append(f"Language '{language}' is not validated for voice distress; showing word cues only.")
    if not windows:
        abstain.append("No usable audio windows.")
    low_snr = [w for w in windows if w.acoustic and w.acoustic.get("snr_db", 99) < 8]
    if windows and len(low_snr) > len(windows) / 2:
        abstain.append("Audio quality too low for reliable voice analysis.")

    windows = [w for w in windows if not w.silent]  # silence carries no emotion
    acoustic = _acoustic_scores(windows)
    timeline, scores = [], []
    model_used = False
    for w, a in zip(windows, acoustic):
        emo = adjust_for_role(w.emotions, role)  # a complainant is never read as 'happy'
        e = _emo_score(emo)
        if e is not None:
            model_used = True
        if a is not None and ACOUSTIC_RISE_ONLY:
            a = max(0.0, a - 0.5) * 2  # 0.5 = the caller's usual voice: sounding like yourself is not distress
        s = (0.55 * e + 0.45 * a) if (e is not None and a is not None) else (e if e is not None else a)
        if s is None:
            continue
        scores.append(s)
        top = max(emo.items(), key=lambda kv: kv[1]) if emo else None
        if top and top[1] < 0.45:
            top = ("uncertain", top[1])  # no emotion clearly dominant
        timeline.append({"t": round((w.t_start + w.t_end) / 2, 2), "score": round(s, 3),
                         "emotion": round(e, 3) if e is not None else None,
                         "acoustic": round(a, 3) if a is not None else None,
                         "label": top[0] if top else None, "p": round(top[1], 3) if top else None})

    reasons = []
    words = min(0.12 * len(lexical_hits), 0.36)
    context = (0.15 if coercion else 0.0) + (0.08 if loss_reported else 0.0)
    if lexical_hits:
        reasons.append(f"{len(lexical_hits)} distress phrase(s) in the caller's words")
    if coercion:
        reasons.append("Caller may still be under the scammer's control")
    if loss_reported:
        reasons.append("Caller reports losing money")
    if crisis_hits:
        reasons.append("Crisis language detected: hand over to a trained person")
    floor = 0.5 if (coercion or len(lexical_hits) >= 2) else 0.0

    voice_ok = bool(scores) and not abstain
    if not voice_ok:  # words and situation only
        score = min(1.0, max(floor, 0.25 + words + context)) if (lexical_hits or coercion or loss_reported) else None
        level = ("high" if crisis_hits else "elevated" if score is not None and score >= 0.45 else
                 "low" if score is not None else "abstain")
        return DistressResult(True, level, None if score is None else round(score, 3),
                              "low" if (lexical_hits or crisis_hits or coercion) else "none",
                              timeline, reasons, abstain or ["Voice signal unavailable: level from words and situation."])

    k = max(2, len(scores) // 4) if len(scores) >= 4 else 1  # "sustained": never a single window once there are 4+
    sustained = mean(sorted(scores)[-k:])  # sustained top quartile, not one spike
    recent = mean(scores[-3:])              # how the caller sounds now
    voice = 0.7 * sustained + 0.3 * recent
    score = min(1.0, max(floor, voice + words + context + (0.25 if crisis_hits else 0.0)))
    level = "high" if crisis_hits or score >= VOICE_HIGH else "elevated" if score >= VOICE_ELEVATED else "low"
    if level == "high" and not crisis_hits and len(scores) < MIN_WINDOWS_HIGH:
        level = "elevated"  # "sustained" needs time: a few seconds of speech can never raise a call's priority
        reasons.append(f"Only {len(scores)} voice window(s) so far: shown as elevated until there is more speech")
    reasons.insert(0, "Voice emotion model + pitch/energy/pace change vs the caller's own baseline" if model_used
                   else "Pitch/energy/pace change vs the caller's own baseline (emotion model not loaded)")
    trend = None
    if len(scores) >= 6:
        early, late = mean(scores[: len(scores) // 2]), mean(scores[-3:])
        trend = "rising" if late - early > 0.12 else "easing" if early - late > 0.12 else "steady"
        reasons.append(f"Voice distress is {trend} through the call")
    conf = "medium" if model_used and len(scores) >= 4 else "low"
    r = DistressResult(True, level, round(score, 3), conf, timeline, reasons, [])
    r.voice_score, r.recent, r.trend = round(voice, 3), round(recent, 3), trend
    return r
