from datetime import datetime

from anvesh import distress, triage
from anvesh.nlp.extract import Field

NOW = datetime(2026, 10, 24, 15, 0)


def _f(**kw):
    return {k: Field(k, v, str(v), 0.8) for k, v in kw.items()}


def test_recent_loss_is_immediate_and_names_missing_fields():
    t = triage.assess("victim", _f(amount_lost=48000, incident_time="2026-10-24T14:38"), NOW, False, False, None)
    assert t.priority == "IMMEDIATE"
    assert "upi_id" in t.missing and "bank" in t.missing
    assert "recover" not in t.next_action.lower()  # never claims recoverability


def test_old_loss_is_not_immediate():
    t = triage.assess("victim", _f(amount_lost=5000, incident_time="2026-10-01T10:00"), NOW, False, False, None)
    assert t.priority == "STANDARD"


def test_crisis_escalates_regardless_of_money_while_live():
    t = triage.assess("victim", {}, NOW, False, True, None, call_age_min=5)
    assert t.priority == "IMMEDIATE" and t.headline == "Human safety first"


def test_emotion_never_changes_the_financial_score():
    a = triage.assess("victim", _f(amount_lost=5000, incident_time="2026-10-01T10:00"), NOW, False, False, None)
    b = triage.assess("victim", _f(amount_lost=5000, incident_time="2026-10-01T10:00"), NOW, False, False, "high")
    assert a.financial_score == b.financial_score and a.emotion_adjustment is None


def test_high_distress_raises_priority_by_one_level_only():
    old = _f(amount_lost=5000, incident_time="2026-10-01T10:00")  # STANDARD on facts alone
    t = triage.assess("victim", old, NOW, False, False, "high", call_age_min=2)
    assert t.priority == "HIGH" and t.score >= 30
    assert t.emotion_adjustment["from"] == "STANDARD" and t.emotion_adjustment["to"] == "HIGH"
    assert any("officer can override" in r for r in t.reasons)
    mid = triage.assess("victim", _f(amount_lost=5000, incident_time="2026-10-20T10:00"), NOW, False, False, "high",
                        call_age_min=2)  # HIGH on facts -> at most IMMEDIATE, never more than one step
    assert mid.emotion_adjustment["from"] == "HIGH" and mid.priority == "IMMEDIATE"


def test_elevated_distress_is_only_a_tiebreak_inside_a_level():
    facts = _f(amount_lost=5000, incident_time="2026-10-01T10:00")
    a = triage.assess("victim", facts, NOW, False, False, None, call_age_min=2)
    b = triage.assess("victim", facts, NOW, False, False, "elevated", call_age_min=2)
    assert a.priority == b.priority and b.score > a.score and b.score < 30


def test_distress_never_reorders_suspect_recordings_or_old_calls():
    s = triage.assess("suspect_recording", {}, NOW, False, False, "high", call_age_min=2)
    assert s.emotion_adjustment is None and s.priority == "STANDARD"
    old = triage.assess("victim", _f(amount_lost=5000, incident_time="2026-10-01T10:00"), NOW, False, False, "high",
                        call_age_min=24 * 60)  # a day after the call the caller's voice then says little about now
    assert old.emotion_adjustment is None and old.priority == "STANDARD"


def test_immediate_cases_are_not_raised_further_by_emotion():
    t = triage.assess("victim", _f(amount_lost=48000, incident_time="2026-10-24T14:38"), NOW, False, False, "high",
                      call_age_min=2)
    assert t.priority == "IMMEDIATE" and t.score == t.financial_score + (t.emotion_adjustment or {}).get("points", 0)


def test_suspect_mode_is_a_lead_not_a_finding():
    t = triage.assess("suspect_recording", {}, NOW, False, False, None)
    assert "lead" in t.headline.lower()


def test_distress_disabled_for_suspects():
    r = distress.assess("suspect", [distress.Window(0, 3, {"fearful": 0.9}, None)], [], [])
    assert r.enabled is False and r.level == "off" and r.score is None


def test_distress_abstains_without_audio_but_respects_crisis_words():
    r = distress.assess("victim", [], [], [{"phrase": "sab khatam"}])
    assert r.score is None and r.level == "high" and r.abstain_reasons


def test_distress_abstains_on_unvalidated_language():
    w = [distress.Window(i * 3, i * 3 + 3, {"fearful": 0.8}, {"pitch_std": 10, "rms": 0.1, "speech_rate": 3,
                                                           "pause_ratio": 0.2, "snr_db": 30}) for i in range(5)]
    r = distress.assess("victim", w, [], [], language="xh")  # never measured
    assert r.score is None and any("not validated" in a for a in r.abstain_reasons)


def test_measured_but_unvalidated_language_is_shown_but_never_validated():
    w = [distress.Window(i * 3, i * 3 + 3, {"fearful": 0.8}, {"pitch_std": 10, "rms": 0.1, "speech_rate": 3,
                                                           "pause_ratio": 0.2, "snr_db": 30}) for i in range(5)]
    r = distress.assess("victim", w, [], [], language="ta", measured_languages={"ta", "hi"})
    assert r.score is not None and not r.validated and r.confidence == "low"
    assert any("not validated for 'ta'" in x for x in r.reasons)
    assert distress.assess("victim", w, [], [], language="bn").validated


def test_distress_uses_sustained_level_not_single_spike():
    calm = {"neutral": 0.9, "fearful": 0.05}
    w = [distress.Window(i * 3, i * 3 + 3, calm, None) for i in range(11)]
    w.append(distress.Window(33, 36, {"fearful": 1.0}, None))
    r = distress.assess("victim", w, [], [], language="hi")
    assert r.level in ("low", "elevated") and r.level != "high"


def test_silence_windows_are_not_scored():
    import numpy as np
    from anvesh import pipeline
    x = np.zeros(16000 * 4, dtype=np.float32)  # silence
    w = pipeline.compute_windows(x, 0.0, [{"start": 0.0, "end": 3.0}])
    assert w and w[0]["silent"] and w[0]["emotions"] is None
    r = distress.assess("victim", [distress.Window(0, 3, {"disgusted": 0.9}, None, True)], [], [])
    assert r.timeline == []


def test_sounding_like_your_usual_self_is_not_distress():
    usual = {"pitch_std": 20.0, "rms": 0.05, "speech_rate": 3.0, "pause_ratio": 0.3, "snr_db": 30}
    calm = {"neutral": 0.8, "sad": 0.1, "fearful": 0.1}
    w = [distress.Window(i * 3, i * 3 + 3, calm, dict(usual)) for i in range(10)]
    r = distress.assess("victim", w, [], [], language="hi")
    assert r.level == "low", r.score  # an unchanged voice must not add distress on its own
    loud = [distress.Window(i * 3, i * 3 + 3, {"fearful": 0.7, "neutral": 0.3},
                            {**usual, "rms": 0.05 + 0.02 * i, "pitch_std": 20 + 6 * i}) for i in range(10)]
    assert distress.assess("victim", loud, [], [], language="hi").score > r.score


def test_a_few_seconds_of_speech_can_never_be_high():
    w = [distress.Window(i * 3, i * 3 + 3, {"fearful": 0.95, "neutral": 0.05}, None) for i in range(2)]
    r = distress.assess("victim", w, [], [], language="hi")
    assert r.level == "elevated" and any("more speech" in x for x in r.reasons)
