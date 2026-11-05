# Evaluation

This page explains **what we tested, how we tested it, what we found, and where the results stop applying**.
Every number here comes from a script in this repository that anyone can re-run. We do not report numbers we have
not measured.

## Summary

| What is tested | Data | How | Result |
|---|---|---|---|
| Software correctness | 105 unit, API and security tests | `pytest -q` (runs in CI on every push) | **105 / 105 pass** |
| Hinglish field extraction | 20 scripted phrases, 38 field checks | `python eval/run_eval.py` (runs in CI) | **38 / 38 (100%)** on the development set |
| Speech emotion: 4 classes (neutral, anger, fear, sadness) | CREMA-D (acted English speech) | `python eval/run_ser_eval.py --crema-d …` | see [§3](#3-speech-emotion-recognition) |
| Speech emotion on call-like audio | Own role-play recordings and public YouTube clips (Hindi / Hinglish / English) | `python eval/run_ser_eval.py --manifest …` | see [§3](#3-speech-emotion-recognition) |
| Safety behaviour (role gating, abstention, coercion alerts) | Unit tests with controlled inputs | `pytest -q` | **pass** |

---

## 1. Software tests

`pytest -q` runs 105 tests with no speech models installed. They cover:

- **Speech-emotion logic:** smoothing, the caller-baseline arousal model, mapping to fear, anger, distress and
  nervousness, abstention on silence and low confidence, and that a victim is never shown as "happy".
- **Distress track:** role gating (off for suspects) and abstention on poor audio or unvalidated languages.
- **Hinglish handling:** number words and identifiers, evidence span mapping, demanded vs lost amounts, and
  telling UTR, phone and account numbers apart.
- **Triage and alerts:** emotion never changes financial priority; coercion and crisis phrases are detected.
- **Investigation tools:** Ring Linker, synthetic-voice acoustic cues, and that officer corrections stick.
- **Security:** every route needs a session, role permissions, CSRF, encryption at rest, and audit-chain tamper
  detection.
- **Failure handling:** the app fails honestly when no ASR model is installed (nothing is simulated).

## 2. Field extraction from Hinglish speech

**Method.** `backend/eval/cases.jsonl` has 20 phrases written the way callers speak, for example:

- *"maine dedh lakh rupaye bheje ramesh at the rate okaxis pe"* → ₹1,50,000 and `ramesh@okaxis`
- *"mere saadhe paanch hazaar kat gaye aadha ghanta pehle"* → ₹5,500, 30 minutes ago

Each phrase lists the fields we expect. The script runs the normaliser and extractor and compares each field
exactly: amount, incident time, UPI ID, phone, account, IFSC, UTR, bank, payment app and claimed agency
(38 checks in total).

**Result.** 38 / 38 correct.

**What this means.** The spoken forms we designed for are handled. This is a **development set** written alongside
the code, so it is not an estimate of accuracy on real calls or on speech-to-text output, where transcription errors
come first.

## 3. Speech emotion recognition

### What is evaluated

The live signals come from two parts. `run_ser_eval.py` tests the model part using **the same decision rules as
the live call**:

1. **Model emotion.** emotion2vec+ (`iic/emotion2vec_plus_base`) on 3-second voiced windows. Low-confidence results
   (< 0.45) become *uncertain*; silent or mostly unvoiced audio is *abstained*.
2. **Acoustic arousal.** Change in loudness, pitch and speaking rate against the caller's own first seconds. This
   is a within-call comparison, so it is covered by the unit tests rather than by single-clip datasets.

### Datasets

| Source | What it is | Why we use it | Labels |
|---|---|---|---|
| **CREMA-D** | 7,442 acted English clips from 91 speakers of varied age and ethnicity | Standard, public SER benchmark with balanced emotions | neutral, anger, fear, sadness (happy and disgust are excluded) |
| **Own role-play recordings** | Victim-call scenarios acted in Hindi, Hinglish and English and recorded on a phone microphone | Closest to the real use case: telephone audio, Indian accents, code-mixing | labelled with the emotion the speaker was asked to act |
| **Public YouTube clips** | Short segments of publicly posted scam-call recordings and news reports with clearly audible emotion | Real, unacted emotion and real telephone noise | labelled by listening; clips with unclear emotion are dropped, not guessed |

Audio is **not** stored in this repository (licences and privacy). YouTube clips are used only for testing, never
for training; the source links are kept outside the repository. Clips are cut to the caller's voice only.

### Protocol

```bash
# CREMA-D: balanced random sample (seed 7), labels taken from the file names
python eval/run_ser_eval.py --crema-d PATH/TO/CREMA-D/AudioWAV --limit 800

# Own and YouTube clips: a CSV with path,label,source,lang
python eval/run_ser_eval.py --manifest eval/audio/manifest.csv --role victim
```

Example `manifest.csv`:

```
path,label,source,lang
roleplay/r01.wav,fearful,roleplay,hi
roleplay/r02.wav,neutral,roleplay,hinglish
youtube/y07.wav,angry,youtube,hi
```

### Metrics

- **4-class accuracy** on clips the system answered.
- **"Needs attention" precision / recall / F1:** fear, anger or sadness vs neutral. This is the decision that matters
  for prioritising a call.
- **Abstention rate:** how often the system said *uncertain* instead of guessing. A higher rate on noisy audio is
  the intended behaviour.
- **Per-class recall** and the most common confusion.
- **Model time per clip** on the demo laptop (CPU).

### Results

> Results are written by the script to `backend/eval/ser_results.md`. Paste the tables below.

**CREMA-D**

*(paste results here)*

**Own role-play and YouTube clips**

*(paste results here)*

### How to read the results

- CREMA-D is acted, studio-quality English. Results there are an **upper bound**, not what to expect on a phone line.
- The role-play and YouTube sets are small. Treat them as a check that the system behaves sensibly on Indian
  call-like audio, not as a precise accuracy figure.
- Fear and sadness are often confused with each other in SER. For triage this matters less, because both raise the
  *distress* signal.

## 4. Live performance

Measured on the demo laptop (CPU only), using the timings in `logs/backend.log` during a live session:

| Metric | How |
|---|---|
| Speech-to-text time per ~5 s chunk (`small`, int8) | `live …: chunk N s … -> K segments in T s` lines in `logs/backend.log` |
| Time from speech to emotion update on screen | emotion readings are emitted every 1.5 s of audio |

---

## Limitations

1. **No public Indian-language emotional call dataset exists.** Our Hindi and Hinglish evaluation relies on a small
   set of role-play recordings and public clips, so per-language results are indicative only.
2. **Acted vs real emotion.** CREMA-D and role-play are acted; real distress can sound different (quieter, flatter,
   or masked by politeness).
3. **Telephone audio.** Narrow bandwidth, compression and background noise lower model confidence. The system is
   designed to abstain in these cases rather than guess, which lowers coverage.
4. **Model bias.** emotion2vec+ was trained mostly on Chinese and English speech. Accents, age, gender and
   speaking style can shift its output. We report only languages we actually tested
   (`ANVESH_VALIDATED_LANGS=en,hi`); the distress track abstains for others.
5. **Emotion is not intent.** A fearful voice does not prove someone is a victim, and a calm voice does not prove
   someone is lying. Outputs are indicative signals for an officer, never evidence. Emotion scoring is switched off
   for suspect recordings.
6. **Extraction depends on transcription.** The 38 / 38 extraction result is on clean text. On live audio,
   speech-to-text errors (especially digits and UPI handles) come first, which is why every field shows its source
   audio and can be corrected.
7. **Synthetic-voice detection** works on known generation methods and can miss new ones. It is shown as *needs
   verification*, never as proof.
8. **Short timeline.** The development sets were written by the same person who wrote the code. A
   held-out set written by someone else would give a fairer number.

## Next steps

- Collect consented, anonymised helpline-style recordings with a partner cyber cell to build a real Hinglish test
  set.
- Fine-tune the emotion model on commercially licensed Indian-language data, and report per-language and per-gender
  results.
- Measure end-to-end extraction accuracy on audio (speech-to-text → fields) and the time saved per intake against a
  manual form.
