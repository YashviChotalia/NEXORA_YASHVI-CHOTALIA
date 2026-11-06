"""Coercion and crisis signals in a victim's or informant's own words.

These are *assistance* signals for a human officer, never findings. They are
only computed for victim and informant roles.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from . import safety_indic

COERCION_PHRASES = [
    "kisi ko mat batana", "kisi ko nahi batana", "kisi ko na batau", "don't tell anyone",
    "do not tell anyone", "told me not to tell", "abhi bhi call pe", "abhi bhi video call",
    "video call pe hain", "video call chal raha", "call chal rahi hai", "still on the call",
    "still on video", "camera on rakhne", "camera on rakhna", "camera band mat", "ghar se bahar mat",
    "room se bahar mat", "phone mat katna", "call mat kaatna", "they are watching",
    "dekh rahe hain", "nazar rakh rahe", "digital arrest mein", "arrest kar lenge",
    "mujhe arrest", "family ko pata nahi", "किसी को मत बताना", "वीडियो कॉल पर",
]

CRISIS_PHRASES = [
    "jeene ka mann nahi", "jine ka man nahi", "sab khatam ho gaya", "sab khatam", "mar jaunga",
    "mar jaungi", "marna chahta", "marna chahti", "khud ko kuch kar", "kuch kar lunga",
    "kuch kar lungi", "suicide", "end my life", "kill myself", "no way out", "life is over",
    "zindagi barbaad", "sab kuch chala gaya", "jaan de dunga", "jaan de dungi",
    "जीने का मन नहीं", "मर जाऊंगा", "सब खत्म",
    # self-harm
    "khud ko nuksan", "khud ko hurt", "hurt myself", "harm myself", "cut myself", "khud ko maar",
    "apni jaan le", "jaan le lunga", "jaan le lungi", "want to die", "wish i was dead", "better off dead",
    "zinda nahi rehna", "nahi jeena", "jeena nahi chahta", "jeena nahi chahti", "khatam kar lunga",
    "khatam kar lungi", "apne aap ko khatam", "खुद को नुकसान", "जीना नहीं", "जान दे दूंगा", "जान दे दूंगी",
    # strongly self-demeaning (worthlessness, being a burden): risk markers, not ordinary self-blame
    "main kisi kaam ka nahi", "main kisi kaam ki nahi", "main bekaar hoon", "main bojh hoon", "sab par bojh",
    "sabke liye bojh", "mujhse koi pyaar nahi", "mera koi nahi", "main ghatiya hoon", "kisi ko munh nahi dikha",
    "munh dikhane layak nahi", "i am worthless", "i'm worthless", "i am a burden", "i'm a burden",
    "nobody would miss me", "no one would miss me", "मैं किसी काम का नहीं", "मैं बोझ हूं", "मुंह दिखाने लायक नहीं",
]

# a threat to the caller's life or safety right now -> 112
DANGER_PHRASES = [
    "jaan se maar", "maar denge", "maar dalenge", "jaan ka khatra", "ghar ke bahar khade", "ghar pe aa gaye",
    "ghar aa gaye hain", "utha le jayenge", "kidnap", "agwa", "bandook", "goli maar", "chaku",
    "kill me", "will kill", "going to kill", "threatening to kill", "outside my house", "at my door",
    "they are coming", "with a gun", "with a knife", "जान से मार", "मार देंगे", "जान का खतरा",
]

# a woman facing sextortion, stalking or harassment -> 181 (officer confirms)
WOMEN_PHRASES = [
    "meri photo morph", "photo morph", "nude photo", "nude video", "nangi video", "gandi video", "private photos",
    "private video", "viral kar denge", "viral kar dega", "leak kar denge", "stalk kar raha", "peecha kar raha",
    "pichha kar raha", "harass kar raha", "blackmail kar raha", "sextortion", "morphed", "leak my photos",
    "stalking me", "harassing me", "न्यूड वीडियो", "वायरल कर देंगे",
]

# the person at risk is a child -> 1098 (officer confirms)
CHILD_PHRASES = [
    "meri beti", "mera beta", "mera bachcha", "meri bachchi", "naabaalig", "nabalig", "minor hai", "school mein padhti",
    "school mein padhta", "class mein padhti", "class mein padhta", "saal ki hai", "saal ka hai", "my daughter",
    "my son", "my child", "she is a minor", "he is a minor", "मेरी बेटी", "मेरा बेटा", "नाबालिग",
]

DISTRESS_LEXICON = [
    "please help", "help kijiye", "madad", "bachao", "ro raha", "ro rahi", "dar lag raha",
    "bahut dar", "ghabra", "panic", "scared", "darr", "pareshan", "please sir", "sir please",
    "kya karun", "kya karu", "meri saari savings", "saari savings", "life savings",
    "help me", "please help me", "i am scared", "i'm scared", "what do i do", "what should i do",
    "bahut tension", "tension ho", "rona aa", "ro rahi hoon", "ro raha hoon", "ghabrahat", "mujhe dar",
    "kuch samajh nahi", "samajh nahi aa raha", "sab chala gaya", "loot liya", "dhokha", "please jaldi",
    "jaldi kuch karo", "kuch kijiye", "please kuch", "bahut bura", "barbaad", "mar gaya", "mar gayi",
    "मदद", "बचाओ", "डर लग", "परेशान", "प्लीज", "क्या करूं", "सब चला गया",
]

# the same signals in the other Indian languages Anvesh transcribes (safety_indic.py)
CRISIS_PHRASES += safety_indic.flat(safety_indic.CRISIS)
DANGER_PHRASES += safety_indic.flat(safety_indic.DANGER)

CRISIS_RESOURCE = "Tele-MANAS mental-health helpline: 14416 (24x7). Hand over to a trained person."


@dataclass
class SafetySignals:
    coercion_hits: list[dict] = field(default_factory=list)
    crisis_hits: list[dict] = field(default_factory=list)
    distress_lexical: list[dict] = field(default_factory=list)
    danger_hits: list[dict] = field(default_factory=list)
    women_hits: list[dict] = field(default_factory=list)
    child_hits: list[dict] = field(default_factory=list)


def _find(text: str, phrases: list[str]) -> list[tuple[str, int, int]]:
    """Exact match first; otherwise a spelling-tolerant match (quote then spans the whole line)."""
    from .lexicon import fold
    low = text.lower()
    folded = fold(text)
    out = []
    for p in phrases:
        hits = [(p, m.start(), m.end()) for m in re.finditer(rf"(?<!\w){re.escape(p)}", low)]
        if not hits and re.search(rf"(?<!\w){re.escape(fold(p))}", folded):
            hits = [(p, 0, len(text))]
        out.extend(hits)
    return out


def scan(segments) -> SafetySignals:
    sig = SafetySignals()
    for seg in segments:
        if getattr(seg, "speaker", None) == "officer":
            continue
        for bucket, phrases in ((sig.coercion_hits, COERCION_PHRASES), (sig.crisis_hits, CRISIS_PHRASES),
                                (sig.distress_lexical, DISTRESS_LEXICON), (sig.danger_hits, DANGER_PHRASES),
                                (sig.women_hits, WOMEN_PHRASES), (sig.child_hits, CHILD_PHRASES)):
            for p, s, e in _find(seg.text, phrases):
                n = max(len(seg.text), 1)
                dur = seg.end - seg.start
                bucket.append({
                    "segment_id": seg.id, "phrase": p, "quote": seg.text[s:e],
                    "lang": safety_indic.language_of(p),
                    "t_start": round(seg.start + dur * s / n, 2), "t_end": round(seg.start + dur * e / n, 2),
                })
    return sig
