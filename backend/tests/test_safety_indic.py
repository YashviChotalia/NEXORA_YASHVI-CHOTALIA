"""Crisis and danger words in Indian languages start the same hand-off as Hindi and English ones."""

from types import SimpleNamespace

from anvesh import handoff
from anvesh.nlp import safety, safety_indic


class Seg:
    def __init__(self, text, speaker="caller"):
        self.id, self.text, self.speaker, self.start, self.end = "s1", text, speaker, 0.0, 4.0


CRISIS_LINES = {
    "bn": "আমি আর বাঁচতে চাই না, সব টাকা চলে গেছে",
    "mr": "माझे सगळे पैसे गेले, मला आता जगायचं नाही",
    "gu": "બધા પૈસા ગયા, મારે હવે જીવવું નથી",
    "pa": "ਸਾਰੇ ਪੈਸੇ ਚਲੇ ਗਏ, ਹੁਣ ਮੈਂ ਜੀਣਾ ਨਹੀਂ ਚਾਹੁੰਦਾ",
    "ta": "எல்லா பணமும் போய்விட்டது, எனக்கு வாழ விருப்பமில்லை",
    "te": "డబ్బంతా పోయింది, నాకు బతకాలని లేదు",
    "kn": "ಎಲ್ಲಾ ಹಣ ಹೋಯಿತು, ನನಗೆ ಬದುಕಲು ಇಷ್ಟವಿಲ್ಲ",
    "ml": "പണം മുഴുവൻ പോയി, എനിക്ക് ജീവിക്കാൻ തോന്നുന്നില്ല",
    "or": "ସବୁ ଟଙ୍କା ଚାଲିଗଲା, ମୋର ବଞ୍ଚିବାକୁ ଇଚ୍ଛା ନାହିଁ",
    "as": "সকলো টকা গ'ল, মই মৰি যাম",
    "ne": "सबै पैसा गयो, मलाई बाँच्न मन छैन",
    "hi": "मैं आत्महत्या कर लूंगा",
    "ur": "میں خودکشی کر لوں گا",
}

DANGER_LINES = {
    "bn": "ওরা বলছে আমাকে মেরে ফেলবে",
    "mr": "ते मला मारून टाकतील असं म्हणाले",
    "ta": "என்னை கொன்றுவிடுவார்கள் என்று மிரட்டினார்கள்",
    "te": "నన్ను చంపేస్తామని బెదిరించారు",
    "kn": "ನನ್ನನ್ನು ಕೊಲ್ಲುತ್ತಾರೆ ಎಂದು ಹೇಳಿದರು",
    "ml": "അവർ എന്നെ കൊന്നുകളയും എന്ന് പറഞ്ഞു",
    "gu": "એ લોકો મને મારી નાખશે",
}

# ordinary fraud-complaint talk in the same languages: must not start a hand-off
CALM_LINES = [
    "আমার অ্যাকাউন্ট থেকে পঞ্চাশ হাজার টাকা কেটে নিয়েছে",
    "माझ्या खात्यातून पन्नास हजार रुपये गेले, OTP विचारला होता",
    "என் கணக்கிலிருந்து ஐம்பதாயிரம் ரூபாய் போய்விட்டது",
    "నా ఖాతా నుండి యాభై వేల రూపాయలు పోయాయి",
    "ਮੇਰੇ ਖਾਤੇ ਵਿੱਚੋਂ ਪੰਜਾਹ ਹਜ਼ਾਰ ਰੁਪਏ ਕੱਟੇ ਗਏ",
    "उनकी अगवानी के लिए हम स्टेशन गए थे",
]


def test_crisis_words_in_every_listed_language():
    for lang, line in CRISIS_LINES.items():
        hits = safety.scan([Seg(line)]).crisis_hits
        assert hits, lang
        assert hits[0]["lang"], lang  # the hit says which list it came from


def test_danger_words_in_indian_languages():
    for lang, line in DANGER_LINES.items():
        assert safety.scan([Seg(line)]).danger_hits, lang


def test_ordinary_complaints_raise_nothing():
    for line in CALM_LINES:
        s = safety.scan([Seg(line)])
        assert not s.crisis_hits and not s.danger_hits, line


def test_officer_words_are_ignored_in_any_language():
    assert not safety.scan([Seg("আপনি কি আত্মহত্যা করার কথা ভাবছেন?", speaker="officer")]).crisis_hits


def test_a_tamil_crisis_line_on_a_live_call_starts_tele_manas():
    hits = safety.scan([Seg(CRISIS_LINES["ta"])]).crisis_hits
    case = SimpleNamespace(id="AV-2026-TEST1", mode="victim", triage={}, safety={"crisis": True, "crisis_hits": hits})
    need = handoff.needed(case, live=True)
    assert need and need[0][0].key == "telemanas" and need[0][0].auto
    assert "வாழ விருப்பமில்லை" in need[0][1]


def test_every_transcribed_major_language_is_covered():
    for lang in ("bn", "mr", "gu", "pa", "ta", "te", "kn", "ml", "or", "as", "ne", "ur"):
        assert safety_indic.CRISIS.get(lang), lang
