"""Crisis and danger words in Indian languages other than Hindi and English.

The automatic helpline hand-off (handoff.py) starts from these signals, so they must exist in every language
Anvesh transcribes, not only in Hindi, English and Hinglish. IndicConformer writes each language in its own
script, so each list is in that script. Entries are short stems that survive the common inflections
("I will die", "I want to die" share one stem) and are kept specific, so ordinary complaint talk does not match.

Limits (also stated in EVALUATION.md): the lists were written without native-speaker review and have not been
measured on real calls. They raise a cancellable hand-off for a human officer; they never decide anything alone.
Urdu speech is transcribed as Hindi (Devanagari), so the Urdu-script list only matters for typed or imported text.
"""

from __future__ import annotations

# language code -> phrases about suicide, self-harm or not wanting to live  (-> Tele-MANAS 14416)
CRISIS = {
    "hi": ["आत्महत्या", "ख़ुदकुशी", "खुदकुशी", "aatmahatya", "atmahatya", "khudkushi", "मरना चाहता", "मरना चाहती",
           "खुद को खत्म", "अपनी जान ले"],
    "bn": ["আত্মহত্যা", "মরে যেতে চাই", "মরে যাব", "বাঁচতে চাই না", "বেঁচে থাকতে চাই না", "নিজেকে শেষ করে",
           "আর বাঁচব না"],
    "as": ["আত্মহত্যা", "মৰি যাম", "মৰিব বিচাৰো", "জীয়াই থাকিব নিবিচাৰো", "নিজকে শেষ কৰি"],
    "mr": ["आत्महत्या", "जगायचं नाही", "जगायचे नाही", "मरायचं आहे", "मरायचे आहे", "मरून जाईन", "स्वतःला संपवून",
           "जीव द्यायचा", "जीव देईन"],
    "ne": ["आत्महत्या", "मर्न मन", "बाँच्न मन छैन", "मर्छु", "आफूलाई सिध्याउँछु"],
    "gu": ["આત્મહત્યા", "જીવવું નથી", "મરી જવું છે", "મરી જઈશ", "જીવ આપી દઈશ"],
    "pa": ["ਖੁਦਕੁਸ਼ੀ", "ਆਤਮ ਹੱਤਿਆ", "ਆਤਮਹੱਤਿਆ", "ਜੀਣਾ ਨਹੀਂ", "ਮਰ ਜਾਵਾਂਗਾ", "ਮਰ ਜਾਵਾਂਗੀ", "ਮਰਨਾ ਚਾਹੁੰਦਾ",
           "ਮਰਨਾ ਚਾਹੁੰਦੀ"],
    "or": ["ଆତ୍ମହତ୍ୟା", "ମରିଯିବି", "ମରିବାକୁ ଚାହେଁ", "ବଞ୍ଚିବାକୁ ଇଚ୍ଛା ନାହିଁ"],
    "ta": ["தற்கொலை", "சாக வேண்டும்", "சாகப் போகிறேன்", "செத்துவிடுவேன்", "வாழ விருப்பமில்லை", "வாழ பிடிக்கவில்லை"],
    "te": ["ఆత్మహత్య", "చచ్చిపోవాలని", "చనిపోవాలని", "చచ్చిపోతాను", "బతకాలని లేదు", "బ్రతకాలని లేదు"],
    "kn": ["ಆತ್ಮಹತ್ಯೆ", "ಸಾಯಬೇಕು", "ಸಾಯುತ್ತೇನೆ", "ಸತ್ತು ಹೋಗುತ್ತೇನೆ", "ಬದುಕಲು ಇಷ್ಟವಿಲ್ಲ", "ಬದುಕೋಕೆ ಇಷ್ಟ ಇಲ್ಲ"],
    "ml": ["ആത്മഹത്യ", "മരിക്കണം", "ചാകണം", "മരിക്കാൻ പോകുന്നു", "ജീവിക്കാൻ തോന്നുന്നില്ല", "ജീവിക്കണ്ട"],
    "ur": ["خودکشی", "مرنا چاہتا", "مرنا چاہتی", "جینا نہیں", "اپنی جان لے"],
}

# language code -> a threat to the caller's life or safety right now  (-> Emergency 112)
DANGER = {
    "hi": ["जान से मारने", "मार डालेंगे", "घर के बाहर खड़े", "अगवा कर"],
    "bn": ["মেরে ফেলবে", "মেরে ফেলব", "মেরে ফেলার হুমকি", "প্রাণনাশের হুমকি", "বাড়ির বাইরে দাঁড়িয়ে", "অপহরণ"],
    "as": ["মাৰি পেলাব", "মাৰি পেলাম", "প্ৰাণনাশৰ ভাবুকি", "অপহৰণ"],
    "mr": ["मारून टाकतील", "मारून टाकेन", "मारून टाकू", "जीवे मारण्याची", "घराबाहेर उभे", "अपहरण"],
    "ne": ["मारिदिन्छु", "मारिदिन्छन्", "ज्यान मार्ने", "अपहरण"],
    "gu": ["મારી નાખશે", "મારી નાખીશ", "જાનથી મારી", "ઘરની બહાર ઊભા", "અપહરણ"],
    "pa": ["ਮਾਰ ਦੇਣਗੇ", "ਜਾਨੋਂ ਮਾਰ", "ਜਾਨ ਤੋਂ ਮਾਰ", "ਘਰ ਦੇ ਬਾਹਰ ਖੜ੍ਹੇ", "ਅਗਵਾ ਕਰ"],
    "or": ["ମାରିଦେବେ", "ମାରିଦେବି", "ଜୀବନରେ ମାରିବା", "ଅପହରଣ"],
    "ta": ["கொன்றுவிடுவேன்", "கொன்று விடுவேன்", "கொன்றுவிடுவார்கள்", "கொன்று விடுவார்கள்", "கொலை மிரட்டல்"],
    "te": ["చంపేస్తాను", "చంపేస్తారు", "చంపుతాం", "చంపేస్తామని", "ప్రాణహాని", "కిడ్నాప్"],
    "kn": ["ಕೊಲ್ಲುತ್ತೇನೆ", "ಕೊಲ್ಲುತ್ತಾರೆ", "ಕೊಂದು ಬಿಡುತ್ತಾರೆ", "ಜೀವ ಬೆದರಿಕೆ", "ಅಪಹರಣ"],
    "ml": ["കൊന്നുകളയും", "കൊല്ലും", "വധഭീഷണി", "തട്ടിക്കൊണ്ടുപോ"],
    "ur": ["جان سے مار", "مار ڈالیں گے", "اغوا"],
}


def flat(table: dict[str, list[str]]) -> list[str]:
    out: list[str] = []
    for words in table.values():
        out += [w for w in words if w not in out]
    return out


def language_of(phrase: str) -> str | None:
    for table in (CRISIS, DANGER):
        for lang, words in table.items():
            if phrase in words:
                return lang
    return None


LANGUAGES = sorted(set(CRISIS) | set(DANGER))
