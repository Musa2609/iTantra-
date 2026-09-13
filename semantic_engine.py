"""
iTantra Semantic Communication Engine (Python Port)
Parity with Android HybridIntentClassifier, IntentTaxonomy, and SemanticPayload.
"""

import os
import json
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

class IntentCategory(Enum):
    # (type_id, label, default_severity, is_emergency)
    MEDICAL = (0, "Medical", 2, True)
    FIRE = (1, "Fire", 3, True)
    RESCUE = (2, "Rescue", 3, True)
    EVACUATION = (3, "Evacuation", 2, True)
    SAFE = (4, "Safe", 0, False)
    INJURED = (5, "Injured", 2, True)
    TRAPPED = (6, "Trapped", 3, True)
    MISSING = (7, "Missing", 1, True)

    def __init__(self, type_id: int, label: str, default_severity: int, is_emergency: bool):
        self.type_id = type_id
        self.label = label
        self.default_severity = default_severity
        self.is_emergency = is_emergency

    @classmethod
    def from_id(cls, type_id: int) -> "IntentCategory":
        for cat in cls:
            if cat.type_id == type_id:
                return cat
        return cls.SAFE

    @classmethod
    def from_name(cls, name: str) -> "IntentCategory":
        for cat in cls:
            if cat.label.lower() == name.lower() or cat.name.lower() == name.lower():
                return cat
        return cls.SAFE

@dataclass
class SemanticPayload:
    """
    10-bit bit-packed semantic payload representation:
    - Type ID: 4 bits (0-15)
    - Intent ID: 4 bits (0-15)
    - Severity: 2 bits (0-3)
    Total: 10 bits (packed into 2 bytes)
    """
    type_id: int      # 4 bits (0-15)
    intent_id: int    # 4 bits (0-15)
    severity: int     # 2 bits (0-3)

    def encode(self) -> bytes:
        t = self.type_id & 0x0F
        i = self.intent_id & 0x0F
        s = self.severity & 0x03
        b0 = (t << 4) | i
        b1 = (s << 6)
        return bytes([b0, b1])

    @classmethod
    def decode(cls, data: bytes) -> "SemanticPayload":
        if len(data) < 2:
            raise ValueError("SemanticPayload requires at least 2 bytes (10 active bits)")
        b0 = data[0]
        b1 = data[1]
        t = (b0 >> 4) & 0x0F
        i = b0 & 0x0F
        s = (b1 >> 6) & 0x03
        return cls(type_id=t, intent_id=i, severity=s)

@dataclass
class ClassificationResult:
    category: IntentCategory
    intent_id: int
    severity: int
    confidence: float
    is_emergency: bool
    mode_name: str
    matched_keyword: Optional[str] = None
    reason: str = ""

class KeywordMatcher:
    KEYWORDS: Dict[IntentCategory, List[str]] = {
        IntentCategory.MEDICAL: [
            "doctor", "medicine", "medical", "first aid", "hospital", "ambulance", "fever", "bleeding", "sick", "headache", "asthma", "heart attack", "diabetic", "medic", "paramedic", "critical", "clinic",
            "डॉक्टर", "चिकित्सा", "एम्बुलेंस", "दवा", "बुखार", "खून", "मरीज", "इलाज", "अस्पताल", "मेडिकल", "चिकित्सीय", "स्वास्थ्य",
            "மருத்துவர்", "மருத்துவம்", "ஆம்புலன்ஸ்", "மருந்து", "காய்ச்சல்", "ரத்தப்போக்கு", "நோயாளி", "சிகிச்சை", "மருத்துவ", "மெடிக்கல்"
        ],
        IntentCategory.FIRE: [
            "fire", "flames", "smoke", "burn", "explosion", "firefighter", "gas leak", "wildfire", "short circuit",
            "आग", "धुआं", "लपटें", "फायर", "ब्लास्ट", "सिलेंडर", "जलने", "दमकल",
            "தீ", "புகை", "தீயணைப்பு", "வெடிப்பு", "சிலிண்டர்", "தீக்காயம்"
        ],
        IntentCategory.RESCUE: [
            "rescue", "flood", "helicopter", "stranded", "boat", "ndrf", "landslide", "submerged", "submerge", "river", "drowning", "emergency", "help", "danger", "sos", "save us", "distress", "urgent",
            "बचाव", "बाढ़", "हेलीकॉप्टर", "नाव", "मलबा", "भूस्खलन", "नदी", "एनडीआरएफ", "आपातकाल", "आपातकालीन", "मदद", "सहायता", "बचाओ", "खतरा", "खतरे", "खतरनाक", "इमरजेंसी",
            "மீட்பு", "வெள்ளம்", "ஹெலிகாப்டர்", "படகு", "நிலச்சரிவு", "ஆறு", "அவசரம்", "அவசர", "உதவி", "காப்பாற்றுங்கள்", "காப்பாத்துங்க", "உதவுங்கள்", "ஆபத்து", "ஆபத்தான", "எமர்ஜென்சி"
        ],
        IntentCategory.EVACUATION: [
            "evacuate", "evacuation", "shelter", "cyclone", "relocate", "relocation", "clear area", "warning", "tsunami",
            "खाली", "निकासी", "आश्रय", "चक्रवात", "शिविर", "स्थानांतरण",
            "காலி", "வெளியேற்ற", "தங்குமிடம்", "சூறாவளி", "முகாம்"
        ],
        IntentCategory.SAFE: [
            "safe", "secure", "fine", "unhurt", "ok", "okay", "reached", "alive", "sound", "hello", "how are you", "good morning", "meeting", "weather",
            "सुरक्षित", "कुशल", "ठीक", "पहुंच", "कोई खतरा नहीं", "नमस्ते", "कैसे हैं", "मौसम",
            "பாதுகாப்பு", "நலமாக", "நன்றாக", "அடைந்தோம்", "வணக்கம்", "எப்படி இருக்கிறீர்கள்"
        ],
        IntentCategory.INJURED: [
            "injured", "injury", "fracture", "wounded", "cuts", "broken leg", "broken arm", "bleeding", "hurt", "pain", "unconscious", "wound",
            "घायल", "चोट", "फक्चर", "कटा", "टूटा", "दर्द", "बेहोश", "गंभीर",
            "காயம்", "காயமடைந்து", "முறிவு", "ரத்தம்", "வலி", "மயக்கம்", "கடுமையான"
        ],
        IntentCategory.TRAPPED: [
            "trapped", "stuck", "rubble", "collapsed", "under debris", "elevator stuck", "tunnel",
            "फंसे", "फंसा", "मलबा", "इमारत ढह", "सुरंग",
            "சிக்கி", "சிக்கியுள்ளனர்", "இடிபாடுகள்", "சுரங்கம்"
        ],
        IntentCategory.MISSING: [
            "missing", "lost", "unreachable", "disappeared", "search party", "search",
            "लापता", "बिछड़", "खो गया", "संपर्क नहीं",
            "காணவில்லை", "காணாமல்", "தேடுதல்"
        ]
    }

    @classmethod
    def match(cls, text: str) -> Optional[Tuple[IntentCategory, float, str]]:
        lower_text = text.lower().strip()
        if not lower_text:
            return None

        # Check SAFE phrases explicitly to avoid false alarms
        for word in cls.KEYWORDS[IntentCategory.SAFE]:
            if word in lower_text:
                # If conversational greeting/safe phrase with no emergency words
                has_emergency = False
                for cat in [IntentCategory.FIRE, IntentCategory.MEDICAL, IntentCategory.RESCUE, IntentCategory.TRAPPED]:
                    for kw in cls.KEYWORDS[cat]:
                        if kw in lower_text:
                            has_emergency = True
                            break
                    if has_emergency:
                        break
                if not has_emergency:
                    return (IntentCategory.SAFE, 0.95, word)

        best_cat = None
        max_matches = 0
        best_kw = None

        for cat, kw_list in cls.KEYWORDS.items():
            matches = 0
            curr_kw = None
            for kw in kw_list:
                if kw in lower_text:
                    matches += 1
                    if curr_kw is None:
                        curr_kw = kw
            if matches > max_matches:
                max_matches = matches
                best_cat = cat
                best_kw = curr_kw

        if best_cat and max_matches > 0:
            confidence = 1.0 if max_matches >= 2 else 0.92
            return (best_cat, confidence, best_kw or "")

        return None

class HybridIntentClassifier:
    def __init__(self, dataset_path: Optional[str] = None):
        self.dataset = []
        if dataset_path and os.path.exists(dataset_path):
            try:
                with open(dataset_path, "r", encoding="utf-8") as f:
                    self.dataset = json.load(f)
            except Exception as e:
                print(f"[HybridClassifier] Warning: Could not load dataset: {e}")

    def classify(self, text: str) -> ClassificationResult:
        if not text or not text.strip():
            return ClassificationResult(
                category=IntentCategory.SAFE,
                intent_id=0,
                severity=0,
                confidence=0.0,
                is_emergency=False,
                mode_name="NONE",
                reason="Empty transcript"
            )

        clean_text = text.strip()
        lower_clean = clean_text.lower()

        # Guard: Noise / single-character CTC hallucinations / artifacts
        if len(lower_clean) < 3:
            return ClassificationResult(
                category=IntentCategory.SAFE,
                intent_id=0,
                severity=0,
                confidence=0.0,
                is_emergency=False,
                mode_name="NONE",
                reason="Transcript too short (< 3 characters)"
            )

        # Guard: If ASR produced a no-speech placeholder or bracketed artifact, do not classify
        if (
            lower_clean.startswith("—")
            or lower_clean.startswith("-")
            or lower_clean.startswith("[")
            or "no speech recognized" in lower_clean
            or "asr unavailable" in lower_clean
        ):
            return ClassificationResult(
                category=IntentCategory.SAFE,
                intent_id=0,
                severity=0,
                confidence=0.0,
                is_emergency=False,
                mode_name="NONE",
                reason="No speech recognized in audio"
            )

        # Step 1: Rule-based fast path via KeywordMatcher
        kw_match = KeywordMatcher.match(clean_text)
        if kw_match:
            cat, conf, kw = kw_match
            is_emergency = cat.is_emergency and conf >= 0.70
            return ClassificationResult(
                category=cat,
                intent_id=0,
                severity=cat.default_severity,
                confidence=conf,
                is_emergency=is_emergency,
                mode_name="MODE 1 SEMANTIC" if is_emergency else "MODE 2 FREE TEXT",
                matched_keyword=kw,
                reason=f"Keyword fast-path matched: '{kw}'"
            )

        # Step 2: Phrase-level token and character n-gram matching against intent dataset
        STOPWORDS = {
            "in", "at", "the", "a", "an", "is", "are", "am", "was", "were", "to", "of", "and", "or",
            "for", "with", "on", "it", "by", "as", "from", "be", "this", "that", "there", "i", "you",
            "he", "she", "we", "they", "me", "him", "her", "us", "them", "my", "your", "our", "their"
        }

        import re
        tokens = set(re.findall(r'\b\w+\b', lower_clean))
        content_tokens = tokens - STOPWORDS

        if not content_tokens:
            return ClassificationResult(
                category=IntentCategory.SAFE,
                intent_id=0,
                severity=0,
                confidence=0.10,
                is_emergency=False,
                mode_name="MODE 2 FREE TEXT",
                reason="Conversational tokens (no emergency content words)"
            )

        best_score = 0.0
        best_cat = IntentCategory.SAFE
        best_sev = 0
        best_intent_id = 0

        for entry in self.dataset:
            cat_name = entry.get("category", "")
            cat = IntentCategory.from_name(cat_name)
            type_id = entry.get("type_id", 0)
            sev = entry.get("severity", cat.default_severity)

            phrases_dict = entry.get("phrases", {})
            for lang, phrase_list in phrases_dict.items():
                for p in phrase_list:
                    p_lower = p.lower()
                    p_tokens = set(re.findall(r'\b\w+\b', p_lower)) - STOPWORDS
                    if not p_tokens:
                        continue
                    
                    # Content Word Jaccard
                    intersection = len(content_tokens.intersection(p_tokens))
                    union = len(content_tokens.union(p_tokens))
                    jaccard = intersection / union if union > 0 else 0.0

                    # Substring overlap (requires at least 4 characters on both sides to prevent noise)
                    sub_score = 0.0
                    if len(p_lower) >= 4 and len(lower_clean) >= 4:
                        if p_lower in lower_clean or (lower_clean in p_lower and len(lower_clean) >= len(p_lower) * 0.6):
                            sub_score = 0.85

                    score = max(jaccard, sub_score)
                    if score > best_score:
                        best_score = score
                        best_cat = cat
                        best_sev = sev
                        best_intent_id = type_id

        # Contrastive margin threshold
        if best_score >= 0.35 and best_cat.is_emergency:
            confidence = min(0.95, best_score + 0.35)
            is_emergency = True
            mode_name = "MODE 1 SEMANTIC"
        else:
            confidence = max(0.10, best_score)
            is_emergency = False
            best_cat = IntentCategory.SAFE
            best_sev = 0
            best_intent_id = 0
            mode_name = "MODE 2 FREE TEXT"

        return ClassificationResult(
            category=best_cat,
            intent_id=best_intent_id,
            severity=best_sev,
            confidence=round(confidence, 3),
            is_emergency=is_emergency,
            mode_name=mode_name,
            reason="Semantic phrase similarity evaluation"
        )

# Receiver Language Templates (Canonical 1:1 parity with IntentCategory type_id)
# 0: Medical, 1: Fire, 2: Rescue, 3: Evacuation, 4: Safe, 5: Injured, 6: Trapped, 7: Missing
EMERGENCY_TEMPLATES = {
    "en": {
        0: "Medical assistance needed immediately.",
        1: "Fire emergency! Evacuation required.",
        2: "Emergency rescue needed immediately.",
        3: "Immediate evacuation order in effect.",
        4: "All clear. Situation is safe and normal.",
        5: "Casualties reported! Medical triage required.",
        6: "People trapped under debris! Send search and rescue.",
        7: "Persons missing! Search operation required."
    },
    "hi": {
        0: "तुरंत चिकित्सा सहायता की आवश्यकता है।",
        1: "आग की आपात स्थिति! निकासी की आवश्यकता है।",
        2: "तुरंत आपातकालीन बचाव की आवश्यकता है।",
        3: "तत्काल निकासी का आदेश जारी है।",
        4: "सब ठीक है। स्थिति सुरक्षित और सामान्य है।",
        5: "घायल लोग हैं! तत्काल चिकित्सा देखभाल की आवश्यकता है।",
        6: "लोग मलबे में फंसे हैं! खोज और बचाव दल भेजें।",
        7: "व्यक्ति लापता हैं! खोज अभियान की आवश्यकता है।"
    },
    "ta": {
        0: "உடனடியாக மருத்துவ உதவி தேவைப்படுகிறது.",
        1: "தீ அவசரநிலை! வெளியேற்றம் தேவைப்படுகிறது.",
        2: "உடனடி அவசர மீட்பு தேவைப்படுகிறது.",
        3: "உடனடி வெளியேற்ற உத்தரவு அமலில் உள்ளது.",
        4: "அனைத்தும் தெளிவாக உள்ளது. நிலைமை பாதுகாப்பானது.",
        5: "காயமடைந்தவர்கள் உள்ளனர்! அவசர சிகிச்சை தேவை.",
        6: "மக்கள் இடிபாடுகளில் சிக்கியுள்ளனர்! மீட்புக் குழுவை அனுப்பவும்.",
        7: "நபர்களை காணவில்லை! தேடுதல் பணி தேவைப்படுகிறது."
    }
}

# Phonetic Romanized versions for clean English TTS speech synthesizers (Windows SAPI / espeak)
# Prevents SAPI from spelling out individual Unicode character names
EMERGENCY_TTS_PHONETICS = {
    "hi": {
        0: "Turant chikitsa sahayata ki aavashyakta hai!",
        1: "Aag ki aapaat sthiti! Nikaasi ki aavashyakta hai!",
        2: "Turant aapaatkaaleen bachaav ki aavashyakta hai!",
        3: "Tatkaal nikaasi ka aadesh jaari hai!",
        4: "Sab theek hai. Sthiti surakshit aur saamaanya hai.",
        5: "Ghaayal log hain! Tatkaal chikitsa dekhbhaal ki aavashyakta hai!",
        6: "Log malbe mein phanse hain! Khoj aur bachaav dal bhejein!",
        7: "Vyakti laapata hain! Khoj abhiyaan ki aavashyakta hai!"
    },
    "ta": {
        0: "Udanadiyaaga maruthuva udhavi thevaipadugiradhu.",
        1: "Thee avasaranilai! Veliyeatram thevaipadugiradhu.",
        2: "Udanadi avasara meetpu thevaipadugiradhu.",
        3: "Udanadi veliyeatra utharavu amalil ulladhu.",
        4: "Anaithum thelivaaga ulladhu. Nilaimai paadhukaapaanadhu.",
        5: "Kaayamadaindhavargal ullaargal! Avasara sigitchai thevai.",
        6: "Makkal idibaadugalil sikkiyullaargal! Meetpu kuzhuvai anuppavum.",
        7: "Nabargalai kaanavillai! Theadudhal pani thevaipadugiradhu."
    }
}

# General Devanagari transliterator for arbitrary Mode 2 Indic text
DEVA_VOWELS = {
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee', '\u0909': 'u',
    '\u090A': 'oo', '\u090B': 'ri', '\u090E': 'e', '\u090F': 'e', '\u0910': 'ai',
    '\u0912': 'o', '\u0913': 'o', '\u0914': 'au'
}

DEVA_MATRAS = {
    '\u093E': 'aa', '\u093F': 'i', '\u0940': 'ee', '\u0941': 'u', '\u0942': 'oo',
    '\u0943': 'ri', '\u0947': 'e', '\u0948': 'ai', '\u094B': 'o', '\u094C': 'au',
    '\u0902': 'n', '\u0901': 'n', '\u0903': 'h'
}

DEVA_CONSONANTS = {
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091A': 'ch', '\u091B': 'chh', '\u091C': 'j', '\u091D': 'jh', '\u091E': 'ny',
    '\u091F': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092A': 'p', '\u092B': 'ph', '\u092C': 'b', '\u092D': 'bh', '\u092E': 'm',
    '\u092F': 'y', '\u0930': 'r', '\u0932': 'l', '\u0933': 'l', '\u0935': 'v',
    '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    '\u0958': 'q', '\u0959': 'kh', '\u095A': 'gh', '\u095B': 'z', '\u095C': 'r', '\u095D': 'rh', '\u095E': 'f'
}

VIRAMA = '\u094D'

def transliterate_devanagari_to_roman(text: str) -> str:
    """Convert Devanagari Hindi text to phonetically speakable Roman text for SAPI."""
    import re
    res = []
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        if c in DEVA_VOWELS:
            res.append(DEVA_VOWELS[c])
        elif c in DEVA_CONSONANTS:
            base = DEVA_CONSONANTS[c]
            if i + 1 < n:
                nxt = text[i + 1]
                if nxt == VIRAMA:
                    res.append(base)
                    i += 1
                elif nxt in DEVA_MATRAS:
                    res.append(base + DEVA_MATRAS[nxt])
                    i += 1
                elif nxt in DEVA_CONSONANTS or nxt in DEVA_VOWELS or nxt == ' ':
                    res.append(base + 'a' if nxt != ' ' else base)
                else:
                    res.append(base + 'a')
            else:
                res.append(base)
        elif c in DEVA_MATRAS:
            res.append(DEVA_MATRAS[c])
        elif c == '।':
            res.append('.')
        else:
            res.append(c)
        i += 1
    out = "".join(res)
    out = re.sub(r'\s+', ' ', out).strip()
    return out

def get_receiver_emergency_template(type_id: int, lang_code: str = "en") -> str:
    """Retrieve canonical receiver text for a decoded type_id. Never defaults to Medical (0)."""
    lang = lang_code.lower()
    if lang not in EMERGENCY_TEMPLATES:
        lang = "en"
    templates = EMERGENCY_TEMPLATES[lang]
    if type_id in templates:
        return templates[type_id]
    return "Emergency alert received. Immediate assistance required."

def get_receiver_emergency_phonetic(type_id: int, lang_code: str = "en") -> str:
    """Retrieve phonetic representation for speech synthesis."""
    lang = lang_code.lower()
    if lang in EMERGENCY_TTS_PHONETICS and type_id in EMERGENCY_TTS_PHONETICS[lang]:
        return EMERGENCY_TTS_PHONETICS[lang][type_id]
    return get_receiver_emergency_template(type_id, lang_code)

