"""
TASK 7.1 - MULTILINGUAL SENTIMENT ANALYSIS

Classifies customer messages as positive, neutral, negative,
frustrated, urgent or sarcastic across English, Hindi, Telugu and
Spanish, and reports a confidence score.

Two design points matter more than the rest:

1. Risk detection is deliberately INDEPENDENT of sentiment polarity.
   detect_high_risk_topics() reads the raw text and never consults the
   sentiment label. A calm, politely worded "I think I may have been
   charged twice" is neutral in tone and still critical in risk, so
   deriving risk from sentiment would silently miss it.

2. Sarcasm never erases the underlying complaint. A sarcastic message
   keeps its pre-sarcasm label in ``underlying_label`` and keeps a
   negative polarity, so downstream escalation still fires.

Everything is deterministic and offline. An optional, injectable
LLM fallback hook (``SentimentConfig.llm_fallback``) gives callers
a seam for a second opinion on low-confidence messages; it is None
by default, so this module always runs without an API key and its
behaviour is unchanged unless a hook is supplied.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import re
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from task6_multilingual import (
    ConversationMessage,
    Task6Config,
    detect_language,
)


class SentimentLabel(str, Enum):

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    FRUSTRATED = "frustrated"
    URGENT = "urgent"
    SARCASTIC = "sarcastic"


class RiskTopic(str, Enum):

    ACCOUNT_COMPROMISE = "account_compromise"
    DUPLICATE_PAYMENT = "duplicate_payment"
    LEGAL_THREAT = "legal_threat"
    UNAUTHORIZED_CHARGE = "unauthorized_charge"
    DATA_BREACH = "data_breach"


# Highest score wins; ties break on this order, most actionable first.
LABEL_TIE_ORDER = (
    SentimentLabel.URGENT,
    SentimentLabel.FRUSTRATED,
    SentimentLabel.NEGATIVE,
    SentimentLabel.SARCASTIC,
    SentimentLabel.POSITIVE,
    SentimentLabel.NEUTRAL,
)

NEGATIVE_LABELS = frozenset(
    {
        SentimentLabel.NEGATIVE.value,
        SentimentLabel.FRUSTRATED.value,
        SentimentLabel.SARCASTIC.value,
        SentimentLabel.URGENT.value,
    }
)

SEVERITY_ORDER = {
    "none": 0,
    "low": 1,
    "medium": 2,
    "high": 3,
    "critical": 4,
}

RISK_SEVERITY = {
    RiskTopic.ACCOUNT_COMPROMISE.value: "critical",
    RiskTopic.LEGAL_THREAT.value: "critical",
    RiskTopic.DATA_BREACH.value: "critical",
    RiskTopic.DUPLICATE_PAYMENT.value: "high",
    RiskTopic.UNAUTHORIZED_CHARGE.value: "high",
}


# ============================================================
# CONFIGURATION
# ============================================================

@dataclass
class SentimentConfig:

    supported_languages: Tuple[str, ...] = (
        "english",
        "hindi",
        "telugu",
        "spanish",
    )

    # Below this confidence the caller may ask the LLM for a second
    # opinion. The rule layer always clamps the final answer.
    minimum_confidence: float = 0.40

    # Optional second opinion for results below that confidence.
    # Called as ``llm_fallback(text, language, preliminary)`` and may
    # return a replacement SentimentResult, or None to keep the
    # rule-based one. None by default: no hook, no API call, fully
    # deterministic offline behaviour.
    llm_fallback: Optional[Callable] = None

    # Confidence returned when no marker fires at all.
    neutral_default_confidence: float = 0.55

    # Absolute evidence needed before confidence reaches 1.0.
    evidence_saturation: float = 1.2

    history_window: int = 6
    current_weight: float = 0.65
    history_weight: float = 0.35

    intensifier_multiplier: float = 1.5
    negation_window: int = 3

    sarcasm_min_score: float = 0.55
    sarcasm_positive_weight: float = 0.20
    sarcasm_base_score: float = 0.45

    # Weight a praise-plus-complaint polarity mismatch earns on its
    # own ("Fantastic, you charged me twice."): praise words fired as
    # a mock cheer while the same sentence carries a complaint.
    sarcasm_mismatch_weight: float = 0.75

    repeated_negative_window: int = 4
    repeated_negative_count: int = 2

    risk_topics_enabled: Tuple[str, ...] = (
        RiskTopic.ACCOUNT_COMPROMISE.value,
        RiskTopic.DUPLICATE_PAYMENT.value,
        RiskTopic.LEGAL_THREAT.value,
        RiskTopic.UNAUTHORIZED_CHARGE.value,
        RiskTopic.DATA_BREACH.value,
    )

    language_config: Task6Config = field(default_factory=Task6Config)


@dataclass
class SentimentResult:

    label: str = SentimentLabel.NEUTRAL.value

    # Label before the sarcasm override.
    underlying_label: str = SentimentLabel.NEUTRAL.value

    confidence: float = 0.0

    # -1.0 (very negative) .. 1.0 (very positive), independent of the
    # six-class label.
    polarity: float = 0.0

    intensity: float = 0.0

    sarcasm: bool = False
    sarcasm_score: float = 0.0

    urgency_score: float = 0.0

    label_scores: Dict[str, float] = field(default_factory=dict)

    risk_topics: List[str] = field(default_factory=list)
    risk_severity: str = "none"

    language: str = "unknown"

    evidence: List[str] = field(default_factory=list)

    repeated_negative: bool = False
    message_count: int = 1


# ============================================================
# LEXICONS
# ============================================================

# language -> label -> phrase -> weight
SENTIMENT_LEXICON: Dict[str, Dict[str, Dict[str, float]]] = {

    "english": {

        "positive": {
            "thanks": 0.6,
            "thank you": 0.6,
            "great": 0.7,
            "excellent": 0.8,
            "perfect": 0.8,
            "resolved": 0.7,
            "love": 0.7,
            "awesome": 0.7,
            "wonderful": 0.7,
            "fantastic": 0.7,
            "brilliant": 0.7,
            "good": 0.5,
            "helpful": 0.6,
            "appreciate": 0.6,
            "happy": 0.6,
            "satisfied": 0.7,
            "glad": 0.5,
            "working now": 0.7,
        },

        "negative": {
            "bad": 0.6,
            "worst": 0.9,
            "terrible": 0.9,
            "awful": 0.9,
            "angry": 0.8,
            "useless": 0.8,
            "broken": 0.7,
            "damaged": 0.7,
            "late": 0.5,
            "delay": 0.5,
            "delayed": 0.5,
            "missing": 0.6,
            "wrong": 0.5,
            "complaint": 0.6,
            "unacceptable": 0.85,
            "poor": 0.6,
            "not received": 0.7,
            "never received": 0.8,
            "no response": 0.7,
            "disappointed": 0.7,
            "ridiculous": 0.8,
            "pathetic": 0.85,
            "nonsense": 0.75,
            "rude": 0.8,
            "no one is responding": 0.8,
            "nothing has happened": 0.75,
            # The most common complaint words in support traffic.
            # Their absence made "this is very frustrating" read as
            # neutral, which is the single worst miss available.
            "frustrating": 0.8,
            "frustrated": 0.8,
            "annoying": 0.7,
            "annoyed": 0.7,
            "upset": 0.7,
            "unhelpful": 0.75,
            "appalling": 0.85,
            "disgusting": 0.85,
            "shocking": 0.7,
            "fed up": 0.8,
            "sick of": 0.8,
            "not happy": 0.65,
        },

        "frustrated": {
            "again and again": 0.8,
            "still not": 0.8,
            "still no": 0.75,
            "how many times": 0.85,
            "third time": 0.85,
            "second time": 0.6,
            "yet again": 0.8,
            "once again": 0.7,
            "still waiting": 0.75,
            "i already told": 0.7,
            "keep telling": 0.8,
            "no update": 0.7,
            "nothing yet": 0.7,
            "been waiting": 0.75,
            "every time": 0.7,
            "over a week": 0.75,
            "for days": 0.7,
        },

        "urgent": {
            "urgent": 0.9,
            "immediately": 0.85,
            "asap": 0.85,
            "as soon as possible": 0.85,
            "right now": 0.8,
            "right away": 0.8,
            "emergency": 1.0,
            "critical": 0.85,
            "today itself": 0.8,
        },
    },

    "hindi": {

        "positive": {
            "धन्यवाद": 0.6,
            "शुक्रिया": 0.6,
            "बहुत अच्छा": 0.75,
            "शानदार": 0.8,
            "बढ़िया": 0.7,
            # अच्छा inflects for gender and number; all three forms
            # are listed so "सेवा अच्छी नहीं है" is scored before the
            # negation flips it.
            "अच्छा": 0.5,
            "अच्छी": 0.5,
            "अच्छे": 0.5,
            "खुश": 0.6,
            "ठीक हो गया": 0.7,
            "dhanyavad": 0.6,
            "shukriya": 0.6,
            "bahut accha": 0.75,
            "badhiya": 0.7,
            "shandar": 0.8,
            "accha": 0.5,
            "acchi": 0.5,
            "acche": 0.5,
            "khush": 0.6,
            "theek ho gaya": 0.7,
        },

        "negative": {
            "खराब": 0.7,
            "गलत": 0.55,
            "बहुत बुरा": 0.85,
            "परेशानी": 0.7,
            "शिकायत": 0.65,
            "नाराज़": 0.8,
            "देरी": 0.55,
            "गायब": 0.7,
            "नहीं मिला": 0.75,
            "kharab": 0.7,
            "galat": 0.55,
            "bahut bura": 0.85,
            "pareshani": 0.7,
            "shikayat": 0.65,
            "naraz": 0.8,
            "निराश": 0.7,
            "परेशान": 0.7,
            "बेकार": 0.75,
            "niraash": 0.7,
            "pareshan": 0.7,
            "bekaar": 0.75,
            "deri": 0.55,
            "gayab": 0.7,
            "nahi mila": 0.75,
        },

        "frustrated": {
            "अभी तक नहीं": 0.8,
            "कितनी बार": 0.85,
            "फिर से": 0.7,
            "बार बार": 0.8,
            "तीसरी बार": 0.85,
            "abhi tak nahi": 0.8,
            "kitni baar": 0.85,
            "phir se": 0.7,
            "baar baar": 0.8,
            "teesri baar": 0.85,
        },

        "urgent": {
            "तुरंत": 0.9,
            "जल्दी": 0.75,
            "अभी": 0.7,
            "जरूरी": 0.7,
            "इमरजेंसी": 1.0,
            "turant": 0.9,
            "jaldi": 0.75,
            "abhi": 0.7,
            "zaroori": 0.7,
            "emergency": 1.0,
        },
    },

    "telugu": {

        "positive": {
            "ధన్యవాదాలు": 0.6,
            "చాలా బాగుంది": 0.8,
            "బాగుంది": 0.65,
            "సూపర్": 0.7,
            "అద్భుతం": 0.8,
            "సంతోషం": 0.6,
            "బాగా": 0.5,
            "dhanyavadalu": 0.6,
            "chala bagundi": 0.8,
            "bagundi": 0.65,
            "super": 0.7,
            "adbhutam": 0.8,
            "santosham": 0.6,
            "baga": 0.5,
        },

        "negative": {
            "చెడు": 0.7,
            "తప్పు": 0.55,
            "చాలా చెడు": 0.85,
            "సమస్య": 0.6,
            "ఫిర్యాదు": 0.65,
            "కోపం": 0.8,
            "రాలేదు": 0.75,
            "ఆలస్యం": 0.6,
            "chedu": 0.7,
            "tappu": 0.55,
            "chala chedu": 0.85,
            "samasya": 0.6,
            "firyadu": 0.65,
            "kopam": 0.8,
            "నిరాశ": 0.7,
            "బాధ": 0.7,
            "niras": 0.7,
            "badha": 0.7,
            "raledu": 0.75,
            "kaledu": 0.75,
            "aalasyam": 0.6,
        },

        "frustrated": {
            "ఇంకా రాలేదు": 0.8,
            "ఎన్నిసార్లు": 0.85,
            "మళ్ళీ మళ్ళీ": 0.85,
            "మళ్ళీ": 0.7,
            "మూడోసారి": 0.85,
            "inka raledu": 0.8,
            "ennisarlu": 0.85,
            "malli malli": 0.85,
            "malli": 0.7,
            "moodosari": 0.85,
        },

        "urgent": {
            "వెంటనే": 0.9,
            "తొందరగా": 0.75,
            "ఇప్పుడే": 0.8,
            "అత్యవసరం": 1.0,
            "ventane": 0.9,
            "tondaraga": 0.75,
            "ippude": 0.8,
            "atyavasaram": 1.0,
        },
    },

    "spanish": {

        "positive": {
            "gracias": 0.6,
            "excelente": 0.8,
            "perfecto": 0.8,
            "resuelto": 0.7,
            "maravilloso": 0.75,
            "genial": 0.7,
            "bueno": 0.5,
            "contento": 0.6,
            "feliz": 0.6,
            "satisfecho": 0.7,
            "muy bien": 0.7,
        },

        "negative": {
            "malo": 0.7,
            "pésimo": 0.9,
            "terrible": 0.9,
            "horrible": 0.9,
            "enojado": 0.8,
            "inútil": 0.8,
            "roto": 0.7,
            "dañado": 0.75,
            "tarde": 0.5,
            "retraso": 0.55,
            "falta": 0.6,
            "queja": 0.65,
            "inaceptable": 0.85,
            "decepcionado": 0.7,
            "frustrante": 0.8,
            "frustrado": 0.8,
            "molesto": 0.7,
            "molesta": 0.7,
            "nunca recibí": 0.8,
            "no recibí": 0.75,
            "sin respuesta": 0.75,
        },

        "frustrated": {
            "otra vez": 0.8,
            "de nuevo": 0.75,
            "cuántas veces": 0.85,
            "cuantas veces": 0.85,
            "tercera vez": 0.85,
            "sigo esperando": 0.8,
            "todavía no": 0.8,
            "todavia no": 0.8,
        },

        "urgent": {
            "urgente": 0.9,
            "inmediatamente": 0.85,
            "ahora mismo": 0.85,
            "emergencia": 1.0,
            "cuanto antes": 0.8,
        },
    },
}


INTENSIFIERS: Dict[str, Tuple[str, ...]] = {
    "english": (
        "very", "extremely", "really", "so", "too",
        "absolutely", "totally", "completely",
    ),
    "hindi": (
        "बहुत", "बिल्कुल", "एकदम", "bahut", "bilkul", "ekdum",
    ),
    "telugu": (
        "చాలా", "భలే", "chala", "bhale",
    ),
    "spanish": (
        "muy", "demasiado", "extremadamente", "súper", "super",
    ),
}

NEGATIONS: Dict[str, Tuple[str, ...]] = {
    "english": (
        "not", "never", "no", "dont", "didnt", "doesnt", "isnt",
        "wasnt", "wont", "cannot", "cant", "nothing", "none",
    ),
    "hindi": (
        "नहीं", "नही", "मत", "nahi", "nahin", "mat",
    ),
    "telugu": (
        "లేదు", "కాదు", "వద్దు", "ledu", "kaadu", "vadu",
    ),
    "spanish": (
        "no", "nunca", "nada", "jamás", "jamas", "tampoco",
    ),
}


# A negated fulfilment or account verb is a complaint, not a neutral
# question: "my order has not arrived", "hasn't been delivered",
# "refund never credited". The lexicon above cannot cover these
# because the polarity lives in the negation, not in the verb --
# "arrived" on its own is a perfectly neutral word ("when will it
# arrive?"), so it must never become a positive marker. These are
# matched as whole patterns instead, which keeps questions neutral
# while catching the complaints.
NEGATED_COMPLAINT_PATTERNS: Dict[str, Tuple[Tuple[str, float], ...]] = {

    "english": (
        (
            r"(?:\b(?:not|never|no longer)\b|n't)\s+"
            r"(?:been\s+|being\s+|yet\s+|still\s+|even\s+)?"
            r"(?:\w+\s+){0,2}?"
            r"(?:arriv\w*|deliver\w*|receiv\w*|resolv\w*|refund\w*|"
            r"credit\w*|ship\w*|process\w*|confirm\w*|work\w*|"
            r"updat\w*|repl\w*|respond\w*|restor\w*|complet\w*|"
            r"dispatch\w*|fix\w*|here|got|get)\b",
            0.7,
        ),
    ),

    "hindi": (
        (
            r"(नहीं|नही|अभी तक नहीं)\s*(आया|आई|मिला|मिली|हुआ|हुई|"
            r"भेजा|डिलीवर|क्रेडिट|वापस)",
            0.7,
        ),
        (
            r"\b(nahi|nahin|abhi tak nahi)\s+"
            r"(aaya|aayi|mila|mili|hua|hui|bheja|deliver|credit|wapas)\b",
            0.7,
        ),
    ),

    # Telugu needs no entry here: "రాలేదు" / "కాలేదు" are inherently
    # negative completion words and already sit in the Telugu
    # negative lexicon (native and romanised), so the phrase pass
    # catches them directly.

    "spanish": (
        (
            r"\b(no|nunca|todavía no|todavia no)\s+"
            r"(?:ha\s+|han\s+|he\s+|se\s+ha\s+)?"
            r"(?:llegado|recibido|entregado|resuelto|reembolsado|"
            r"acreditado|enviado|procesado|funciona|actualizado|"
            r"respondido|llegó|llego)\b",
            0.7,
        ),
    ),
}


# Sarcasm requires a positive marker to CO-OCCUR with a mock or
# negative marker. A bare "great, thanks!" matches none of these.
#
# Each entry is (pattern, standalone_weight). The weight is the score
# the construction earns on its own; the positive-marker bonus in
# _sarcasm_score is added on top. Constructions that are sarcastic by
# themselves -- "thanks for nothing", "sure, because ... never" --
# carry a weight at or above the threshold, while those that only
# become sarcastic when paired with praise ("great ... still") stay
# below it so that sincere praise never trips them.
SARCASM_PATTERNS: Dict[str, Tuple[Tuple[str, float], ...]] = {

    "english": (
        (
            r"\b(great|wonderful|perfect|brilliant|awesome|fantastic|lovely)\b"
            r"[^.!?]{0,40}\b(again|another|still|yet again|third|once more)\b",
            0.45,
        ),
        (
            r"\b(thanks|thank you)\s+(a lot|so much)\s+for nothing\b",
            0.80,
        ),
        (r"\bthanks\s+for\s+nothing\b", 0.80),
        (
            r"\b(oh|wow)\b[,\s]+(great|perfect|wonderful|brilliant)\b",
            0.50,
        ),
        (
            r"\b(sure|of course|obviously)\b[,\s]+because\b[^.!?]{0,30}"
            r"\b(never|not|no)\b",
            0.70,
        ),
        # Mock dismissals and gratitude aimed at a complaint. "Yeah
        # right" mocks on its own; the others carry the complaint
        # inside the match, so sincere praise ("thanks so much for
        # resolving this") never trips them.
        (r"\byeah,? right\b", 0.80),
        (
            r"\b(thanks|thank you)\b[^.!?]{0,30}\bfor\b[^.!?]{0,40}"
            r"\b(ignor\w+|no response|no update|nothing|no help|"
            r"delay\w*|charged me twice|charging me twice)\b",
            0.80,
        ),
        (
            r"\b(fantastic|great|wonderful|brilliant|awesome|lovely|"
            r"perfect|amazing)\b[^.!?]{0,50}\b(charged|billed|debited|"
            r"deducted)\b[^.!?]{0,20}\b(twice|two times|2 times|"
            r"double)\b",
            0.75,
        ),
        (
            r"\b(because|after)\b[^.!?]{0,50}\bworked\s+so\s+well\b",
            0.75,
        ),
        (
            r"\bworked\s+so\s+well\b[^.!?]{0,25}\blast\s+time\b",
            0.75,
        ),
    ),

    "hindi": (
        (
            r"(वाह|बहुत)\s*(बढ़िया|शानदार|अच्छा)[^।.!?]{0,30}"
            r"(फिर|अभी तक|नहीं)",
            0.45,
        ),
        (
            r"\b(wah|bahut)\s+(badhiya|shandar|accha)\b[^.!?]{0,30}"
            r"\b(phir|abhi tak|nahi|nahin)\b",
            0.45,
        ),
    ),

    "telugu": (
        (
            r"(వావ్|చాలా)\s*(బాగుంది|సూపర్)[^.!?]{0,30}"
            r"(మళ్ళీ|ఇంకా|లేదు)",
            0.45,
        ),
        (
            r"\b(chala|wow)\s+(bagundi|super)\b[^.!?]{0,30}"
            r"\b(malli|inka|ledu|kaadu)\b",
            0.45,
        ),
    ),

    "spanish": (
        (
            r"\b(qué|que)\s+(bueno|maravilla|genial|perfecto)\b[^.!?]{0,30}"
            r"\b(otra vez|de nuevo|nunca|aún|aun|todavía|todavia)\b",
            0.45,
        ),
        (r"\bgracias\s+por\s+nada\b", 0.80),
    ),
}


# The polarity mismatch: exclamation-grade praise sharing a sentence
# with a complaint. Only the mock-cheer words are praise markers --
# never the polite-request grade ("appreciate", "thank you"), so a
# calm "I would appreciate it if you could look into a duplicate
# charge" is never mistaken for sarcasm.
SARCASM_PRAISE_MARKERS: Tuple[str, ...] = (
    r"\bwow\b",
    r"\bgreat\b",
    r"\bfantastic\b",
    r"\bwonderful\b",
    r"\bbrilliant\b",
    r"\bawesome\b",
    r"\blovely\b",
    r"\bperfect\b",
    r"\bamazing\b",
)

# Complaint contexts that contradict the praise they share a
# sentence with.
SARCASM_COMPLAINT_CONTEXTS: Tuple[str, ...] = (
    r"\bignor\w+\b",
    r"\b(?:charged|billed|debited|deducted)\b[^.!?]{0,20}"
    r"\b(?:twice|two times|2 times|double)\b",
    r"\bworked\s+so\s+well\b",
    r"\bstill\s+(?:not|no|waiting)\b",
    r"\bno\s+response\b",
    r"\bbroken\b",
    r"\bmissing\b",
    r"\bdelay\w+\b",
    r"\blost\s+my\b",
)

# A contrast cancels the mismatch: "the delivery was delayed but
# your support was great" is a mixed review, not a mock cheer.
SARCASM_CONTRAST_MARKERS: Tuple[str, ...] = (
    r"\bbut\b",
    r"\bhowever\b",
    r"\bthough\b",
)


# High-risk topics. Matched against raw text, never against the
# sentiment label, so calm phrasing still escalates.
RISK_PATTERNS: Dict[str, Tuple[str, ...]] = {

    RiskTopic.ACCOUNT_COMPROMISE.value: (
        r"someone else (logged|signed) ?(in|into)",
        r"account (was |has been |got )?(hacked|compromised|taken over|stolen)",
        r"(hacked|compromised) (my |the )?account",
        r"unauthori[sz]ed (access|login|activity)",
        r"unrecogni[sz]ed (login|device|access|activity)",
        r"(login|sign[- ]?in).{0,30}(not me|unknown device|another device)",
        r"(someone|somebody).{0,20}(accessed|used|logged into).{0,15}my account",
        r"password (was )?(changed|reset) without",
        r"otp .{0,25}without my",
        # Semantic compromise, however calmly phrased: password or
        # account plus used-by / accessed-by / someone else /
        # unknown person / not-me, in any voice.
        r"(password|passwd|account)[^.]{0,50}\b(used|accessed|"
        r"viewed|tried)[^.]{0,25}\bby\b[^.]{0,25}\b(someone|"
        r"somebody|another|unknown|stranger)",
        r"(someone|somebody)[^.]{0,25}\b(else )?(used|uses|has used|"
        r"accessed|is using|been using|using|tried)[^.]{0,30}\b"
        r"(my |the )?(password|account|email)",
        r"someone (else )?(logged|signed) ?(in|into)",
        r"\b(i|we) (did ?n[o']?t|did not|never)\b[^.]{0,25}\b"
        r"(login|log in|sign in|signed in)\b",
        r"(login|logins|access|password|activity)[^.]{0,25}"
        r"(was )?not (me|mine)\b",
        r"(unknown|unrecogni[sz]ed) (person|device)[^.]{0,30}"
        r"(account|password|access)",
        r"अकाउंट.{0,12}(हैक|चोरी|हैक हो)",
        r"बिना (बताए|अनुमति|पूछे).{0,15}पासवर्ड",
        r"अनधिकृत (पहुंच|लॉगिन)",
        r"mera account hack",
        r"account hack ho gaya",
        r"bina permission password",
        r"(किसी ने|किसीने).{0,30}(पासवर्ड|खाता|अकाउंट|password|account)",
        r"(मेरा|मेरी).{0,15}(खाता|अकाउंट|पासवर्ड).{0,30}(हैक|चोरी|किसी ने)",
        r"(मेरा|मेरी) पासवर्ड.{0,30}(इस्तेमाल|उपयोग)",
        r"kisi ne.{0,30}(mera|meri|password|account|khata)",
        r"అకౌంట్.{0,12}హ్యాక్",
        r"అనుమతి లేకుండా.{0,15}పాస్వర్డ్",
        r"ఖాతా.{0,20}హ్యాక్",
        r"హ్యాక్.{0,20}ఖాతా",
        r"ఎవరో.{0,30}(లాగిన్|ప్రవేశించ|వాడి|ఉపయోగించ)",
        r"(ఖాతా|పాస్వర్డ్).{0,30}(ఎవరో|వాడినారు|ఉపయోగించారు)",
        r"naa (khata|password|account).{0,25}(hack|vadaru|vadina|use)",
        r"(evado|evaro|evadu).{0,30}(naa )?(account|password|khata|login)",
        r"naa account hack",
        r"cuenta.{0,18}(hackeada|comprometida|robada)",
        r"acceso no autorizado",
        r"alguien (entró|entro|accedió|accedio) (en )?mi cuenta",
    ),

    RiskTopic.DUPLICATE_PAYMENT.value: (
        r"(charged|billed|debited|deducted|paid) (me )?(twice|two times|2 times|double)",
        r"double (charge|charged|payment|billing)",
        r"duplicate (payment|charge|transaction|billing)",
        r"two (identical )?(transactions|charges|payments)",
        r"दो बार.{0,15}(कट|भुगतान|पैसे|चार्ज)",
        r"डुप्लीकेट (पेमेंट|भुगतान|चार्ज)",
        r"do baar.{0,15}(kat|pais|payment|charge)",
        r"duplicate payment",
        r"రెండు సార్లు.{0,15}(కట్|చెల్లించ|డబ్బు|చార్జ్)",
        r"rendu sarlu.{0,15}(cut|pay|dabbu|charge)",
        r"(cobraron|cargaron|pagaron).{0,15}dos veces",
        r"pago duplicado",
        r"(cobro|doble cargo|cargo doble)",
        r"me cobraron dos veces",
    ),

    RiskTopic.LEGAL_THREAT.value: (
        r"\b(lawyer|attorney|solicitor)\b",
        r"\blegal (action|proceedings|notice|case)\b",
        r"\bconsumer (court|forum|commission)\b",
        r"\bsue (you|them|the company)\b",
        r"\bfile a (case|complaint|lawsuit)\b",
        r"\b(police complaint|ombudsman|small claims)\b",
        r"वकील",
        r"कोर्ट",
        r"कानूनी कार्रवाई",
        r"पुलिस.{0,10}शिकायत",
        r"मुकदमा",
        r"\b(vakil|kanooni|muqadma)\b",
        r"న్యాయవాది",
        r"కోర్టు",
        r"పోలీస్.{0,10}ఫిర్యాదు",
        r"\b(abogado|demanda|denuncia|tribunal|juzgado)\b",
        r"acción legal",
        r"accion legal",
    ),

    RiskTopic.UNAUTHORIZED_CHARGE.value: (
        r"(did ?n[o']?t|never) (make|authori[sz]e|approve) (this|the|that) "
        r"(charge|payment|transaction)",
        r"fraudulent (charge|transaction|payment)",
        r"unknown (charge|transaction|debit)",
        r"(charge|transaction) i (did ?n[o']?t|never) (make|authori[sz]e)",
        r"मैंने.{0,20}(नहीं किया|नहीं किए)",
        r"फर्जी (चार्ज|लेनदेन|भुगतान)",
        r"maine nahi kiya",
        r"fraudulent charge",
        r"నేను చేయలేదు",
        r"ఫ్రాడ్.{0,10}(చార్జ్|లావాదేవీ)",
        r"cargo (no autorizado|desconocido)",
        r"no (hice|autoricé|autorice) (este|el) cargo",
    ),

    RiskTopic.DATA_BREACH.value: (
        r"data (breach|leak)",
        r"personal (data|information|details).{0,25}"
        r"(leaked|exposed|shared|stolen)",
        r"my (data|details|information) (was|were|got) (leaked|exposed|stolen)",
        r"डेटा.{0,10}(लीक|चोरी|लिक)",
        r"data (leak|chori)",
        r"డేటా.{0,10}(లీక్|దొంగతనం)",
        r"filtración de datos",
        r"filtracion de datos",
        r"mis datos.{0,25}(filtr|expuest|rob)",
    ),
}


# Patterns that ask for a human, in any language.
HUMAN_REQUEST_PATTERNS: Tuple[str, ...] = (
    r"\b(talk|speak|chat) to (a )?(human|person|agent|representative|someone)",
    r"\b(real|actual|live) (person|agent|human)",
    r"\b(escalate|transfer) (me|this)",
    r"\b(supervisor|manager)\b",
    r"इंसान.{0,10}(से|को) (बात|बातचीत)",
    r"मैनेजर",
    r"మనిషి.{0,10}(తో|కి)",
    r"మేనేజర్",
    r"\b(hablar|comunicar) con (una |un )?(persona|agente|humano|representante)",
    r"\b(supervisor|gerente)\b",
)


# ============================================================
# TEXT HELPERS
# ============================================================

_WHITESPACE = re.compile(r"\s+")

_TOKEN_CLEAN = re.compile(r"[^\wऀ-ॿఀ-౿]+", re.UNICODE)


def _normalize(text: str) -> str:

    return _WHITESPACE.sub(" ", (text or "").strip().lower())


def _tokenize(text: str) -> List[str]:

    normalized = _normalize(text)

    tokens: List[str] = []

    for raw in normalized.split(" "):

        cleaned = _TOKEN_CLEAN.sub("", raw)

        if cleaned:

            tokens.append(cleaned)

    return tokens


def _lexicon_languages(language: str) -> List[str]:
    """
    Languages to score in.

    English is always included alongside the detected language,
    because loanwords and code-switching are common in the supported
    languages and a Hindi message often carries English nouns.
    """

    languages = ["english"]

    if language in SENTIMENT_LEXICON and language != "english":

        languages.append(language)

    return languages


# ============================================================
# SCORING
# ============================================================

# Inflectional endings that attach to a word without changing its
# sentiment. Telugu is agglutinative -- "నిరాశ" (disappointment)
# becomes "నిరాశగా" (disappointedly) -- and Hindi case markers attach
# the same way. Without these, a perfectly ordinary sentence scores
# as having no sentiment at all.
#
# Applied only to a single-token lookup that failed exactly, and only
# when the stem that remains is still substantial, so an ordinary
# word is never chopped into a false match. English and Spanish are
# inflected too, but with far more irregularity; their forms are
# listed explicitly in the lexicon instead.
INFLECTIONAL_SUFFIXES: Dict[str, Tuple[str, ...]] = {
    # "లు" is the Telugu plural marker. A bare "ల" is deliberately
    # absent: it is short enough to shear ordinary words into false
    # stems.
    "telugu": (
        "గా", "లో", "కి", "కు", "ను", "తో", "పై", "నుండి", "లు",
    ),
    "hindi": (
        "ों", "ें", "ियों", "ियां", "पन", "ता", "ती",
    ),
}

MIN_STEM_LENGTH = 2


def _strip_inflection(
    token: str,
    language: str,
) -> Optional[str]:
    """
    The stem of a single token, or None when no known ending applies.
    """

    for candidate_language in _lexicon_languages(language):

        for suffix in INFLECTIONAL_SUFFIXES.get(
            candidate_language, ()
        ):

            if not token.endswith(suffix):

                continue

            stem = token[: -len(suffix)]

            if len(stem) >= MIN_STEM_LENGTH:

                return stem

    return None


def _lookup_phrase(
    phrase: str,
    language: str,
    allow_inflection: bool = False,
) -> Optional[Tuple[str, float]]:
    """
    Resolve a phrase against the lexicons, English first.

    When ``allow_inflection`` is set -- which the caller does only for
    a single token -- an exact miss is retried once with a known
    inflectional ending removed.
    """

    for candidate_language in _lexicon_languages(language):

        lexicon = SENTIMENT_LEXICON.get(candidate_language, {})

        for label_name in (
            SentimentLabel.POSITIVE.value,
            SentimentLabel.NEGATIVE.value,
            SentimentLabel.FRUSTRATED.value,
            SentimentLabel.URGENT.value,
        ):

            weight = lexicon.get(label_name, {}).get(phrase)

            if weight is not None:

                return label_name, weight

    if not allow_inflection:

        return None

    stem = _strip_inflection(phrase, language)

    if stem is None or stem == phrase:

        return None

    return _lookup_phrase(stem, language, allow_inflection=False)


def _score_lexicon(
    text: str,
    language: str,
    config: SentimentConfig,
) -> Tuple[Dict[str, float], List[str]]:

    tokens = _tokenize(text)

    consumed = set()

    matches: List[Tuple[int, str, float, str]] = []

    # Phase 1: claim phrases, longest first, so "not received" wins
    # over "received". Every claim is recorded before any modifier is
    # evaluated -- otherwise the "not" inside "still not" would still
    # be free to negate an unrelated "how many times" matched in the
    # longer-phrase pass.
    for size in (3, 2, 1):

        for start in range(len(tokens) - size + 1):

            if any(
                index in consumed
                for index in range(start, start + size)
            ):

                continue

            phrase = " ".join(tokens[start:start + size])

            found = _lookup_phrase(
                phrase, language, allow_inflection=(size == 1)
            )

            if found is None:

                continue

            for index in range(start, start + size):

                consumed.add(index)

            matches.append((start, found[0], found[1], phrase, size))

    # Phase 2: apply intensifiers and negation against the final
    # consumption map.
    scores: Dict[str, float] = {
        SentimentLabel.POSITIVE.value: 0.0,
        SentimentLabel.NEGATIVE.value: 0.0,
        SentimentLabel.FRUSTRATED.value: 0.0,
        SentimentLabel.URGENT.value: 0.0,
    }

    evidence: List[str] = []

    for start, label_name, weight, phrase, size in matches:

        for back in (1, 2):

            position = start - back

            if position < 0:

                break

            if position in consumed:

                continue

            if _is_intensifier(tokens[position], language):

                weight *= config.intensifier_multiplier

                break

        negated = _is_negated(
            tokens, start, size, language, config, consumed
        )

        if negated:

            label_name = _flip(label_name)

        scores[label_name] += weight

        # "negated:" rather than "not " because the phrase may be in
        # Devanagari or Telugu script, where an English prefix reads
        # as a bug in the audit trail.
        evidence.append(
            phrase if not negated else f"negated: {phrase}"
        )

    # Phase 3: negated-fulfilment complaints, which have no lexicon
    # entry of their own. Scored in every applicable language because
    # a customer may write "order nahi aaya" alongside English.
    normalized = _normalize(text)

    for candidate_language in _lexicon_languages(language):

        for pattern, weight in NEGATED_COMPLAINT_PATTERNS.get(
            candidate_language, ()
        ):

            found = re.search(pattern, normalized)

            if found:

                scores[SentimentLabel.NEGATIVE.value] += weight

                evidence.append(found.group(0).strip())

    return scores, evidence


def _is_intensifier(token: str, language: str) -> bool:

    for candidate_language in _lexicon_languages(language):

        if token in INTENSIFIERS.get(candidate_language, ()):

            return True

    return False


# Languages that negate after the word they negate. Hindi and Telugu
# are verb-final, so "सेवा अच्छी नहीं है" ("service good not is") puts
# the नहीं after अच्छी. Scanning only backwards, as English requires,
# would read that sentence as praise. English and Spanish are
# negator-first, so a forward scan there would wrongly negate "good,
# not bad".
POST_POSITIONAL_NEGATION_LANGUAGES = ("hindi", "telugu")


def _is_negated(
    tokens: List[str],
    start: int,
    size: int,
    language: str,
    config: SentimentConfig,
    consumed: Optional[set] = None,
) -> bool:
    """
    Whether a negation applies to the phrase at [start, start + size).

    Scans backwards for every language, and forwards as well when the
    language negates post-positionally.

    Tokens already claimed by a longer phrase are skipped. Without
    that, the "नहीं" inside the frustrated phrase "अभी तक नहीं" would
    also negate a separate "खराब" later in the same sentence, and the
    "not" inside "still not" would negate an unrelated "how many
    times".
    """

    consumed = consumed or set()

    languages = _lexicon_languages(language)

    lower = max(0, start - config.negation_window)
    upper = min(len(tokens), start + size + config.negation_window)

    for index in range(lower, start):

        if index in consumed:

            continue

        token = tokens[index]

        for candidate_language in languages:

            if token in NEGATIONS.get(candidate_language, ()):

                return True

    if language not in POST_POSITIONAL_NEGATION_LANGUAGES:

        return False

    for index in range(start + size, upper):

        if index in consumed:

            continue

        token = tokens[index]

        for candidate_language in languages:

            if token in NEGATIONS.get(candidate_language, ()):

                return True

    return False


def _flip(label_name: str) -> str:

    if label_name == SentimentLabel.POSITIVE.value:

        return SentimentLabel.NEGATIVE.value

    if label_name == SentimentLabel.NEGATIVE.value:

        return SentimentLabel.POSITIVE.value

    return label_name


def _praise_complaint_mismatch(normalized: str) -> bool:
    """
    Whether exclamation-grade praise shares a sentence with a
    complaint -- the signature of sarcasm the construction patterns
    cannot see, because the praise words outscore the complaint in
    the lexicon and the message reads as satisfied praise.

    A contrastive "but" cancels the mismatch: a mixed review is not
    a mock cheer.
    """

    for sentence in re.split(r"[.!?]+", normalized):

        if any(
            re.search(marker, sentence)
            for marker in SARCASM_CONTRAST_MARKERS
        ):

            continue

        if not any(
            re.search(marker, sentence)
            for marker in SARCASM_PRAISE_MARKERS
        ):

            continue

        if any(
            re.search(context, sentence)
            for context in SARCASM_COMPLAINT_CONTEXTS
        ):

            return True

    return False


def _sarcasm_score(
    text: str,
    language: str,
    positive_score: float,
    config: SentimentConfig,
) -> Tuple[bool, float, Optional[str]]:

    normalized = _normalize(text)

    matched_pattern: Optional[str] = None
    pattern_weight = config.sarcasm_base_score

    for candidate_language in _lexicon_languages(language):

        for pattern, weight in SARCASM_PATTERNS.get(
            candidate_language, ()
        ):

            if re.search(pattern, normalized):

                matched_pattern = pattern
                pattern_weight = weight

                break

        if matched_pattern:

            break

    if matched_pattern is None:

        # No known construction -- but praise and a complaint may
        # still share a sentence ("Fantastic, you charged me
        # twice."), with the praise outscoring the complaint so
        # heavily that only this polarity mismatch reveals the
        # sarcasm.
        if not _praise_complaint_mismatch(normalized):

            return False, 0.0, None

        matched_pattern = "praise attached to a complaint"

        pattern_weight = config.sarcasm_mismatch_weight

    # The construction's own weight, plus a bonus for real praise in
    # the same message. "Great, still not refunded" clears the
    # threshold on the two together; "Great, thank you!" matches no
    # pattern at all and never reaches here.
    score = pattern_weight

    score += config.sarcasm_positive_weight * min(
        1.0, positive_score
    )

    # Explicit gratitude toward a real outcome is usually sincere.
    if re.search(r"\b(thanks|thank you|gracias|dhanyavad)\b", normalized):

        if positive_score >= 1.0 and not re.search(
            r"nothing|nada|nahi|ledu", normalized
        ):

            score -= 0.1

    score = max(0.0, min(1.0, score))

    return score >= config.sarcasm_min_score, score, matched_pattern


def _polarity(scores: Dict[str, float]) -> float:

    positive = scores.get(SentimentLabel.POSITIVE.value, 0.0)

    negative = (
        scores.get(SentimentLabel.NEGATIVE.value, 0.0)
        + scores.get(SentimentLabel.FRUSTRATED.value, 0.0)
    )

    total = positive + negative

    if total <= 0:

        return 0.0

    return max(-1.0, min(1.0, (positive - negative) / total))


# ============================================================
# HIGH RISK DETECTION (INDEPENDENT OF SENTIMENT)
# ============================================================

def detect_high_risk_topics(
    text: str,
    config: Optional[SentimentConfig] = None,
) -> List[Dict]:
    """
    Detect high-risk topics in raw text.

    This never reads the sentiment label or polarity. That is the whole
    point: a polite, calm complaint about a duplicate charge must still
    escalate, and it would be missed if risk were inferred from tone.
    """

    config = config or SentimentConfig()

    normalized = _normalize(text)

    matches: List[Dict] = []

    for topic in config.risk_topics_enabled:

        for pattern in RISK_PATTERNS.get(topic, ()):

            found = re.search(pattern, normalized)

            if found:

                matches.append(
                    {
                        "topic": topic,
                        "severity": RISK_SEVERITY.get(topic, "medium"),
                        "matched": found.group(0),
                        "language": detect_language(
                            found.group(0), config.language_config
                        )["language"],
                    }
                )

                break

    return matches


def _risk_severity(matches: Sequence[Dict]) -> str:

    severity = "none"

    for match in matches:

        candidate = match.get("severity", "none")

        if SEVERITY_ORDER.get(candidate, 0) > SEVERITY_ORDER[severity]:

            severity = candidate

    return severity


# ============================================================
# MESSAGE ANALYSIS
# ============================================================

def _finalize(
    scores: Dict[str, float],
    config: SentimentConfig,
) -> Tuple[str, float, Dict[str, float]]:

    total = sum(scores.values())

    if total <= 0:

        return (
            SentimentLabel.NEUTRAL.value,
            config.neutral_default_confidence,
            dict(scores),
        )

    best_label = LABEL_TIE_ORDER[-1].value
    best_score = -1.0

    for candidate in LABEL_TIE_ORDER:

        value = scores.get(candidate.value, 0.0)

        if value > best_score:

            best_score = value
            best_label = candidate.value

    share = best_score / total

    confidence = share * min(
        1.0, total / config.evidence_saturation
    )

    return best_label, round(confidence, 4), dict(scores)


def _apply_llm_fallback(
    text: str,
    language: str,
    config: SentimentConfig,
    result: SentimentResult,
) -> SentimentResult:
    """
    The optional second opinion.

    Invoked only when a hook was injected AND the rule layer's own
    confidence sits below the configured threshold. With the default
    None hook this never runs, so the module stays deterministic and
    offline; a hook that raises or returns nothing degrades to the
    rule-based result rather than taking message classification down
    with it.
    """

    hook = config.llm_fallback

    if hook is None or result.confidence >= config.minimum_confidence:

        return result

    try:

        second_opinion = hook(text, language, result)

    except Exception:

        return result

    if not isinstance(second_opinion, SentimentResult):

        return result

    if not any(
        "llm fallback" in item for item in second_opinion.evidence
    ):

        second_opinion.evidence.append(
            "llm fallback: rule confidence below "
            f"{config.minimum_confidence}"
        )

    return second_opinion


def analyse_message(
    text: str,
    language: Optional[str] = None,
    config: Optional[SentimentConfig] = None,
    now: Optional[datetime] = None,
) -> SentimentResult:
    """
    Classify one message.

    ``now`` is accepted for interface consistency with the rest of the
    support-ops modules; classification itself is time-independent.

    When ``config.llm_fallback`` is set and the rule layer's own
    confidence falls below ``config.minimum_confidence``, that hook
    is consulted for a second opinion (see ``_apply_llm_fallback``);
    with the default None hook the classification is fully
    deterministic and offline.
    """

    config = config or SentimentConfig()

    if language is None:

        language = detect_language(
            text or "", config.language_config
        )["language"]

    scores, evidence = _score_lexicon(
        text or "", language, config
    )

    label, confidence, label_scores = _finalize(scores, config)

    positive_score = scores.get(
        SentimentLabel.POSITIVE.value, 0.0
    )

    sarcastic, sarcasm_score, sarcasm_pattern = _sarcasm_score(
        text or "", language, positive_score, config
    )

    underlying = label

    polarity = _polarity(scores)

    if sarcastic:

        label = SentimentLabel.SARCASTIC.value

        # A sarcastic complaint is negative in intent.
        polarity = -max(0.35, abs(polarity))

        # The sarcasm evidence is real even when the praise words
        # dominated the underlying score so heavily that the
        # share-of-evidence confidence is low: a sarcastic reading
        # is never reported at near-zero confidence.
        confidence = round(max(confidence, sarcasm_score), 4)

        label_scores[SentimentLabel.SARCASTIC.value] = (
            label_scores.get(SentimentLabel.SARCASTIC.value, 0.0)
            + sarcasm_score
        )

        if sarcasm_pattern:

            evidence.append(f"sarcasm: {underlying}")

    risk_matches = detect_high_risk_topics(text or "", config)

    urgency_score = min(
        1.0,
        scores.get(SentimentLabel.URGENT.value, 0.0) / 0.9,
    )

    result = SentimentResult(
        label=label,
        underlying_label=underlying,
        confidence=confidence,
        polarity=round(polarity, 4),
        intensity=round(
            min(1.0, sum(scores.values()) / config.evidence_saturation),
            4,
        ),
        sarcasm=sarcastic,
        sarcasm_score=round(sarcasm_score, 4),
        urgency_score=round(urgency_score, 4),
        label_scores=label_scores,
        risk_topics=[match["topic"] for match in risk_matches],
        risk_severity=_risk_severity(risk_matches),
        language=language,
        evidence=evidence,
        repeated_negative=False,
        message_count=1,
    )

    # The optional injected second opinion, only for low-confidence
    # results; with the default None hook this is a no-op.
    return _apply_llm_fallback(
        text or "", language, config, result
    )


# ============================================================
# HISTORY BLEND
# ============================================================

def combine_with_history(
    current: SentimentResult,
    history: Sequence[SentimentResult],
    config: Optional[SentimentConfig] = None,
) -> SentimentResult:
    """
    Blend the current message with recent history.

    Repeated negativity is detected here and promotes a merely
    negative message to frustrated, unless the current message is
    urgent or carries a risk topic -- those already outrank it.
    """

    config = config or SentimentConfig()

    labels = [
        SentimentLabel.POSITIVE.value,
        SentimentLabel.NEGATIVE.value,
        SentimentLabel.FRUSTRATED.value,
        SentimentLabel.URGENT.value,
        SentimentLabel.SARCASTIC.value,
    ]

    blended = {
        label: config.current_weight
        * current.label_scores.get(label, 0.0)
        for label in labels
    }

    window = list(history)[-config.history_window:]

    if window:

        for label in labels:

            mean = sum(
                item.label_scores.get(label, 0.0)
                for item in window
            ) / len(window)

            blended[label] += config.history_weight * mean

    # The window counts the current message too -- "repeated" means the
    # customer has now complained more than once, and the current turn
    # is one of those complaints.
    recent = ([current] + list(history))[
        -config.repeated_negative_window:
    ]

    recent_negative = sum(
        1
        for item in recent
        if item.label in NEGATIVE_LABELS or item.polarity <= -0.3
    )

    repeated = recent_negative >= config.repeated_negative_count

    label, confidence, label_scores = _finalize(blended, config)

    if (
        repeated
        and current.risk_severity in ("none", "low")
        and label in (
            SentimentLabel.NEGATIVE.value,
            SentimentLabel.POSITIVE.value,
            SentimentLabel.NEUTRAL.value,
        )
    ):

        label = SentimentLabel.FRUSTRATED.value

    polarity = _polarity(blended)

    if current.sarcasm:

        label = SentimentLabel.SARCASTIC.value

        polarity = min(polarity, -0.35)

    risk_topics = list(dict.fromkeys(
        list(current.risk_topics)
        + [topic for item in window for topic in item.risk_topics]
    ))

    severity = "none"

    for item in [current] + window:

        if SEVERITY_ORDER.get(item.risk_severity, 0) > SEVERITY_ORDER[severity]:

            severity = item.risk_severity

    return SentimentResult(
        label=label,
        underlying_label=current.underlying_label,
        confidence=confidence,
        polarity=round(polarity, 4),
        intensity=current.intensity,
        sarcasm=current.sarcasm,
        sarcasm_score=current.sarcasm_score,
        urgency_score=max(
            [current.urgency_score]
            + [item.urgency_score for item in window]
        ),
        label_scores=label_scores,
        risk_topics=risk_topics,
        risk_severity=severity,
        language=current.language,
        evidence=list(current.evidence),
        repeated_negative=repeated,
        message_count=1 + len(window),
    )


def analyse_conversation(
    messages: Sequence[ConversationMessage],
    config: Optional[SentimentConfig] = None,
    now: Optional[datetime] = None,
) -> SentimentResult:
    """
    Classify the latest customer message in the context of the
    conversation so far.
    """

    config = config or SentimentConfig()

    user_messages = [
        message for message in messages if message.role == "user"
    ]

    if not user_messages:

        return SentimentResult(
            language="unknown",
            confidence=config.neutral_default_confidence,
        )

    results = [
        analyse_message(
            message.content,
            message.language if message.language != "unknown" else None,
            config,
            message.timestamp,
        )
        for message in user_messages
    ]

    current = results[-1]

    return combine_with_history(current, results[:-1], config)


# ============================================================
# TONE ADJUSTMENT (NEVER CHANGES POLICY)
# ============================================================

TONE_BY_LABEL = {
    SentimentLabel.POSITIVE.value: (
        "The customer is satisfied. Keep the reply warm, brief and "
        "confirmatory."
    ),
    SentimentLabel.NEUTRAL.value: (
        "Keep the reply clear, plain and factual."
    ),
    SentimentLabel.NEGATIVE.value: (
        "Acknowledge the customer's difficulty with empathy and "
        "apologise once for the inconvenience."
    ),
    SentimentLabel.FRUSTRATED.value: (
        "The customer has contacted us repeatedly. Acknowledge that "
        "directly, avoid excuses, and focus on the next concrete step."
    ),
    SentimentLabel.URGENT.value: (
        "Lead with the immediate next step and any time-critical "
        "information. Keep everything else to a minimum."
    ),
    SentimentLabel.SARCASTIC.value: (
        "Stay professional and courteous. Never mirror sarcasm. "
        "Address the underlying problem directly and without "
        "defensiveness."
    ),
}

POLICY_GUARD = (
    "Adjust wording and empathy only. Do NOT change any policy, "
    "eligibility, timeline, amount, or outcome. Do not promise "
    "anything beyond the knowledge base."
)


def build_tone_instruction(
    sentiment: SentimentResult,
    language: str = "english",
) -> str:
    """
    Build a style-only instruction block for the LLM.

    This deliberately contains no policy content: it can change how an
    answer is phrased but not what the business allows.
    """

    from task6_multilingual import LANGUAGE_NAMES

    style = TONE_BY_LABEL.get(
        sentiment.label,
        TONE_BY_LABEL[SentimentLabel.NEUTRAL.value],
    )

    display_name = LANGUAGE_NAMES.get(language, "the customer's language")

    return (
        f"Communication style: {style} "
        f"Respond in {display_name}. "
        f"Rules: {POLICY_GUARD}"
    )


def requests_human(text: str) -> bool:
    """
    Whether the customer explicitly asked for a human.
    """

    normalized = _normalize(text)

    return any(
        re.search(pattern, normalized)
        for pattern in HUMAN_REQUEST_PATTERNS
    )
