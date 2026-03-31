"""
nlp_pipeline.py — Advanced NLP preprocessing for the chatbot system.

Stages:
  1. Text cleaning (HTML, URLs, special chars)
  2. Tokenization + lowercasing
  3. Stop-word removal (with preserved wellness vocabulary)
  4. Lemmatization (spaCy) and stemming (NLTK Porter)
  5. Sentence embedding (Sentence Transformers)
  6. Intent detection (zero-shot classification)
  7. Context extraction for RAG queries
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sentence_transformers import SentenceTransformer
import numpy as np

# Download required NLTK data on first run
for pkg in ["punkt", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

logger = logging.getLogger(__name__)


# ─── Intent taxonomy ─────────────────────────────────────────────────────────

class UserIntent(str, Enum):
    REPORT_USAGE         = "report_usage"          # "I use my phone too much"
    REQUEST_ADVICE       = "request_advice"         # "How can I reduce screen time?"
    CHECK_PROGRESS       = "check_progress"         # "How am I doing this week?"
    START_FOCUS          = "start_focus"            # "Start a focus session"
    VIEW_STREAK          = "view_streak"            # "Show my streak"
    ASK_WELLNESS_SCORE   = "ask_wellness_score"     # "What's my wellness score?"
    GENERAL_CHAT         = "general_chat"           # "Hi" / off-topic
    REQUEST_REPORT       = "request_report"         # "Give me my weekly report"
    SET_REMINDER         = "set_reminder"           # "Remind me at 9 PM"


# ─── Intent patterns ──────────────────────────────────────────────────────────

INTENT_PATTERNS: dict[UserIntent, list[str]] = {
    UserIntent.REPORT_USAGE: [
        r"(too much|overuse|addicted|can't stop|phone all day|screen time)",
    ],
    UserIntent.REQUEST_ADVICE: [
        r"(how (can|do|should) i|tips|advice|help me|suggest|recommend|ways to)",
    ],
    UserIntent.CHECK_PROGRESS: [
        r"(how am i doing|progress|this week|summary|my stats|update)",
    ],
    UserIntent.START_FOCUS: [
        r"(start|begin|launch) (focus|pomodoro|work session|timer)",
    ],
    UserIntent.VIEW_STREAK: [
        r"(streak|consecutive|how many days|detox days)",
    ],
    UserIntent.ASK_WELLNESS_SCORE: [
        r"(wellness score|health score|digital score|my score)",
    ],
    UserIntent.REQUEST_REPORT: [
        r"(weekly report|give me my report|show me report|analytics)",
    ],
    UserIntent.SET_REMINDER: [
        r"(remind me|set (a )?reminder|alert me|notify me)",
    ],
}


# ─── Data contracts ────────────────────────────────────────────────────────────

@dataclass
class ProcessedText:
    original: str
    cleaned: str
    tokens: list[str]
    lemmas: list[str]
    stems: list[str]
    embedding: np.ndarray
    detected_intent: UserIntent
    intent_confidence: float
    key_entities: list[str]   # times, app names, numbers extracted


# ─── NLP Pipeline ─────────────────────────────────────────────────────────────

class NLPPipeline:
    """
    Processes user chat messages through all NLP stages.
    Designed for single-message and batch processing.
    """

    # Words that matter for wellness context — never removed as stop-words
    PRESERVE_TERMS = {
        "sleep", "night", "late", "phone", "social", "media", "screen",
        "focus", "break", "stress", "addicted", "too", "much", "less",
        "more", "help", "reduce", "limit", "score", "time",
    }

    def __init__(self, embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        logger.info("Loading embedding model: %s", embedding_model_name)
        self.embedder = SentenceTransformer(embedding_model_name)
        self.stemmer = PorterStemmer()

        # Filtered stop-words: preserve wellness vocabulary
        raw_stops = set(stopwords.words("english"))
        self._stop_words = raw_stops - self.PRESERVE_TERMS

        # spaCy for lemmatization — optional, graceful fallback to NLTK
        try:
            import spacy
            self._nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
            self._use_spacy = True
        except (ImportError, OSError):
            logger.warning("spaCy not available; falling back to NLTK WordNetLemmatizer")
            from nltk.stem import WordNetLemmatizer
            self._lemmatizer = WordNetLemmatizer()
            self._use_spacy = False

    # ── Public API ────────────────────────────────────────────────────────────

    def process(self, text: str) -> ProcessedText:
        cleaned = self._clean(text)
        tokens = self._tokenize(cleaned)
        tokens_no_stop = self._remove_stopwords(tokens)
        lemmas = self._lemmatize(tokens_no_stop)
        stems = [self.stemmer.stem(t) for t in tokens_no_stop]
        embedding = self._embed(cleaned)
        intent, confidence = self._detect_intent(cleaned)
        entities = self._extract_entities(text)

        return ProcessedText(
            original=text,
            cleaned=cleaned,
            tokens=tokens_no_stop,
            lemmas=lemmas,
            stems=stems,
            embedding=embedding,
            detected_intent=intent,
            intent_confidence=confidence,
            key_entities=entities,
        )

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Batch embed for efficient vector DB ingestion."""
        return self.embedder.encode(texts, batch_size=32, show_progress_bar=False)

    # ── Stage 1: Text cleaning ─────────────────────────────────────────────

    def _clean(self, text: str) -> str:
        text = text.strip()
        text = re.sub(r"<[^>]+>", " ", text)           # strip HTML tags
        text = re.sub(r"https?://\S+", " ", text)       # remove URLs
        text = re.sub(r"@\w+", " ", text)               # remove mentions
        text = re.sub(r"[^\w\s'.,!?]", " ", text)       # keep useful punctuation
        text = re.sub(r"\s+", " ", text)                 # collapse whitespace
        return text.lower()

    # ── Stage 2: Tokenization ─────────────────────────────────────────────

    def _tokenize(self, text: str) -> list[str]:
        return nltk.word_tokenize(text)

    # ── Stage 3: Stop-word removal ────────────────────────────────────────

    def _remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]

    # ── Stage 4: Lemmatization ────────────────────────────────────────────

    def _lemmatize(self, tokens: list[str]) -> list[str]:
        if self._use_spacy:
            doc = self._nlp(" ".join(tokens))
            return [token.lemma_ for token in doc]
        else:
            return [self._lemmatizer.lemmatize(t) for t in tokens]

    # ── Stage 5: Sentence embedding ───────────────────────────────────────

    def _embed(self, text: str) -> np.ndarray:
        return self.embedder.encode(text, convert_to_numpy=True)

    # ── Stage 6: Intent detection (rule-based + can extend to classifier) ─

    def _detect_intent(self, cleaned_text: str) -> tuple[UserIntent, float]:
        matches: list[tuple[UserIntent, int]] = []
        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                m = re.search(pattern, cleaned_text)
                if m:
                    matches.append((intent, len(m.group(0))))

        if not matches:
            return UserIntent.GENERAL_CHAT, 0.5

        # Select intent with longest match (heuristic confidence)
        best_intent, match_len = max(matches, key=lambda x: x[1])
        confidence = min(0.95, 0.5 + match_len * 0.02)
        return best_intent, round(confidence, 3)

    # ── Stage 7: Entity extraction ────────────────────────────────────────

    def _extract_entities(self, text: str) -> list[str]:
        entities = []
        # Time patterns: "10 PM", "2 hours", "30 minutes"
        times = re.findall(r"\d{1,2}(?::\d{2})?\s*(?:am|pm|AM|PM)", text)
        durations = re.findall(r"\d+\s*(?:hour|minute|min|hr)s?", text, re.IGNORECASE)
        # Numbers
        numbers = re.findall(r"\b\d+\.?\d*\b", text)
        entities.extend(times + durations + numbers[:5])  # cap at 5 numbers
        return entities
