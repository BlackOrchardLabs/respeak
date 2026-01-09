"""
Emotional Analyzer
==================

Extracts emotional signals from transcript text.
Designed for <10ms latency on GPU.
"""

import re
from dataclasses import dataclass, field
from typing import Optional, List
from collections import deque


@dataclass
class EmotionalSignals:
    """Emotional features extracted from transcript."""

    # Core affect (Russell circumplex)
    valence: float = 0.0      # -1.0 (negative) to 1.0 (positive)
    arousal: float = 0.5      # 0.0 (calm) to 1.0 (excited)
    dominance: float = 0.5    # 0.0 (submissive) to 1.0 (dominant)

    # Linguistic markers
    is_question: bool = False
    is_exclamation: bool = False
    hesitation_count: int = 0

    # Contextual
    intimacy_level: float = 0.0
    topic_shift: bool = False

    # Raw
    keywords_detected: List[str] = field(default_factory=list)


class EmotionalAnalyzer:
    """
    Lightweight emotional analysis for real-time use.

    Uses heuristics + optional ML model for speed.
    Target: <10ms per chunk.
    """

    # Sentiment word lists (fast lookup)
    POSITIVE_WORDS = {
        "love", "great", "amazing", "wonderful", "happy", "excited",
        "beautiful", "perfect", "awesome", "fantastic", "good", "nice",
        "yes", "sure", "absolutely", "definitely", "thanks", "thank",
        "please", "appreciate", "enjoy", "glad", "delighted"
    }

    NEGATIVE_WORDS = {
        "hate", "terrible", "awful", "bad", "sad", "angry", "upset",
        "frustrated", "annoyed", "disappointed", "wrong", "no", "not",
        "never", "can't", "won't", "don't", "shouldn't", "horrible",
        "sorry", "unfortunately", "problem", "issue", "difficult"
    }

    AROUSAL_HIGH = {
        "excited", "amazing", "incredible", "wow", "oh", "yes", "no",
        "what", "really", "seriously", "absolutely", "definitely",
        "urgent", "now", "immediately", "quick", "fast", "hurry"
    }

    AROUSAL_LOW = {
        "calm", "quiet", "peaceful", "gentle", "soft", "slow",
        "relax", "rest", "sleep", "tired", "maybe", "perhaps",
        "sometime", "whenever", "eventually"
    }

    INTIMACY_MARKERS = {
        "love", "baby", "honey", "sweetheart", "darling", "babe",
        "miss you", "need you", "want you", "feel", "heart",
        "together", "us", "we", "our", "close", "hold", "touch"
    }

    HESITATION_MARKERS = {"um", "uh", "er", "ah", "like", "you know", "i mean"}

    def __init__(
        self,
        use_ml_model: bool = False,
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
        context_window: int = 5
    ):
        self.use_ml = use_ml_model
        self.context_window = context_window
        self.context = deque(maxlen=context_window)

        self._ml_pipeline = None
        if use_ml_model:
            self._init_ml_model(model_name)

    def _init_ml_model(self, model_name: str):
        """Initialize ML sentiment model."""
        try:
            from transformers import pipeline
            self._ml_pipeline = pipeline(
                "sentiment-analysis",
                model=model_name,
                device=0  # GPU
            )
        except Exception as e:
            print(f"ML model init failed, using heuristics: {e}")
            self.use_ml = False

    def analyze(self, text: str) -> EmotionalSignals:
        """
        Analyze text and return emotional signals.

        Args:
            text: Transcript text to analyze

        Returns:
            EmotionalSignals with detected features
        """
        # Add to context
        self.context.append(text.lower())
        context_text = " ".join(self.context)

        signals = EmotionalSignals()

        # Tokenize
        words = set(re.findall(r'\b\w+\b', text.lower()))
        context_words = set(re.findall(r'\b\w+\b', context_text))

        # === Valence ===
        if self.use_ml and self._ml_pipeline:
            signals.valence = self._ml_valence(text)
        else:
            signals.valence = self._heuristic_valence(words)

        # === Arousal ===
        signals.arousal = self._compute_arousal(text, words)

        # === Linguistic markers ===
        signals.is_question = text.strip().endswith("?")
        signals.is_exclamation = text.strip().endswith("!")
        signals.hesitation_count = sum(
            1 for h in self.HESITATION_MARKERS if h in text.lower()
        )

        # === Intimacy ===
        intimacy_hits = sum(1 for m in self.INTIMACY_MARKERS if m in context_text)
        signals.intimacy_level = min(intimacy_hits / 5, 1.0)

        # === Detected keywords ===
        signals.keywords_detected = list(
            words & (self.POSITIVE_WORDS | self.NEGATIVE_WORDS |
                    self.AROUSAL_HIGH | self.INTIMACY_MARKERS)
        )

        return signals

    def _heuristic_valence(self, words: set) -> float:
        """Fast valence from word lists."""
        pos_count = len(words & self.POSITIVE_WORDS)
        neg_count = len(words & self.NEGATIVE_WORDS)

        total = pos_count + neg_count
        if total == 0:
            return 0.0

        return (pos_count - neg_count) / total

    def _ml_valence(self, text: str) -> float:
        """ML-based valence (slower but more accurate)."""
        if not self._ml_pipeline:
            return 0.0

        try:
            result = self._ml_pipeline(text[:512])[0]  # Truncate for speed
            label = result["label"]
            score = result["score"]

            if label == "POSITIVE":
                return score
            else:
                return -score
        except Exception:
            return 0.0

    def _compute_arousal(self, text: str, words: set) -> float:
        """Compute arousal from multiple signals."""
        arousal = 0.5  # Baseline

        # Word-based
        high_hits = len(words & self.AROUSAL_HIGH)
        low_hits = len(words & self.AROUSAL_LOW)
        arousal += (high_hits - low_hits) * 0.1

        # Punctuation intensity
        exclamation_count = text.count("!")
        question_count = text.count("?")
        caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)

        arousal += exclamation_count * 0.1
        arousal += caps_ratio * 0.3

        # Clamp
        return max(0.0, min(1.0, arousal))

    def reset_context(self):
        """Clear context window."""
        self.context.clear()


class QuickAnalyzer:
    """
    Ultra-fast analyzer using only regex patterns.
    For when even heuristics are too slow.
    Target: <1ms
    """

    PATTERNS = {
        "positive": re.compile(r'\b(love|great|good|yes|thanks|happy|amazing)\b', re.I),
        "negative": re.compile(r'\b(hate|bad|no|sorry|sad|angry|wrong)\b', re.I),
        "question": re.compile(r'\?$'),
        "exclaim": re.compile(r'!$'),
        "intimate": re.compile(r'\b(love|baby|honey|miss you|need you)\b', re.I),
    }

    @classmethod
    def quick_signals(cls, text: str) -> dict:
        """Super fast signal extraction."""
        return {
            "valence": (
                1.0 if cls.PATTERNS["positive"].search(text) else
                -1.0 if cls.PATTERNS["negative"].search(text) else
                0.0
            ),
            "is_question": bool(cls.PATTERNS["question"].search(text)),
            "is_exclaim": bool(cls.PATTERNS["exclaim"].search(text)),
            "intimacy": 0.8 if cls.PATTERNS["intimate"].search(text) else 0.0
        }
