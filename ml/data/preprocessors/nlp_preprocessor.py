"""
NLP text preprocessor for noisy social/news crypto text.
"""
from __future__ import annotations

import re
import string
from typing import Optional

import emoji
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from loguru import logger

# Download required NLTK data once
for _resource in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_resource}")
    except LookupError:
        nltk.download(_resource, quiet=True)


CRYPTO_SLANG: dict[str, str] = {
    "hodl": "hold",
    "fud": "fear uncertainty doubt",
    "fomo": "fear of missing out",
    "btfd": "buy the dip",
    "rekt": "destroyed",
    "ath": "all time high",
    "atl": "all time low",
    "dyor": "do your own research",
    "ngmi": "not going to make it",
    "wagmi": "we are all going to make it",
    "gm": "good morning",
    "wen": "when",
    "ser": "sir",
    "nfa": "not financial advice",
    "lmao": "laughing",
    "lol": "laughing",
    "imo": "in my opinion",
    "tbh": "to be honest",
    "idk": "i do not know",
}


class NLPPreprocessor:
    """
    Clean and normalise raw text from Twitter, Reddit, and news.
    Pipeline:
      1. Decode emojis to text
      2. Lowercase
      3. Expand crypto slang
      4. Remove URLs, mentions, hashtag symbols, special chars
      5. Tokenise
      6. Remove stop-words
      7. Lemmatize
    """

    def __init__(self, language: str = "english") -> None:
        self._stop_words = set(stopwords.words(language))
        self._lemmatizer = WordNetLemmatizer()
        logger.info("NLPPreprocessor ready.")

    # ── Individual steps ─────────────────────────────────────────────────

    @staticmethod
    def decode_emojis(text: str) -> str:
        """Replace emojis with their text description."""
        return emoji.demojize(text, delimiters=(" ", " "))

    @staticmethod
    def expand_slang(text: str) -> str:
        tokens = text.split()
        return " ".join(CRYPTO_SLANG.get(t.lower(), t) for t in tokens)

    @staticmethod
    def remove_noise(text: str) -> str:
        """Strip URLs, mentions, hashtag symbols, and special characters."""
        text = re.sub(r"http\S+|www\.\S+", " ", text)       # URLs
        text = re.sub(r"@\w+", " ", text)                   # mentions
        text = re.sub(r"#(\w+)", r"\1", text)               # keep hashtag word
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)        # special chars
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def lemmatize(self, tokens: list[str]) -> list[str]:
        return [self._lemmatizer.lemmatize(t) for t in tokens]

    def remove_stopwords(self, tokens: list[str]) -> list[str]:
        return [t for t in tokens if t not in self._stop_words and len(t) > 1]

    # ── Full pipeline ─────────────────────────────────────────────────────

    def preprocess(self, text: str) -> str:
        """Return a cleaned, lemmatized sentence (space-joined tokens)."""
        if not isinstance(text, str) or not text.strip():
            return ""
        text = self.decode_emojis(text)
        text = text.lower()
        text = self.expand_slang(text)
        text = self.remove_noise(text)
        tokens = text.split()
        tokens = self.remove_stopwords(tokens)
        tokens = self.lemmatize(tokens)
        return " ".join(tokens)

    def preprocess_batch(self, texts: list[str]) -> list[str]:
        return [self.preprocess(t) for t in texts]
