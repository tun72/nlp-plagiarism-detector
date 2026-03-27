from __future__ import annotations

import re
from functools import lru_cache

from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import WordNetLemmatizer

_FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "was",
    "were",
    "with",
}


def _simple_sentence_split(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def _simple_word_tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


@lru_cache
def _load_spacy():
    try:
        import spacy
    except Exception:
        return None

    try:
        return spacy.load("en_core_web_sm", disable=["ner", "parser"])
    except OSError:
        return None


@lru_cache
def _load_stopwords() -> set[str]:
    try:
        return set(stopwords.words("english"))
    except LookupError:
        # Do not block requests by downloading corpora at runtime.
        return _FALLBACK_STOPWORDS


def split_sentences(text: str) -> list[str]:
    try:
        return [s.strip() for s in sent_tokenize(text) if s.strip()]
    except LookupError:
        return _simple_sentence_split(text)


@lru_cache(maxsize=20000)
def normalize_sentence(sentence: str) -> str:
    sentence = sentence.lower().strip()
    sentence = re.sub(r"\s+", " ", sentence)

    try:
        tokens = word_tokenize(sentence)
    except LookupError:
        tokens = _simple_word_tokenize(sentence)

    stop_words = _load_stopwords()
    # Keep tokens that contain at least one alphabetic character.
    filtered = [t for t in tokens if any(ch.isalpha() for ch in t) and t not in stop_words]

    if not filtered:
        return ""

    nlp = _load_spacy()
    if nlp is not None:
        doc = nlp(" ".join(filtered))
        lemmas = [token.lemma_ for token in doc if token.lemma_ and token.lemma_ != "-PRON-"]
        return " ".join(lemmas)

    # Fallback when spaCy is unavailable in the active Python environment.
    lemmatizer = WordNetLemmatizer()
    try:
        lemmas = [lemmatizer.lemmatize(token) for token in filtered]
    except LookupError:
        lemmas = filtered
    return " ".join(lemmas)
