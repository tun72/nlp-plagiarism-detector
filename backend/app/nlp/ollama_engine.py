from __future__ import annotations

import json
import re
from collections import OrderedDict

import requests


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


class OllamaSemanticEngine:
    """Local Ollama helper that returns a semantic similarity score in [0, 1]."""

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: float = 4.0,
        cache_size: int = 256,
    ) -> None:
        self.base_url = (base_url or "http://localhost:11434").rstrip("/")
        self.model_name = (model_name or "").strip()
        self.timeout_seconds = max(0.2, float(timeout_seconds))
        self.cache_size = max(16, int(cache_size))
        self._session = requests.Session()
        self._cache: OrderedDict[tuple[str, str], float] = OrderedDict()

    def _cache_get(self, key: tuple[str, str]) -> float | None:
        value = self._cache.get(key)
        if value is None:
            return None
        self._cache.move_to_end(key)
        return value

    def _cache_set(self, key: tuple[str, str], value: float) -> None:
        self._cache[key] = value
        self._cache.move_to_end(key)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _parse_score(self, raw: str) -> float | None:
        text = (raw or "").strip()
        if not text:
            return None

        # Prefer structured output when available.
        try:
            parsed_json = json.loads(text)
            if isinstance(parsed_json, dict) and "score" in parsed_json:
                return _clamp_01(float(parsed_json["score"]))
            if isinstance(parsed_json, (int, float)):
                return _clamp_01(float(parsed_json))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

        # Fallback for non-JSON model responses.
        match = re.search(r"-?\d+(?:\.\d+)?", text)
        if not match:
            return None

        parsed = float(match.group(0))
        if "%" in text and parsed > 1.0:
            parsed = parsed / 100.0
        elif parsed > 1.0 and parsed <= 100.0:
            parsed = parsed / 100.0
        return _clamp_01(parsed)

    def _request_timeout(self, timeout_seconds: float | None) -> float:
        if timeout_seconds is None:
            return self.timeout_seconds
        return max(0.2, min(float(timeout_seconds), self.timeout_seconds))

    def similarity(self, sentence_a: str, sentence_b: str, timeout_seconds: float | None = None) -> float | None:
        if not self.model_name:
            return None

        text_a = " ".join((sentence_a or "").split())
        text_b = " ".join((sentence_b or "").split())
        if not text_a or not text_b:
            return None

        key = tuple(sorted((text_a.lower(), text_b.lower())))
        cached = self._cache_get(key)
        if cached is not None:
            return cached

        prompt = (
            "You are a strict semantic similarity grader for plagiarism detection.\n"
            'Respond with strict JSON only: {"score": <number between 0 and 1>}.\n'
            "Scoring rubric:\n"
            "- 0.0: unrelated meaning\n"
            "- 0.2: same broad domain but different claims\n"
            "- 0.5: partial overlap in meaning\n"
            "- 0.8: mostly the same claim\n"
            "- 1.0: same meaning or close paraphrase\n"
            'Example 1: A=Bananas contain potassium. B=The Eiffel Tower is in Paris. => {"score": 0.0}\n'
            'Example 2: A=The cat sat on the mat. B=A cat was sitting on the mat. => {"score": 0.95}\n'
            'Example 3: A=Climate change increases heatwaves. B=Global warming causes more extreme heat events. => {"score": 0.9}\n'
            f"Sentence A: {text_a}\n"
            f"Sentence B: {text_b}\n"
            "JSON:"
        )

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": 96,
            },
        }

        try:
            response = self._session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self._request_timeout(timeout_seconds),
            )
            response.raise_for_status()
            score = self._parse_score((response.json() or {}).get("response", ""))
            if score is None:
                return None
            self._cache_set(key, score)
            return score
        except (requests.RequestException, ValueError, TypeError):
            return None

    def is_available(self, timeout_seconds: float | None = None) -> bool:
        if not self.model_name:
            return False
        try:
            response = self._session.get(
                f"{self.base_url}/api/tags",
                timeout=self._request_timeout(timeout_seconds),
            )
            response.raise_for_status()
            payload = response.json() or {}
            models = payload.get("models", [])
            requested = self.model_name.lower()
            requested_base = requested.split(":")[0]
            for item in models:
                name = str(item.get("name") or "").lower()
                if name == requested or name.startswith(f"{requested_base}:"):
                    return True
        except (requests.RequestException, ValueError, TypeError):
            return False
        return False

    def warmup(self) -> None:
        self.similarity("This is a warmup sentence.", "This is a warmup sentence.", timeout_seconds=1.5)
