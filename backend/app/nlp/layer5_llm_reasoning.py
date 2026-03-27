from __future__ import annotations

from dataclasses import dataclass

from app.core.config import Settings, get_settings
from app.nlp.ollama_engine import OllamaSemanticEngine


def _clamp_01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


@dataclass
class LlmReasoningResult:
    score: float
    label: str
    rationale: str


class Layer5LlmReasoning:
    """Optional local LLM reasoning layer for paraphrase/AI rewrite detection."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._engine: OllamaSemanticEngine | None = None

        if self.settings.enable_ollama_semantic:
            engine = OllamaSemanticEngine(
                base_url=self.settings.ollama_base_url,
                model_name=self.settings.ollama_model_name,
                timeout_seconds=self.settings.ollama_timeout_seconds,
            )
            if engine.is_available(timeout_seconds=min(1.0, self.settings.ollama_timeout_seconds)):
                self._engine = engine

    @property
    def enabled(self) -> bool:
        return self._engine is not None

    def analyze_pair(
        self,
        sentence_a: str,
        sentence_b: str,
        *,
        lexical: float,
        semantic: float,
        evidence: float,
        timeout_seconds: float | None = None,
    ) -> LlmReasoningResult:
        llm_score: float | None = None
        if self._engine is not None:
            llm_score = self._engine.similarity(sentence_a, sentence_b, timeout_seconds=timeout_seconds)

        if llm_score is None:
            llm_score = (semantic * 0.70) + (evidence * 0.20) + (lexical * 0.10)

        score = _clamp_01(llm_score)
        if score >= 0.82 and lexical < 0.60:
            label = "ai_rewrite"
            rationale = "Meaning is strongly preserved while lexical overlap is low, indicating rewrite/paraphrase behavior."
        elif score >= 0.76:
            label = "paraphrased"
            rationale = "Semantic equivalence is high with moderate lexical divergence."
        elif score >= 0.60:
            label = "semantic_overlap"
            rationale = "Partial semantic overlap detected, but evidence is not strong enough for direct-copy classification."
        else:
            label = "weak_signal"
            rationale = "LLM found weak semantic alignment."

        return LlmReasoningResult(score=round(score, 4), label=label, rationale=rationale)
