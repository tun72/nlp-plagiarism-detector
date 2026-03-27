from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import get_db
from app.nlp.ollama_engine import OllamaSemanticEngine
from app.nlp.semantic_engine import SemanticEngine
from app.schemas.plagiarism_v2 import PlagiarismRequest, PlagiarismResponseV2
from app.scraper.search_client import SearchClient
from app.services.plagiarism_service import detect_plagiarism

router = APIRouter(prefix="/plagiarism", tags=["plagiarism"])


@router.post("/check", response_model=PlagiarismResponseV2)
def check_plagiarism(payload: PlagiarismRequest, db: Session = Depends(get_db)) -> PlagiarismResponseV2:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text input is required")
    return detect_plagiarism(db, text)


@router.get("/diagnostics")
def diagnostics() -> dict:
    settings = get_settings()
    client = SearchClient()
    report = client.connectivity_report(timeout=3)
    semantic_models = settings.parsed_semantic_models or [""]
    semantic_status = [
        SemanticEngine(model_name=name, local_files_only=settings.semantic_local_files_only).status()
        for name in semantic_models
    ]
    semantic_loaded = any(bool(item.get("model_loaded")) for item in semantic_status)
    ollama_engine = OllamaSemanticEngine(
        base_url=settings.ollama_base_url,
        model_name=settings.ollama_model_name,
        timeout_seconds=settings.ollama_timeout_seconds,
    )
    ollama_up = settings.enable_ollama_semantic and ollama_engine.is_available(timeout_seconds=2.0)
    return {
        "providers_up": report,
        "up_count": sum(1 for v in report.values() if v),
        "total": len(report),
        "ollama_enabled": settings.enable_ollama_semantic,
        "ollama_model": settings.ollama_model_name,
        "ollama_up": ollama_up,
        "semantic_enabled": settings.enable_semantic_model,
        "semantic_local_files_only": settings.semantic_local_files_only,
        "semantic_model_loaded": semantic_loaded,
        "semantic_models": semantic_status,
    }
