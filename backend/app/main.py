from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import get_settings
from app.core.database import Base, engine
from app.nlp.ollama_engine import OllamaSemanticEngine
from app.nlp.semantic_engine import SemanticEngine
from app.routes import plagiarism

settings = get_settings()

Base.metadata.create_all(bind=engine)

app = FastAPI(title=settings.app_name)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(plagiarism.router, prefix=settings.api_v1_prefix)


@app.on_event("startup")
def warmup_models() -> None:
    # Preload semantic models during startup so first request is faster.
    if settings.preload_models_on_startup and settings.enable_semantic_model:
        models = settings.parsed_semantic_models or [""]
        for name in models:
            SemanticEngine(name, local_files_only=settings.semantic_local_files_only).fit(["warmup model text"])

    if settings.preload_ollama_on_startup and settings.enable_ollama_semantic:
        OllamaSemanticEngine(
            base_url=settings.ollama_base_url,
            model_name=settings.ollama_model_name,
            timeout_seconds=settings.ollama_timeout_seconds,
        ).warmup()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
