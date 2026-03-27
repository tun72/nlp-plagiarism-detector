from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Plagiarism Detector API"
    api_v1_prefix: str = "/api/v1"
    database_url: str = "sqlite:///./plagiarism.db"

    # Detection thresholds and scoring controls.
    similarity_threshold: float = 0.25
    top_matches_per_sentence: int = 4
    min_sentence_length: int = 20
    lexical_candidate_multiplier: int = 3
    min_match_probability: float = 0.22
    min_lexical_similarity: float = 0.03
    min_semantic_similarity: float = 0.22
    min_word_coverage: float = 0.24
    min_char_similarity: float = 0.20

    # Global latency budget (hard stop) to avoid long hangs.
    max_detection_seconds: int = 45
    # Maximum input size accepted for a single plagiarism check.
    max_input_words: int = 1500
    query_parallelism: int = 4

    allow_web_search: bool = True
    web_max_results: int = 4
    # `<= 0` means query all candidate sentences.
    web_query_sentences: int = 0
    web_fetch_timeout: int = 7
    web_google_deep_pages: int = 3
    web_google_results_per_page: int = 20
    web_sentence_cap: int = 110
    web_search_domains: str = (
        "en.wikipedia.org,arxiv.org,semanticscholar.org,openalex.org,crossref.org,"
        "nature.com,sciencedirect.com,springer.com,ieeexplore.ieee.org,jstor.org,"
        "researchgate.net,pubmed.ncbi.nlm.nih.gov,acm.org,wiley.com,tandfonline.com,"
        "plos.org,mdpi.com,ssrn.com,doaj.org,eric.ed.gov,academia.edu,"
        "ucsy.edu.mm,www.ucsy.edu.mm,https://www.facebook.com/lwinmay.thant.796"
    )
    allow_wikipedia_fallback: bool = True

    allow_book_search: bool = True
    book_max_results: int = 4
    # `<= 0` means query all candidate sentences.
    book_query_sentences: int = 0
    book_fetch_timeout: int = 9
    book_sentence_cap: int = 140
    book_search_domains: str = (
        "gutenberg.org,books.google.com,archive.org,openlibrary.org,"
        "hathitrust.org,standardebooks.org,manybooks.net,books.openedition.org,"
        "worldcat.org,bartleby.com,oclc.org"
    )

    source_text_char_limit: int = 45000
    source_sentences_per_doc: int = 20
    source_min_tokens_per_sentence: int = 4

    # Multi-layer scoring weights (must roughly sum to 1.0; normalized at runtime).
    plagiarism_weight_nlp: float = 0.40
    plagiarism_weight_academic: float = 0.20
    plagiarism_weight_web: float = 0.20
    plagiarism_weight_google: float = 0.10
    plagiarism_weight_llm: float = 0.10

    # Retrieval/scoring controls for redesigned pipeline.
    max_query_sentences: int = 6
    semantic_candidate_k: int = 10
    google_verification_sentences: int = 3

    enable_semantic_model: bool = True
    # Comma-separated Hugging Face models. Keep one strong default for better semantic detection.
    semantic_model_names: str = "sentence-transformers/all-mpnet-base-v2"
    # True avoids startup/download stalls when Hugging Face is unreachable.
    semantic_local_files_only: bool = True
    # Keep startup responsive; model can still be loaded lazily on demand.
    preload_models_on_startup: bool = False

    # Optional local LLM signal via Ollama.
    enable_ollama_semantic: bool = True
    ollama_base_url: str = "http://localhost:11434"
    ollama_model_name: str = "llama3.2"
    ollama_timeout_seconds: float = 4.0
    ollama_semantic_blend: float = 0.35
    ollama_max_pairs_per_sentence: int = 1
    ollama_min_lexical_score: float = 0.18
    preload_ollama_on_startup: bool = False

    calibrator_model_path: str | None = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def parsed_book_domains(self) -> list[str]:
        return [d.strip() for d in self.book_search_domains.split(",") if d.strip()]

    @property
    def parsed_web_domains(self) -> list[str]:
        return [d.strip() for d in self.web_search_domains.split(",") if d.strip()]

    @property
    def parsed_semantic_models(self) -> list[str]:
        return [m.strip() for m in self.semantic_model_names.split(",") if m.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
