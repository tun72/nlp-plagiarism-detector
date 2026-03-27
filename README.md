# AI-Powered Plagiarism Detection Web Application

This project is a full-stack plagiarism detection system that combines information retrieval, semantic similarity, and feature-based probability scoring.

It scans user text sentence-by-sentence, compares it against automatically discovered web/book sources, and returns confidence-ranked matches.

## What This Project Does

- Accepts a paragraph/document from the user.
- Splits the input into sentences and normalizes each sentence.
- Retrieves candidate sources from web and book providers (including Wikipedia).
- Builds lexical and semantic similarity signals.
- Computes final plagiarism probability per sentence.
- Returns sentence-level matches, source-level summary, and overall similarity.

## Tech Stack

### Backend

- FastAPI
- SQLAlchemy (SQLite by default)
- scikit-learn
- sentence-transformers
- Ollama local LLM (optional, default model: `llama3.2`)
- NLTK + spaCy (with robust fallbacks)
- requests + BeautifulSoup

### Frontend

- Next.js (App Router)
- React
- Axios

## High-Level Architecture

1. User enters text in the frontend.
2. Frontend calls `POST /api/v1/plagiarism/check`.
3. Backend service performs retrieval + NLP scoring pipeline.
4. Backend returns per-sentence matches, probabilities, top sources, and overall similarity.
5. Frontend visualizes results with risk indicators and source panels.

## Plagiarism Detection Pipeline

The main pipeline is implemented in:
- `backend/app/services/plagiarism_engine.py`
- `backend/app/services/plagiarism_service.py` (route wrapper)

### Step 1: Sentence segmentation and normalization

- Split input into sentences.
- Normalize sentences with tokenization, stopword removal, and lemmatization.
- Fallback logic is used when NLTK/spaCy resources are not available, so requests do not hang.

### Step 2: Candidate source retrieval

- Build search queries from representative input sentences.
- Retrieve from web and book providers in parallel.
- Extract and normalize candidate source sentences.
- Use strict timeout budgeting and non-blocking executor shutdown to avoid long stalls.

### Step 3: Lexical retrieval scoring

- Use TF-IDF (`word` n-grams 1-2) for lexical similarity.
- Select top lexical candidates per sentence before deeper scoring.

### Step 4: Semantic scoring

- Use Hugging Face `sentence-transformers` embeddings (default: `all-mpnet-base-v2`).
- Compute cosine similarity between input sentence embedding and candidate sentence embedding.
- Optionally blend a local Ollama semantic score (`llama3.2` by default) for top candidates.
- Fallback to character n-gram TF-IDF semantic proxy if embedding model is unavailable.

### Step 5: Feature engineering

For each sentence-candidate pair:
- `lexical_score` (TF-IDF cosine similarity)
- `semantic_score` (embedding cosine similarity or fallback)
- `overlap_score` (token overlap/Jaccard-style)
- `containment_score` (short-text token containment in long text)

### Step 6: Final plagiarism probability

Sentence-level final probability is a multi-layer weighted score:

`0.40 * NLP + 0.20 * Academic + 0.20 * Website + 0.10 * Google + 0.10 * LLM`

Where:
- NLP = lexical + semantic + evidence features calibrated by `AIPlagiarismDetectorModel`
- Academic = source-level evidence from academic/book providers
- Website = source-level evidence from website crawl layer
- Google = deep Google snippet/verification evidence
- LLM = Ollama local reasoning signal (optional)

### Step 7: Decision and aggregation

- Sentence is marked `flagged` using an effective threshold derived from `SIMILARITY_THRESHOLD`.
- Overall similarity is aggregated from weighted layer evidence at sentence level.
- Top sources are aggregated by best probability + hit count.

## Source Providers

### Web side

- Wikipedia API (explicitly supported)
- Crossref
- OpenAlex
- arXiv
- Semantic Scholar
- Search result crawling fallback (DuckDuckGo/Brave/Google discovered links)
- Default domain-targeted crawl includes:
  `en.wikipedia.org`, `arxiv.org`, `semanticscholar.org`, `openalex.org`, `crossref.org`,
  `nature.com`, `sciencedirect.com`, `springer.com`, `ieeexplore.ieee.org`, `jstor.org`,
  `researchgate.net`, `pubmed.ncbi.nlm.nih.gov`

### Book side

- OpenLibrary
- Google Books
- Archive.org
- Gutendex / Project Gutenberg
- Additional domain-targeted crawl includes:
  `hathitrust.org`, `standardebooks.org`, `manybooks.net`

## API

### `POST /api/v1/plagiarism/check`

Request:

```json
{
  "text": "Your paragraph..."
}
```

Response fields:

- `overall_similarity_percent`
- `flagged_sentences`
- `total_sentences`
- `scanned_web_sources`
- `scanned_book_sources`
- `processing_ms`
- `top_accuracy_sources[]`
  each source includes `source_url`, `best_probability`, `average_probability`, `average_word_coverage`
- `results[]` (per sentence, with detailed matches)
  each match includes `source_url`, `containment_score`, `word_coverage`

### `GET /api/v1/plagiarism/diagnostics`

Returns provider connectivity status plus Hugging Face semantic model load status.

### `GET /health`

Basic health endpoint.

## UI Behavior Notes

- UI red underline currently appears when sentence confidence is `>= 60%`.
- Backend `flagged` metric still uses backend threshold (default `75%`).

## Configuration

Backend env file: `backend/.env`

Key controls:

- `SIMILARITY_THRESHOLD=0.75`
- `MIN_MATCH_PROBABILITY=0.22`
- `MAX_DETECTION_SECONDS=45`
- `MAX_INPUT_WORDS=1500`
- `QUERY_PARALLELISM=4`
- `ALLOW_WEB_SEARCH=true`
- `WEB_QUERY_SENTENCES=0` (0 = query all input sentences/lines)
- `WEB_GOOGLE_DEEP_PAGES=3` (Google pagination depth)
- `WEB_GOOGLE_RESULTS_PER_PAGE=20`
- `WEB_SEARCH_DOMAINS=en.wikipedia.org,arxiv.org,semanticscholar.org,openalex.org,crossref.org,nature.com,sciencedirect.com,springer.com,ieeexplore.ieee.org,jstor.org,researchgate.net,pubmed.ncbi.nlm.nih.gov,ucsy.edu.mm,www.ucsy.edu.mm,https://www.facebook.com/lwinmay.thant.796`
- `ALLOW_WIKIPEDIA_FALLBACK=true`
- `ALLOW_BOOK_SEARCH=true`
- `BOOK_QUERY_SENTENCES=0` (0 = query all input sentences/lines)
- `BOOK_SEARCH_DOMAINS=gutenberg.org,books.google.com,archive.org,openlibrary.org,hathitrust.org,standardebooks.org,manybooks.net`
- `PLAGIARISM_WEIGHT_NLP=0.40`
- `PLAGIARISM_WEIGHT_ACADEMIC=0.20`
- `PLAGIARISM_WEIGHT_WEB=0.20`
- `PLAGIARISM_WEIGHT_GOOGLE=0.10`
- `PLAGIARISM_WEIGHT_LLM=0.10`
- `MAX_QUERY_SENTENCES=6`
- `SEMANTIC_CANDIDATE_K=10`
- `GOOGLE_VERIFICATION_SENTENCES=3`
- `ENABLE_SEMANTIC_MODEL=true`
- `SEMANTIC_MODEL_NAMES=sentence-transformers/all-mpnet-base-v2`
- `SEMANTIC_LOCAL_FILES_ONLY=true` (offline-safe; uses cached local model only)
- `PRELOAD_MODELS_ON_STARTUP=false` (faster startup, avoids HF network retries at boot)
- `ENABLE_OLLAMA_SEMANTIC=true`
- `OLLAMA_BASE_URL=http://localhost:11434`
- `OLLAMA_MODEL_NAME=llama3.2`
- `OLLAMA_TIMEOUT_SECONDS=4`
- `OLLAMA_SEMANTIC_BLEND=0.35`
- `OLLAMA_MAX_PAIRS_PER_SENTENCE=1`
- `OLLAMA_MIN_LEXICAL_SCORE=0.18`
- `PRELOAD_OLLAMA_ON_STARTUP=false`
- `CALIBRATOR_MODEL_PATH=`

Frontend env file: `frontend/.env.local`

- `NEXT_PUBLIC_API_BASE_URL=http://localhost:8000/api/v1`
- `NEXT_PUBLIC_API_TIMEOUT_MS=180000`

## Local Setup

### 1) Backend

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --port 8000
```

Optional for richer NLTK tokenization resources:

```bash
python -m nltk.downloader punkt stopwords wordnet omw-1.4
```

Optional for local LLM semantic scoring with Ollama:

```bash
ollama pull llama3.2
ollama serve
```

### 2) Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

## Training a Calibrator Model (Optional)

Train logistic regression calibrator using labeled sentence pairs:

```bash
cd backend
source .venv/bin/activate
python -m scripts.train_calibrator --input /path/to/train.csv --output ./models/plagiarism_calibrator.joblib --enable-semantic
```

Expected CSV columns:

- `text_a`
- `text_b`
- `label` (0/1)

## Project Structure

```text
backend/
  app/
    core/
    nlp/
    routes/
    scraper/
    services/
  scripts/
frontend/
  app/
  components/
  lib/
docs/
```

## Documentation Deliverables

- Project details: this `README.md`
- NLP technical write-up (PDF): `docs/NLP_Detection_Theory_and_Methods.pdf`
- NLP technical source (Markdown): `docs/NLP_Detection_Theory_and_Methods.md`

## Current Limitations

- Provider coverage/quality depends on network availability and API responsiveness.
- Some sources return snippets rather than full-text passages.
- Without a trained calibrator model, scoring uses fixed fallback weights.
- Cross-domain plagiarism can still require more advanced paraphrase/translation detection modules.
