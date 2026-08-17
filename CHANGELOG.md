# CHANGELOG — StudyMate AI rewrite

## Bugs found and fixed
- **`processing/validate.py` executed a full pipeline run on import.** A leftover
  copy-pasted script at the bottom of the file ran `DocumentLoader.load(...)` →
  ... → `ChunkValidator.validate(...)` every time the module was imported.
  Deleted the file; `processing/validator.py` is now the single validator.
- **`config.py` declared `SEARCH_TYPE = "hybrid"` as an option, but no code
  implemented it.** Passing it into FAISS's `.as_retriever()` would raise a
  `ValueError`. `retriever/retriever.py` now actually builds a
  BM25 + FAISS `EnsembleRetriever` for this case.
- **Tables were silently destroyed on ingestion.** `PyPDFLoader` and
  `Docx2txtLoader` both flatten tables into column-less text. Rewrote
  `loaders/loaders_factory.py` to extract tables as separate, clean
  Markdown-table `Document`s (caught and fixed a duplication bug along the
  way: pdfplumber's page-level text extraction re-includes table content
  unless you explicitly exclude the table bounding boxes first).
- **`source` metadata was inconsistent across loaders.** `TextLoader`
  defaults to the full file path; the PDF/DOCX loaders used just the
  filename. Normalized all three to filename-only (caught by the
  `test_process_documents_merges_multiple_files` test).
- **`req.txt` was saved as UTF-16**, which is why it looked like garbage
  when opened as plain text. Converted to UTF-8 and renamed to the
  conventional `requirements.txt`.

## New capability
- **Hybrid retrieval** (`retriever/retriever.py`, `vectorstore/faiss_store.py`):
  BM25 (keyword) + FAISS (semantic) combined via `EnsembleRetriever`, weighted
  60/40 vector/BM25 by default (`config.BM25_WEIGHT` / `VECTOR_WEIGHT`).
  Note for your version pin: `EnsembleRetriever` now lives in
  `langchain_classic`, not `langchain.retrievers` — added to requirements.
- **Table-aware chunking** (`processing/splitters.py`): tables are never
  split, regardless of size, since breaking a Markdown table mid-row
  corrupts it.
- **`pipeline.py`**: single orchestrator function (load → clean → analyze →
  chunk → validate) that both `app.py` and the test suite now call, instead
  of the load/clean/chunk sequence being copy-pasted across test scripts.
- **Structured-output generation** (`schemas/study_schemas.py`,
  `chains/study_chains.py`): Flashcards, Quiz, and Notes all use
  `llm.with_structured_output(PydanticSchema)` so the app gets validated
  Python objects, not prose to regex apart.
- **Full Streamlit UI** (`app.py`, previously empty): 5 tabs — Chat,
  Notes Generator (doubles as Topic Summaries via a style parameter),
  Flashcards, Quiz Generator, Exam Mode (a timed run of the Quiz Generator
  with a scored review screen).

## Known simplifications (worth mentioning in an interview, not hiding)
- Notes/Flashcards/Quiz generate from the full chunk set with a hard
  character-count truncation (`chains/study_chains.py:_build_context`),
  not a map-reduce pass. Fine for a single document/day's coursework; a
  production version would map-reduce over chunk groups for very large
  documents instead of truncating.
- Exam Mode's timer recalculates "time remaining" on every Streamlit
  rerun rather than ticking live; a true live countdown needs a small JS
  component or an autorefresh loop.
- The FAISS/vector-store singletons are process-wide, which is fine for a
  single local user running the app, but would need per-session isolation
  for multi-user deployment.

## Test suite
- `tests/unit/` — automated pytest suite (27 tests), runs without a live
  Ollama server via `FakeEmbeddings` (a real `Embeddings` subclass, not a
  bare callable — FAISS does an `isinstance` check) and a fake
  structured-output LLM.
- `tests/manual/` — your original exploratory scripts, moved here (with a
  README) because pytest's default `test*.py` discovery was picking them
  up and running the full pipeline against a live Ollama server as a
  side effect of test *collection*. `pytest.ini` now scopes discovery to
  `tests/unit/`.
- `tests/fixtures/` — 3 generated documents (`sample.pdf`, `sample.docx`,
  `sample.txt`) matching your actual coursework topics (LL(1) parsing +
  FIRST/FOLLOW table, CAP theorem + database comparison table, IEEE CS UCP
  event notes), regenerable via `tests/fixtures/generate_fixtures.py`.

## To run
```
pip install -r requirements.txt
cp .env.example .env   # add your HUGGINGFACEHUB_API_TOKEN
streamlit run app.py
```

To run the test suite:
```
pip install -r requirements-dev.txt
pytest
```

## Deployment: Ollama -> Hugging Face Inference API

The LLM and embeddings backend originally talked to a locally-running
Ollama server (`llama3.2:3b` + `nomic-embed-text`). That's fine for
local development, but Streamlit Community Cloud only hosts the
Streamlit process itself -- there's no way to also run Ollama alongside
it, so a public deployment had nothing to connect to.

Swapped both `llm/ollama_llm.py` and `embeddings/ollama_embeddings.py`
(renamed to `llm/huggingface_llm.py` / `embeddings/huggingface_embeddings.py`)
for Hugging Face's hosted Inference API:
- LLM: `ChatHuggingFace` wrapping `HuggingFaceEndpoint`
  (`Qwen/Qwen2.5-7B-Instruct` by default). `ChatHuggingFace` is a real
  LangChain `BaseChatModel`, so `.with_structured_output(Schema)` in
  `chains/study_chains.py` needed no changes.
- Embeddings: `HuggingFaceEndpointEmbeddings`
  (`sentence-transformers/all-MiniLM-L6-v2` by default), which calls the
  Inference API remotely -- no local model download, no torch install.

Requires a free Hugging Face access token (`HUGGINGFACEHUB_API_TOKEN`,
via `.env` locally or Streamlit Cloud Secrets in production).

Also removed `spacy`, `unstructured`, and both packages' otherwise-unused
transitive dependency trees (31 pinned lines) from `requirements.txt` --
neither is ever imported by this codebase (document loading is hand-rolled
directly on `pdfplumber`/`python-docx` in `loaders/loaders_factory.py`),
and `spacy==3.8.14` specifically doesn't exist on PyPI, which is what broke
the first deploy attempt. Added `runtime.txt` pinning Python 3.13 to match
local dev, since the rest of the pinned list is an exact `pip freeze` of
compiled packages that may not have wheels yet for a brand-new Python
version.

Known caveat: unlike Ollama's native structured-output support,
`.with_structured_output()` on the Inference API depends on the chosen
model/provider actually supporting tool calling. If you swap `LLM_MODEL`
(`config.py`) for a model without solid tool-calling support,
Flashcards/Quiz/Notes generation may degrade even though plain chat still
works.
