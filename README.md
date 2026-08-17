# StudyMate AI

An AI-powered learning companion built using:

- LangChain (+ langchain-classic, for EnsembleRetriever)
- Hugging Face Inference API (Qwen2.5-7B-Instruct + all-MiniLM-L6-v2)
- FAISS (+ BM25 hybrid retrieval)
- Streamlit

Multi-format ingestion (PDF / TXT / DOCX) with table extraction, hybrid
retrieval, and four study features on top of the core RAG chat: Notes
Generator (incl. Topic Summaries), Flashcards, Quiz Generator, and Exam Mode.
The UI is a landing page + sidebar-nav dashboard built on top of that
pipeline (Chat, Notes, Flashcards, Exam, Documents, Settings).

See `CHANGELOG.md` for what changed in the latest pass and why.

## Progress

- [x] Project Setup
- [x] LLM
- [x] Document Loader (multi-format + table extraction)
- [x] Text Splitter (table-aware)
- [x] Embeddings
- [x] Vector Store
- [x] Retriever (similarity / mmr / hybrid)
- [x] RAG Chain
- [x] Streamlit UI
- [x] Notes Generator / Topic Summaries
- [x] Flashcards
- [x] Quiz Generator
- [x] Exam Mode
- [x] Automated test suite (pytest, 27 tests)

## Run it

Get a free Hugging Face access token at
[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens),
then:

```bash
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your token into HUGGINGFACEHUB_API_TOKEN

streamlit run app.py
```

Deploying to Streamlit Community Cloud: set `HUGGINGFACEHUB_API_TOKEN` in
the app's Secrets instead of using a `.env` file.

## Run the tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Project layout

```
app.py            Streamlit UI -- landing page + dashboard (Chat, Notes, Flashcards, Quiz, Exam Mode)
pipeline.py       Single ingestion orchestrator (load -> clean -> chunk -> validate)
config.py         All project-wide constants
loaders/          Multi-format loading with table extraction (PDF/DOCX/TXT)
processing/       Text cleaning, structural analysis, semantic chunking, splitting, validation
embeddings/       Hugging Face embeddings wrapper
llm/              Hugging Face LLM wrappers (deterministic + generative variants)
vectorstore/      FAISS wrapper (tracks raw docs for hybrid search)
retriever/        Similarity / MMR / hybrid (BM25 + FAISS) retrieval
schemas/          Pydantic schemas for structured LLM output
prompts/          Prompt templates (RAG, Notes, Flashcards, Quiz)
chains/           LCEL chain builders
utils/            Formatting helpers
db/faiss_index/   Generated vector store (gitignored, rebuilt on ingestion)
uploads/          Uploaded source documents (gitignored)
tests/unit/       Automated pytest suite (no live Hugging Face token required)
tests/fixtures/   3 generated sample documents used by the tests
tests/manual/     Original exploratory scripts (need a real Hugging Face token)
```
