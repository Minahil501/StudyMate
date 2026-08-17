"""
Global configuration for StudyMate AI.
All project-wide constants are stored here.
"""

from dotenv import load_dotenv

# Loads HUGGINGFACEHUB_API_TOKEN (and anything else in a local .env) into
# the environment before llm/huggingface_llm.py or
# embeddings/huggingface_embeddings.py read it. No-op if there's no .env
# file (e.g. on Streamlit Cloud, where secrets are injected as real env
# vars directly).
load_dotenv()

# ==========================
# LLM Configuration
# ==========================

# Served via Hugging Face's hosted Inference API (see llm/huggingface_llm.py)
# rather than a local Ollama server, so the app also works on hosting that
# has no local model server -- e.g. Streamlit Community Cloud. Requires a
# free Hugging Face access token set as the HUGGINGFACEHUB_API_TOKEN
# environment variable (locally: a .env file; on Streamlit Cloud: the app's
# Secrets). Get one at https://huggingface.co/settings/tokens.
LLM_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# Slightly higher temperature for creative-generation features
# (flashcards / quiz / notes) where a bit of variety in phrasing is fine.
# Keep the default RAG chat LLM deterministic (see llm/huggingface_llm.py).
GENERATIVE_TEMPERATURE = 0.4

# ==========================
# Embedding Configuration
# ==========================

# Also served via Hugging Face's hosted Inference API (see
# embeddings/huggingface_embeddings.py) -- same HUGGINGFACEHUB_API_TOKEN
# as above. all-MiniLM-L6-v2 is small, fast, and widely available on the
# free tier.
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ==========================
# Text Splitter
# ==========================

CHUNK_SIZE = 700
CHUNK_OVERLAP = 120

SEPARATORS = [
    "\n\n",
    "\n",
    ". ",
    "? ",
    "! ",
    "; ",
    ", ",
    " ",
    ""
]

# ==========================
# Vector Store
# ==========================

VECTOR_DB_PATH = "db/faiss_index"

# ==========================
# Upload Folder
# ==========================

UPLOAD_FOLDER = "uploads"

# ==========================
# Retriever
# ==========================

# "similarity" | "mmr" | "hybrid"
# - similarity : plain vector similarity search
# - mmr        : maximal marginal relevance (diversity + relevance)
# - hybrid     : BM25 (keyword) + FAISS (semantic) combined via EnsembleRetriever
SEARCH_TYPE = "hybrid"

TOP_K = 4
SCORE_THRESHOLD = 0.5
FETCH_K = 20
LAMBDA_MULT = 0.5

# Weights for the hybrid retriever. Must sum to 1.0.
# Higher BM25 weight helps when questions quote exact terms/numbers
# (e.g. "what is the value in row X"), higher vector weight helps
# with paraphrased / conceptual questions.
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6

# ==========================
# Table handling
# ==========================

# Tables are extracted separately from prose and are never split by the
# recursive splitter, regardless of size, so rows/columns stay intact.
MAX_TABLE_CHUNK_SIZE = 4000  # soft warning threshold only, not enforced

# ==========================
# Study Feature Defaults
# ==========================

DEFAULT_FLASHCARD_COUNT = 10
DEFAULT_QUIZ_QUESTION_COUNT = 5
DEFAULT_QUIZ_DIFFICULTY = "medium"  # easy | medium | hard
QUIZ_DIFFICULTIES = ["easy", "medium", "hard"]

NOTES_STYLES = ["detailed", "concise", "topic_summary"]
DEFAULT_NOTES_STYLE = "detailed"
