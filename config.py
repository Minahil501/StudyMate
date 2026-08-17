"""
Global configuration for StudyMate AI.
All project-wide constants are stored here.
"""

# ==========================
# LLM Configuration
# ==========================

LLM_MODEL = "llama3.2:3b"

# Slightly higher temperature for creative-generation features
# (flashcards / quiz / notes) where a bit of variety in phrasing is fine.
# Keep the default RAG chat LLM deterministic (see llm/ollama_llm.py).
GENERATIVE_TEMPERATURE = 0.4

# ==========================
# Embedding Configuration
# ==========================

EMBEDDING_MODEL = "nomic-embed-text"

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
