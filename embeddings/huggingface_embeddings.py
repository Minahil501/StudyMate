"""
huggingface_embeddings.py

Embedding service for StudyMate AI.

Why this was changed
---------------------
The previous version talked to a locally-running Ollama server via the
official `ollama` Python client. Same problem as llm/huggingface_llm.py:
Ollama has nothing to talk to on platforms that only host the Streamlit
process itself (e.g. Streamlit Community Cloud).

This version uses `HuggingFaceEndpointEmbeddings`, which calls Hugging
Face's hosted Inference API over HTTPS instead of running a model
locally -- no local model download, no torch/sentence-transformers
install needed. It already implements LangChain's `Embeddings`
interface (`embed_documents` / `embed_query`) directly, so it can be
handed straight to FAISS without a custom wrapper around it.

Requires the same HUGGINGFACEHUB_API_TOKEN environment variable as
llm/huggingface_llm.py -- see that module's docstring for how to set it.
"""

from langchain_huggingface import HuggingFaceEndpointEmbeddings

from config import EMBEDDING_MODEL


class EmbeddingService:
    """
    Exposes `embedding_model` (used by FAISS's `from_documents` /
    `load_local`, and swapped out directly in tests) as a plain,
    settable attribute -- not a property -- since callers legitimately
    need to replace it (e.g. tests monkeypatching in a fake embedder
    that never calls a real API).
    """

    def __init__(self, model: str = EMBEDDING_MODEL):
        self.model = model
        self.embedding_model = HuggingFaceEndpointEmbeddings(model=model)


embeddings = EmbeddingService()
