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

Why `provider="hf-inference"` is explicit
-------------------------------------------
`huggingface_hub`/`langchain_huggingface` now route through Hugging
Face's "Inference Providers" marketplace (third-party backends like
Together, Fireworks, Groq, ...) when `provider` is left unset, instead
of Hugging Face's own free serverless inference. Those third-party
providers generally don't serve plain feature-extraction/embedding
models at all, and can require billing to be configured on your HF
account even when they do -- which is very likely what actually broke
document upload in production. Pinning `provider="hf-inference"` forces
requests through HF's own infrastructure, which is what's actually free
and is what the model name in config.py was chosen for.
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
        self.embedding_model = HuggingFaceEndpointEmbeddings(
            model=model,
            provider="hf-inference",
        )


embeddings = EmbeddingService()
