# from langchain_ollama import OllamaEmbeddings
# from config import EMBEDDING_MODEL


# class EmbeddingService:
#     """
#     Handles text embeddings using Ollama.
#     """

#     def __init__(self):

#         self.embedding_model = OllamaEmbeddings(
#             model=EMBEDDING_MODEL
#         )

#     def embed_documents(self, texts):
#         """
#         Embed multiple texts.
#         """
#         return self.embedding_model.embed_documents(texts)

#     def embed_query(self, query):
#         """
#         Embed a user query.
#         """
#         return self.embedding_model.embed_query(query)


# embeddings = EmbeddingService()



"""
ollama_embeddings.py

Embedding service for StudyMate AI.

Why this was rewritten
-----------------------
The previous version delegated straight to `langchain_ollama.OllamaEmbeddings`.
Depending on the installed `langchain-ollama` version, that class either:
  (a) sends ONE HTTP request per text to Ollama's `/api/embeddings` endpoint
      (older versions) -- so embedding 150 chunks means 150 sequential
      round trips, each paying HTTP + model-inference overhead, or
  (b) batches internally via `/api/embed` (newer versions) but still as a
      single blocking call with no chunking/backoff for very large jobs.

This version talks to Ollama directly via the official `ollama` Python
client and:
  1. Uses the batch `/api/embed` endpoint (list-in, list-out) instead of
     one-at-a-time calls -- this alone is usually a 5-20x speedup for
     document ingestion, since it removes N-1 round trips.
  2. Sends batches concurrently (thread pool) so large documents don't
     serialize on a single request.
  3. Applies Nomic's recommended task prefixes ("search_document: " /
     "search_query: "), which measurably improves retrieval quality for
     nomic-embed-text -- LangChain's default wrapper does NOT do this for
     you.
  4. Shows progress so a slow embed run is visible instead of looking
     "frozen" for a minute.

If you upgrade `langchain-ollama` in the future this class still works
the same way, since it bypasses that layer entirely and talks to the
Ollama server directly.
"""

from concurrent.futures import ThreadPoolExecutor

from langchain_core.embeddings import Embeddings
from ollama import Client

from config import EMBEDDING_MODEL

# Tune this based on your machine. Ollama serializes requests to the SAME
# model by default unless the server has OLLAMA_NUM_PARALLEL > 1 set as an
# environment variable before `ollama serve` starts. Sending more parallel
# requests than the server can actually run concurrently just queues them
# client-side -- it doesn't help, but it doesn't hurt either. 4 is a safe
# default for a single local GPU/CPU box.
MAX_WORKERS = 4

# How many texts to send per batch call. Ollama's /api/embed endpoint
# accepts a list, so this isn't about API limits -- it's to keep any single
# HTTP payload/response reasonable and to give you granular progress
# reporting instead of one giant all-or-nothing call.
BATCH_SIZE = 16


class EmbeddingService(Embeddings):
    """
    Handles text embeddings using Ollama's native batch embedding endpoint.

    Inherits from langchain_core.embeddings.Embeddings -- this matters,
    not just for style. FAISS (and other LangChain vector stores) check
    `isinstance(embedding_function, Embeddings)` before deciding HOW to
    call it: if the check passes, it calls `.embed_query()` /
    `.embed_documents()` (what we implement below); if it fails, it
    assumes you handed it a bare function and tries to call the object
    directly, which breaks with exactly the error you hit
    ("'EmbeddingService' object is not callable").
    """

    def __init__(self, model: str = EMBEDDING_MODEL, host: str | None = None):
        self.model = model
        self.client = Client(host=host) if host else Client()

    # ------------------------------------------------------------
    # Public API (kept compatible with the previous EmbeddingService,
    # and exposes `embedding_model` for FAISS's `from_documents` /
    # `load_local` calls, which expect a LangChain-style Embeddings
    # object with .embed_documents / .embed_query)
    # ------------------------------------------------------------

    @property
    def embedding_model(self):
        # FAISS just needs any object exposing embed_documents/embed_query.
        return self

    def embed_documents(self, texts: list[str], show_progress: bool = True) -> list[list[float]]:
        """
        Embed multiple texts using batched, concurrent calls to Ollama.
        Applies the "search_document: " prefix Nomic recommends for
        anything being stored/indexed (as opposed to a live query).
        """

        if not texts:
            return []

        prefixed = [f"search_document: {t}" for t in texts]
        batches = [
            prefixed[i:i + BATCH_SIZE]
            for i in range(0, len(prefixed), BATCH_SIZE)
        ]

        results: list[list[list[float]] | None] = [None] * len(batches)

        def _embed_batch(index_batch):
            index, batch = index_batch
            response = self.client.embed(model=self.model, input=batch)
            return index, response["embeddings"]

        completed = 0
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for index, embeddings in pool.map(_embed_batch, enumerate(batches)):
                results[index] = embeddings
                completed += 1
                if show_progress:
                    print(f"  Embedding batch {completed}/{len(batches)}", end="\r")

        if show_progress:
            print()

        flattened: list[list[float]] = []
        for batch_result in results:
            flattened.extend(batch_result)

        return flattened

    def embed_query(self, query: str) -> list[float]:
        """
        Embed a single user query. Uses the "search_query: " prefix, which
        is the counterpart to "search_document: " above -- nomic-embed-text
        was trained with these paired prefixes and retrieval quality drops
        without them.
        """

        response = self.client.embed(model=self.model, input=f"search_query: {query}")
        return response["embeddings"][0]


embeddings = EmbeddingService()