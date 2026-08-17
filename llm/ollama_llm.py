# """
# ollama_llm.py

# This module initializes the Large Language Model (LLM)
# used throughout StudyMate AI.

# The rest of the project should import the LLM from here
# instead of creating multiple LLM instances.
# """

# from langchain_ollama import ChatOllama
# from config import GENERATIVE_TEMPERATURE, LLM_MODEL


# class LLMService:
#     """
#     Creates and manages the Ollama LLM.
#     """

#     def __init__(self, temperature: float = 0.2):
#         self.llm = ChatOllama(
#             model=LLM_MODEL,
#             temperature=temperature
#         )

#     def get_llm(self):
#         """
#         Returns the initialized LLM.
#         """
#         return self.llm


# # Singleton instances.
# # - `llm`: deterministic (low temperature), used for Q&A where we want
# #   consistent, grounded answers.
# # - `generative_llm`: slightly higher temperature, used for Flashcards /
# #   Quiz / Notes where some variety in phrasing is fine and even helpful
# #   (e.g. avoiding near-identical flashcard wording).
# llm = LLMService(temperature=0.2).get_llm()
# generative_llm = LLMService(temperature=GENERATIVE_TEMPERATURE).get_llm()


"""
ollama_llm.py

This module initializes the Large Language Model (LLM)
used throughout StudyMate AI.

The rest of the project should import the LLM from here
instead of creating multiple LLM instances.

Why this was changed
---------------------
You're running two different models on the same Ollama server
(nomic-embed-text for embeddings, llama3.2 for generation). By default,
Ollama unloads a model ~5 minutes after its last use and only keeps a
limited number of models resident at once (governed by the
OLLAMA_MAX_LOADED_MODELS env var on the server, default depends on your
available VRAM/RAM). If embedding calls and generation calls interleave
and only one model fits at a time, every switch forces a reload from
disk -- which is often SLOWER than the actual inference and is the most
common cause of "response takes forever" with an otherwise-fast model.

Two independent fixes here:
  1. `keep_alive` tells Ollama to keep this model loaded in memory for a
     set duration after use, instead of unloading after the default
     5 minutes (or immediately, if you'd previously set keep_alive=0
     anywhere). We set a longer keep-alive since this is an interactive
     study app where the LLM will be called repeatedly in a session.
  2. `num_ctx` / `num_predict` are set explicitly. Ollama's default
     num_ctx (2048) is small enough that long RAG contexts get silently
     truncated, but pushing it up "just in case" without setting it also
     costs proportionally more compute/VRAM per call. Set it to match
     what your retriever actually returns (adjust to your real CHUNK_SIZE
     * top_k from config.py / retriever.py -- I don't have those files
     yet, so 4096 is a reasonable starting point, not a measured value).

You should also set this SERVER-SIDE (before `ollama serve` starts) if
you have the RAM/VRAM for it, since it lets both models stay resident at
once rather than fighting over one slot:
    export OLLAMA_MAX_LOADED_MODELS=2
    export OLLAMA_NUM_PARALLEL=2
    export OLLAMA_KEEP_ALIVE=30m
"""

from langchain_ollama import ChatOllama
from config import GENERATIVE_TEMPERATURE, LLM_MODEL

# How long Ollama keeps this model loaded after the last call.
# "30m" keeps it warm for a typical study session; use "-1" to keep it
# loaded indefinitely (uses more idle RAM/VRAM, but eliminates reload
# latency entirely for a long-running app).
KEEP_ALIVE = "30m"

# RAG chat (rag_chain.py): retrieves TOP_K=4 chunks at CHUNK_SIZE=700 chars
# (config.py) -> roughly 2,800 chars / ~1,000 tokens of context. 4096
# leaves comfortable headroom for the prompt template + answer.
CHAT_NUM_CTX = 4096

# Generative chains (study_chains.py): _build_context() sends up to 12,000
# chars (~3,500-4,500 tokens depending on content) for Flashcards/Quiz/Notes,
# since those features intentionally use the (roughly) full document rather
# than just the top-k chunks. 4096 was too tight for that path and risked
# silently truncating context or leaving too little room for the model's
# own output. llama3.2 supports much larger context windows, so this is
# just a config change, not a model limitation.
GENERATIVE_NUM_CTX = 8192


class LLMService:
    """
    Creates and manages the Ollama LLM.
    """

    def __init__(self, temperature: float = 0.2, num_ctx: int = CHAT_NUM_CTX):
        self.llm = ChatOllama(
            model=LLM_MODEL,
            temperature=temperature,
            keep_alive=KEEP_ALIVE,
            num_ctx=num_ctx,
        )

    def get_llm(self):
        """
        Returns the initialized LLM.
        """
        return self.llm


# Singleton instances.
# - `llm`: deterministic (low temperature), used for Q&A where we want
#   consistent, grounded answers. Smaller context budget -- it only ever
#   sees TOP_K retrieved chunks.
# - `generative_llm`: slightly higher temperature, used for Flashcards /
#   Quiz / Notes where some variety in phrasing is fine and even helpful
#   (e.g. avoiding near-identical flashcard wording). Larger context
#   budget -- it sees much more of the source document per call.
llm = LLMService(temperature=0.2, num_ctx=CHAT_NUM_CTX).get_llm()
generative_llm = LLMService(temperature=GENERATIVE_TEMPERATURE, num_ctx=GENERATIVE_NUM_CTX).get_llm()