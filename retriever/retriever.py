# """
# retriever.py

# Why this file was rewritten
# ----------------------------
# config.py already listed "hybrid" as a valid SEARCH_TYPE option (with a
# comment describing it), but no code actually implemented it -- passing
# search_type="hybrid" straight into FAISS's `.as_retriever()` would have
# raised a ValueError, since FAISS itself only knows "similarity" and "mmr".

# Hybrid here means BM25 (keyword/exact-match) combined with FAISS (semantic
# similarity) via LangChain's EnsembleRetriever. This matters a lot for a
# study app: a plain vector search over "FIRST(E')" can miss the exact chunk
# if the embedding model doesn't treat that token specially, while BM25
# nails exact term/number/symbol matches; the ensemble gets both.
# """

# from langchain_community.retrievers import BM25Retriever
# from langchain_classic.retrievers import EnsembleRetriever
# from vectorstore.faiss_store import vector_store

# from config import BM25_WEIGHT, FETCH_K, LAMBDA_MULT, SEARCH_TYPE, TOP_K, VECTOR_WEIGHT


# class RetrieverService:

#     def __init__(self, vectorstore, documents=[]):
#         """
#         `documents` is only required when search_type="hybrid" (BM25 needs
#         the raw document list, not just the FAISS index). Pass
#         `vector_store.documents` from vectorstore/faiss_store.py.
#         """

#         self.vectorstore = vectorstore
#         self.documents = vector_store.documents or documents

#     def get_retriever(self, search_type=SEARCH_TYPE, **search_kwargs):

#         if search_type == "hybrid":
#             return self._get_hybrid_retriever(**search_kwargs)

#         if search_type == "mmr":
#             kwargs = {
#                 "k": TOP_K,
#                 "fetch_k": FETCH_K,
#                 "lambda_mult": LAMBDA_MULT,
#                 **search_kwargs,
#             }
#         else:
#             kwargs = {"k": TOP_K, **search_kwargs}

#         return self.vectorstore.as_retriever(
#             search_type=search_type,
#             search_kwargs=kwargs,
#         )

#     def _get_hybrid_retriever(self, k=None, **_ignored):

#         if not self.documents:
#             raise ValueError(
#                 "Hybrid search requires the raw document list "
#                 "(pass documents=vector_store.documents when constructing "
#                 "RetrieverService), since BM25 can't be built from the "
#                 "FAISS index alone."
#             )

#         top_k = k or TOP_K

#         bm25_retriever = BM25Retriever.from_documents(self.documents)
#         bm25_retriever.k = top_k

#         vector_retriever = self.vectorstore.as_retriever(
#             search_type="similarity",
#             search_kwargs={"k": top_k},
#         )

#         return EnsembleRetriever(
#             retrievers=[bm25_retriever, vector_retriever],
#             weights=[BM25_WEIGHT, VECTOR_WEIGHT],
#         )


"""
retriever.py

Why this file was rewritten
----------------------------
config.py already listed "hybrid" as a valid SEARCH_TYPE option (with a
comment describing it), but no code actually implemented it -- passing
search_type="hybrid" straight into FAISS's `.as_retriever()` would have
raised a ValueError, since FAISS itself only knows "similarity" and "mmr".

Hybrid here means BM25 (keyword/exact-match) combined with FAISS (semantic
similarity) via LangChain's EnsembleRetriever. This matters a lot for a
study app: a plain vector search over "FIRST(E')" can miss the exact chunk
if the embedding model doesn't treat that token specially, while BM25
nails exact term/number/symbol matches; the ensemble gets both.
"""

from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever

from config import BM25_WEIGHT, FETCH_K, LAMBDA_MULT, SEARCH_TYPE, TOP_K, VECTOR_WEIGHT


class RetrieverService:

    def __init__(self, vectorstore, documents=None):
        """
        `documents` is only required when search_type="hybrid" (BM25 needs
        the raw document list, not just the FAISS index). Pass the
        `.documents` list from whichever VectorStoreService instance you
        just built/loaded (e.g. `vs_service.documents` in app.py) --
        NOT the module-level `vector_store` singleton from faiss_store.py,
        which is a separate, independent instance and will be empty unless
        you specifically constructed it yourself.
        """

        self.vectorstore = vectorstore
        self.documents = documents or []

    def get_retriever(self, search_type=SEARCH_TYPE, **search_kwargs):

        if search_type == "hybrid":
            return self._get_hybrid_retriever(**search_kwargs)

        if search_type == "mmr":
            kwargs = {
                "k": TOP_K,
                "fetch_k": FETCH_K,
                "lambda_mult": LAMBDA_MULT,
                **search_kwargs,
            }
        else:
            kwargs = {"k": TOP_K, **search_kwargs}

        return self.vectorstore.as_retriever(
            search_type=search_type,
            search_kwargs=kwargs,
        )

    def _get_hybrid_retriever(self, k=None, **_ignored):

        if not self.documents:
            raise ValueError(
                "Hybrid search requires the raw document list "
                "(pass documents=vector_store.documents when constructing "
                "RetrieverService), since BM25 can't be built from the "
                "FAISS index alone."
            )

        top_k = k or TOP_K

        bm25_retriever = BM25Retriever.from_documents(self.documents)
        bm25_retriever.k = top_k

        vector_retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )

        return EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[BM25_WEIGHT, VECTOR_WEIGHT],
        )