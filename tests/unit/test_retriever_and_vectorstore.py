from langchain_core.documents import Document

from retriever.retriever import RetrieverService
from vectorstore.faiss_store import VectorStoreService


def _build_indexed_store(fake_embeddings, monkeypatch):
    """
    VectorStoreService imports the real `embeddings` singleton (which wraps
    HuggingFaceEndpointEmbeddings) at module load time. We monkeypatch its
    underlying `embedding_model` attribute with our fake so `.create()`
    never touches a real Hugging Face API call.
    """

    import embeddings.huggingface_embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module.embeddings, "embedding_model", fake_embeddings)

    docs = [
        Document(
            page_content="FIRST and FOLLOW sets are used in LL(1) parsing.",
            metadata={"source": "compiler.pdf", "page": 1, "content_type": "text"},
        ),
        Document(
            page_content="The CAP theorem relates to distributed databases.",
            metadata={"source": "data_eng.docx", "page": 1, "content_type": "text"},
        ),
        Document(
            page_content="Panic mode recovery discards tokens until a synchronizing symbol.",
            metadata={"source": "compiler.pdf", "page": 2, "content_type": "text"},
        ),
    ]

    service = VectorStoreService()
    service.create(docs)
    return service, docs


class TestVectorStoreService:

    def test_create_indexes_documents(self, fake_embeddings, monkeypatch):
        service, docs = _build_indexed_store(fake_embeddings, monkeypatch)
        assert service.vectorstore is not None
        assert len(service.documents) == 3

    def test_add_documents_appends_to_existing_index(self, fake_embeddings, monkeypatch):
        service, docs = _build_indexed_store(fake_embeddings, monkeypatch)

        new_doc = Document(
            page_content="Batch processing operates on bounded datasets.",
            metadata={"source": "data_eng.docx", "page": 2, "content_type": "text"},
        )
        service.add_documents([new_doc])

        assert len(service.documents) == 4


class TestRetrieverService:

    def test_similarity_search_returns_results(self, fake_embeddings, monkeypatch):
        service, docs = _build_indexed_store(fake_embeddings, monkeypatch)
        retriever = RetrieverService(service.vectorstore).get_retriever(
            search_type="similarity", k=2
        )
        results = retriever.invoke("parsing question")
        assert len(results) == 2

    def test_mmr_search_returns_results(self, fake_embeddings, monkeypatch):
        service, docs = _build_indexed_store(fake_embeddings, monkeypatch)
        retriever = RetrieverService(service.vectorstore).get_retriever(
            search_type="mmr", k=2
        )
        results = retriever.invoke("parsing question")
        assert len(results) == 2

    def test_hybrid_search_combines_bm25_and_vector(self, fake_embeddings, monkeypatch):
        """
        This is the retriever type that config.py declared but the original
        code never implemented -- verifies it actually works end to end.
        """
        service, docs = _build_indexed_store(fake_embeddings, monkeypatch)
        retriever = RetrieverService(service.vectorstore, documents=service.documents).get_retriever(
            search_type="hybrid", k=2
        )
        results = retriever.invoke("What is the FIRST set used for in parsing?")
        assert len(results) > 0
        # BM25 should surface the compiler-theory doc for this keyword-heavy query
        sources = {r.metadata["source"] for r in results}
        assert "compiler.pdf" in sources

    def test_hybrid_search_without_documents_raises(self, fake_embeddings, monkeypatch):
        service, docs = _build_indexed_store(fake_embeddings, monkeypatch)
        retriever_service = RetrieverService(service.vectorstore)  # no documents passed

        try:
            retriever_service.get_retriever(search_type="hybrid")
            assert False, "expected ValueError"
        except ValueError as e:
            assert "Hybrid search requires" in str(e)
