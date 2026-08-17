from langchain_community.vectorstores import FAISS

from config import VECTOR_DB_PATH
from embeddings.ollama_embeddings import embeddings


class VectorStoreService:
    """
    Thin wrapper around a FAISS vector store.

    `self.documents` keeps a plain Python list of every Document that has
    been indexed. This is needed because the hybrid retriever (see
    retriever/retriever.py) builds a BM25Retriever from the *same* raw
    documents, and FAISS itself doesn't expose an easy "give me every
    document back" API once you've moved past `from_documents`.
    """

    def __init__(self):

        self.vectorstore = None
        self.documents = []

    def create(self, documents):
        """
        Create a new FAISS vector database from LangChain Documents,
        replacing anything previously indexed.
        """

        self.vectorstore = FAISS.from_documents(
            documents,
            embeddings.embedding_model
        )

        self.documents = list(documents)

        return self.vectorstore

    def add_documents(self, documents):
        """
        Add more documents to an existing index (e.g. the user uploads a
        second/third file after the first one has already been embedded).
        Falls back to `create` if nothing has been indexed yet.
        """

        if self.vectorstore is None:
            return self.create(documents)

        self.vectorstore.add_documents(documents)
        self.documents.extend(documents)

        return self.vectorstore

    def save(self, path=VECTOR_DB_PATH):

        self.vectorstore.save_local(path)

    def load(self, path=VECTOR_DB_PATH):

        self.vectorstore = FAISS.load_local(
            path,
            embeddings.embedding_model,
            allow_dangerous_deserialization=True
        )

        # Recover the raw documents from FAISS's internal docstore so the
        # hybrid (BM25) retriever still works after a reload from disk.
        self.documents = list(self.vectorstore.docstore._dict.values())

        return self.vectorstore


vector_store = VectorStoreService()
