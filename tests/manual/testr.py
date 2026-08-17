"""Manual smoke test for the retrieval pipeline using `uploads/sample.txt`."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline import process_document
from retriever.retriever import RetrieverService
from vectorstore.faiss_store import VectorStoreService


def main():
    sample_path = PROJECT_ROOT / "uploads" / "sample.txt"

    if not sample_path.exists():
        raise FileNotFoundError(f"Sample file not found: {sample_path}")

    chunks = process_document(str(sample_path), validate=True)

    vector_store = VectorStoreService()
    db = vector_store.create(chunks)
    retriever = RetrieverService(db, documents=vector_store.documents).get_retriever(
        search_type="similarity",
        k=4,
    )

    query = "What is derivation?"
    results = retriever.invoke(query)

    print()
    print("=" * 80)
    print(f"Loaded: {sample_path.name}")
    print(f"Chunks created: {len(chunks)}")
    print(f"Query: {query}")
    print(f"Retrieved {len(results)} documents")
    print("=" * 80)

    for i, doc in enumerate(results, start=1):
        print(f"\nResult {i}")
        print("-" * 50)
        print(doc.metadata)
        print()
        print(doc.page_content)
        print()


if __name__ == "__main__":
    main()