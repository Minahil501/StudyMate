from loaders.loaders_factory import DocumentLoader
from processing.text_cleaner import TextCleaner
from processing.splitters import splitter

from vectorstore.faiss_store import vector_store
from retriever.retriever import RetrieverService
from chains.rag_chain import build_rag_chain
from llm.ollama_llm import llm

docs = DocumentLoader.load("uploads/sample.txt")
docs = TextCleaner.clean(docs)
chunks = splitter.split(docs)

db = vector_store.create(chunks)

retriever = RetrieverService(db, documents=vector_store.documents).get_retriever()

# build_rag_chain no longer takes a retriever -- just llm
rag = build_rag_chain(llm)

question = "What is derivation ?"

# retrieve once yourself
retrieved_docs = retriever.invoke(question)

# pass the question AND the already-retrieved docs in as "context"
response = rag.invoke({
    "question": question,
    "context": retrieved_docs,
})

print("=" * 80)
print(response)
print("=" * 80)