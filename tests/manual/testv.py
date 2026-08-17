from loaders.loaders_factory import DocumentLoader
from processing.text_cleaner import TextCleaner
from processing.headings import HeadingSplitter
from processing.splitters import splitter

from vectorstore.faiss_store import vector_store


# docs = DocumentLoader.load("uploads/sample.pdf")
docs = DocumentLoader.load("uploads/sample.txt")

docs = TextCleaner.clean(docs)

docs = HeadingSplitter.split(docs)

chunks = splitter.split(docs)

print(f"Chunks : {len(chunks)}")

db = vector_store.create(chunks)

print()

print("FAISS Created Successfully")

vector_store.save()

print()

print("Database Saved Successfully")