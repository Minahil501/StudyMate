from loaders.loaders_factory import DocumentLoader
from processing.text_cleaner import TextCleaner
from processing.headings import HeadingSplitter
from processing.splitters import splitter

from embeddings.huggingface_embeddings import embeddings

doc1 = DocumentLoader.load("uploads/sample.pdf")
doc2 =DocumentLoader.load("uploads/sample.txt")
doc3= DocumentLoader.load("uploads/sample.docx")

documents = TextCleaner.clean(doc3)

documents = HeadingSplitter.split(documents)

chunks = splitter.split(documents)

print(f"Chunks : {len(chunks)}")

texts = [doc.page_content for doc in chunks]

vectors = embeddings.embed_documents(texts)

print()

print(f"Vectors Generated : {len(vectors)}")

print()

print(f"Embedding Dimension : {len(vectors[0])}")

print()

print(vectors[0][:15])