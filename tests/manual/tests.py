# from loaders.loaders_factory import DocumentLoader

# from processing.text_cleaner import TextCleaner

# from processing.splitters import splitter


# print("=" * 80)
# print("LOADING")
# print("=" * 80)

# docs = DocumentLoader.load("uploads/sample.pdf")

# print(f"Documents Loaded : {len(docs)}")

# print()

# print("=" * 80)
# print("CLEANING")
# print("=" * 80)

# docs = TextCleaner.clean(docs)

# print(f"Documents After Cleaning : {len(docs)}")

# print()

# print("=" * 80)
# print("SPLITTING")
# print("=" * 80)

# chunks = splitter.split(docs)

# print(f"Chunks Created : {len(chunks)}")

# print()

# for chunk in chunks[:5]:

#     print("=" * 80)

#     print(chunk.metadata)

#     print()

#     print(chunk.page_content)

#     print()


from loaders.loaders_factory import DocumentLoader
from processing.text_cleaner import TextCleaner
from processing.analyzer import DocumentAnalyzer
from processing.semantic_chunks import SemanticChunker
from processing.splitters import splitter


docs = DocumentLoader.load("uploads/sample.pdf")
doc1 =DocumentLoader.load("uploads/sample.txt")
doc2 = DocumentLoader.load("uploads/sample.docx")

docs = TextCleaner.clean(doc2)

analyzed = DocumentAnalyzer.analyze(doc2)

semantic = SemanticChunker.build(analyzed)

chunks = splitter.split(semantic)

print("=" * 80)
print("FINAL CHUNKS")
print("=" * 80)

print(f"Chunks : {len(chunks)}")

for chunk in chunks[:5]:

    print()

    print(chunk.metadata)

    print()

    print(chunk.page_content)

    print("-" * 80)