from loaders.loaders_factory import DocumentLoader

docs = DocumentLoader.load("uploads/sample.pdf")
doc1 =DocumentLoader.load("uploads/sample.txt")
doc2 = DocumentLoader.load("uploads/sample.docx")

print(f"Documents loaded: {len(docs)}")

print("="*50)

print(docs[1].page_content[:500])

print("="*50)

print(docs[0].metadata)

print("="*50)

print(doc1[0].page_content[:500])

print("="*50)   

print(doc1[0].metadata)

print("="*50)

print(doc2[0].page_content[:500])

print("="*50)
print(doc2[0].metadata)
print("="*50)

