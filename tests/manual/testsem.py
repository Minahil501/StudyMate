from loaders.loaders_factory import DocumentLoader
from processing.text_cleaner import TextCleaner
from processing.analyzer import DocumentAnalyzer
from processing.semantic_chunks import SemanticChunker

docs = DocumentLoader.load("uploads/sample.pdf")

docs = TextCleaner.clean(docs)

analyzed = DocumentAnalyzer.analyze(docs)

sections = SemanticChunker.build(analyzed)

print("=" * 80)
print(f"Semantic Sections: {len(sections)}")
print("=" * 80)

for i, section in enumerate(sections[:5], 1):

    print(f"\nSECTION {i}")
    print("-" * 60)
    print(section.metadata)
    print()
    print(section.page_content)
    print()