from loaders.loaders_factory import DocumentLoader

from processing.text_cleaner import TextCleaner

from processing.analyzer import DocumentAnalyzer


docs = DocumentLoader.load("uploads/sample.docx")

docs = TextCleaner.clean(docs)

blocks = DocumentAnalyzer.analyze(docs)

for block in blocks:

    print("=" * 80)

    print(block.metadata["block_type"])

    print()

    print(block.page_content)