from loaders.loaders_factory import DocumentLoader
from processing.text_cleaner import TextCleaner
from processing.analyzer import DocumentAnalyzer
from processing.semantic_chunks import SemanticChunker
from processing.splitters import splitter
from processing.validator import ChunkValidator

docs = DocumentLoader.load("uploads/sample.pdf")

docs = TextCleaner.clean(docs)

analyzed = DocumentAnalyzer.analyze(docs)

semantic_docs = SemanticChunker.build(analyzed)

chunks = splitter.split(semantic_docs)

ChunkValidator.validate(chunks)