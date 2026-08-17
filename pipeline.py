"""
pipeline.py

Single source of truth for the document ingestion pipeline.

Why this file exists
---------------------
Previously, the load -> clean -> analyze -> chunk -> validate sequence was
copy-pasted across several test scripts (tests/testr.py, tests/testc.py,
processing/validate.py's bottom half, ...). Every copy diverged slightly,
which is exactly how pipelines silently drift out of sync. app.py and the
test suite should both call ONE function.

Why tables are routed differently
----------------------------------
`DocumentAnalyzer` classifies text line-by-line (title / heading / bullet /
paragraph) and `SemanticChunker` reassembles those lines into sections.
Both assume prose. Running a Markdown table through them would split the
table onto separate "paragraph" lines and destroy its structure before it
even reaches the splitter's table-protection check. So:

    text documents  -> analyzer -> semantic chunker -> splitter
    table documents -> splitter directly (splitter keeps whole tables intact)

Why process_documents() now runs files concurrently
-----------------------------------------------------
Each file's load/clean/analyze/chunk work is completely independent of
every other file's -- there's no shared state until the chunks get merged
and handed to the embedder. Previously this was a plain sequential `for`
loop, so uploading 3-4 files meant paying for PDF parsing + regex analysis
on each one back-to-back. That work is a mix of I/O (pdfplumber reading
the file) and CPU (regex classification in DocumentAnalyzer), so a thread
pool gives a real speedup for multi-file uploads without needing
multiprocessing. Embedding still happens once, in one batch, after all
files are processed -- that part was already efficient and is unchanged.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from loaders.loaders_factory import DocumentLoader
from processing.analyzer import DocumentAnalyzer
from processing.semantic_chunks import SemanticChunker
from processing.splitters import splitter
from processing.text_cleaner import TextCleaner
from processing.validator import ChunkValidator

# How many files to process in parallel. Kept modest by default -- this is
# bounded more by disk I/O and single-core regex work than by CPU count, so
# there's little benefit past a handful of concurrent files even on a
# many-core machine.
MAX_INGEST_WORKERS = 4


def process_document(file_path: str, validate: bool = False):
    """
    Run the full ingestion pipeline for a single file and return
    final chunks (list[Document]) ready for embedding.
    """

    documents = DocumentLoader.load(file_path)
    documents = TextCleaner.clean(documents)

    text_docs = [d for d in documents if d.metadata.get("content_type") != "table"]
    table_docs = [d for d in documents if d.metadata.get("content_type") == "table"]

    analyzed = DocumentAnalyzer.analyze(text_docs) if text_docs else []
    semantic_docs = SemanticChunker.build(analyzed) if analyzed else []

    # Tables go straight into the splitter, which passes them through
    # untouched (see processing/splitters.py content_type == "table" check).
    chunks = splitter.split(semantic_docs + table_docs)

    if validate:
        ChunkValidator.validate(chunks)

    return chunks


def process_documents(file_paths: list[str], validate: bool = False):
    """
    Run process_document over multiple files concurrently and merge the
    results, preserving the input file order in the output (so chunk
    ordering doesn't depend on which file happened to finish loading
    first).
    """

    if len(file_paths) <= 1:
        # No point spinning up a thread pool for a single file.
        all_chunks = [
            process_document(file_paths[0], validate=False)
        ] if file_paths else []
    else:
        results = [None] * len(file_paths)

        with ThreadPoolExecutor(max_workers=min(MAX_INGEST_WORKERS, len(file_paths))) as pool:
            future_to_index = {
                pool.submit(process_document, file_path, False): index
                for index, file_path in enumerate(file_paths)
            }

            for future in as_completed(future_to_index):
                index = future_to_index[future]
                # Let exceptions from a single bad file propagate -- silently
                # dropping a failed file's chunks would be worse than a
                # visible error, since the student would get incomplete
                # study material with no indication why.
                results[index] = future.result()

        all_chunks = results

    merged_chunks = [chunk for file_chunks in all_chunks for chunk in file_chunks]

    if validate:
        ChunkValidator.validate(merged_chunks)

    return merged_chunks