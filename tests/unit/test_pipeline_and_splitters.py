from langchain_core.documents import Document

from config import CHUNK_SIZE
from pipeline import process_document, process_documents
from processing.splitters import splitter


class TestSplitterTableProtection:

    def test_large_table_is_never_split(self):
        """A table bigger than CHUNK_SIZE must still come out as ONE chunk."""

        huge_table = "| a | b |\n| --- | --- |\n" + "\n".join(
            f"| row{i} | value{i} |" for i in range(200)
        )
        assert len(huge_table) > CHUNK_SIZE

        doc = Document(
            page_content=huge_table,
            metadata={"source": "x", "page": 1, "content_type": "table"},
        )

        chunks = splitter.split([doc])

        assert len(chunks) == 1
        assert chunks[0].page_content == huge_table
        assert chunks[0].metadata["is_split"] is False

    def test_large_prose_section_is_split(self):
        """Sanity check: normal prose text still gets split as before."""

        long_text = "This is a sentence. " * 100
        assert len(long_text) > CHUNK_SIZE

        doc = Document(
            page_content=long_text,
            metadata={"source": "x", "page": 1, "content_type": "text"},
        )

        chunks = splitter.split([doc])

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= CHUNK_SIZE


class TestPipeline:

    def test_process_document_pdf_produces_chunks(self, sample_pdf_path):
        chunks = process_document(sample_pdf_path, validate=True)
        assert len(chunks) > 0
        assert any(c.metadata.get("content_type") == "table" for c in chunks)
        assert any(c.metadata.get("content_type") != "table" for c in chunks)

    def test_process_document_docx_produces_chunks(self, sample_docx_path):
        chunks = process_document(sample_docx_path, validate=True)
        assert len(chunks) > 0
        assert any(c.metadata.get("content_type") == "table" for c in chunks)

    def test_process_document_txt_produces_chunks(self, sample_txt_path):
        chunks = process_document(sample_txt_path, validate=True)
        assert len(chunks) > 0
        assert all(c.metadata.get("content_type") != "table" for c in chunks)

    def test_process_documents_merges_multiple_files(self, all_sample_paths):
        chunks = process_documents(all_sample_paths, validate=True)
        sources = {c.metadata.get("source") for c in chunks}
        assert sources == {"sample.pdf", "sample.docx", "sample.txt"}

    def test_every_chunk_has_source_and_page(self, all_sample_paths):
        chunks = process_documents(all_sample_paths)
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert "page" in chunk.metadata
