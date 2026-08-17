from loaders.loaders_factory import DocumentLoader


class TestPdfLoader:

    def test_loads_both_text_and_table_documents(self, sample_pdf_path):
        docs = DocumentLoader.load(sample_pdf_path)
        content_types = {d.metadata["content_type"] for d in docs}
        assert "text" in content_types
        assert "table" in content_types

    def test_table_is_valid_markdown(self, sample_pdf_path):
        docs = DocumentLoader.load(sample_pdf_path)
        tables = [d for d in docs if d.metadata["content_type"] == "table"]
        assert len(tables) == 1

        table_text = tables[0].page_content
        # header row, separator row, at least one data row
        lines = table_text.strip().splitlines()
        assert len(lines) >= 3
        assert lines[1].replace(" ", "").startswith("|---")

    def test_table_content_not_duplicated_in_text(self, sample_pdf_path):
        """
        Regression test: pdfplumber's page-level extract_text() re-includes
        table cell text unless the table's bounding box is explicitly
        excluded. This caught a real bug during development.
        """
        docs = DocumentLoader.load(sample_pdf_path)
        text_docs = [d for d in docs if d.metadata["content_type"] == "text"]
        combined_text = "\n".join(d.page_content for d in text_docs)

        # A distinctive table cell value that should NOT leak into prose text
        assert "epsilon }" not in combined_text or "FOLLOW Set" not in combined_text

    def test_every_document_has_required_metadata(self, sample_pdf_path):
        docs = DocumentLoader.load(sample_pdf_path)
        for doc in docs:
            assert doc.metadata["source"] == "sample.pdf"
            assert doc.metadata["file_type"] == "pdf"
            assert "page" in doc.metadata
            assert doc.metadata["content_type"] in ("text", "table")


class TestDocxLoader:

    def test_loads_both_text_and_table_documents(self, sample_docx_path):
        docs = DocumentLoader.load(sample_docx_path)
        content_types = {d.metadata["content_type"] for d in docs}
        assert "text" in content_types
        assert "table" in content_types

    def test_table_has_expected_row_count(self, sample_docx_path):
        docs = DocumentLoader.load(sample_docx_path)
        tables = [d for d in docs if d.metadata["content_type"] == "table"]
        assert len(tables) == 1

        lines = tables[0].page_content.strip().splitlines()
        # header + separator + 4 data rows (see generate_fixtures.py)
        assert len(lines) == 6

    def test_text_and_tables_come_out_in_document_order(self, sample_docx_path):
        docs = DocumentLoader.load(sample_docx_path)
        # fixture layout: text, table, text (see generate_fixtures.py)
        content_type_sequence = [d.metadata["content_type"] for d in docs]
        assert content_type_sequence == ["text", "table", "text"]


class TestTxtLoader:

    def test_loads_single_text_document(self, sample_txt_path):
        docs = DocumentLoader.load(sample_txt_path)
        assert len(docs) == 1
        assert docs[0].metadata["content_type"] == "text"
        assert docs[0].metadata["file_type"] == "txt"
        assert "Think2Code" in docs[0].page_content


class TestUnsupportedFileType:

    def test_raises_value_error(self, tmp_path):
        bad_file = tmp_path / "notes.xyz"
        bad_file.write_text("hello")

        try:
            DocumentLoader.load(str(bad_file))
            assert False, "expected ValueError"
        except ValueError as e:
            assert "Unsupported file type" in str(e)
