"""
Document Analyzer
=================

Purpose
-------
Analyze every line of every document and classify it into a semantic block.

The analyzer DOES NOT create chunks.

It only labels document structure for the Semantic Chunk Builder.

Supported block types
---------------------

- title
- heading
- numbered_heading
- bullet
- paragraph

Each analyzed block preserves metadata and adds:

- page
- line
- block_id
- block_type
"""

import re

from langchain_core.documents import Document


class DocumentAnalyzer:

    # Compiled once at import time instead of being rebuilt by re.match()
    # on every single line of every document. For a 50-page document
    # that's the difference between compiling these patterns a handful
    # of times vs. potentially thousands of times.
    _TITLE_PREFIX_RE = re.compile(r"^(Week|Chapter|Lecture|Unit)\s+\d+")
    _NUMBERED_HEADING_RE = re.compile(r"^\d+(\.\d+)*[\.)]?\s+[A-Z]")
    _BULLET_RE = re.compile(r"^(\u2022|-|\*)\s+")
    _STARTS_WITH_DIGIT_RE = re.compile(r"^\d")

    @staticmethod
    def analyze(documents: list[Document]):

        analyzed = []

        block_id = 1

        for doc in documents:

            page = doc.metadata.get("page", 0)

            lines = doc.page_content.splitlines()

            for line_number, line in enumerate(lines):

                line = line.strip()

                if not line:
                    continue

                block_type = DocumentAnalyzer.get_block_type(line)

                metadata = doc.metadata.copy()

                metadata["page"] = page
                metadata["line"] = line_number
                metadata["block_id"] = block_id
                metadata["block_type"] = block_type

                analyzed.append(

                    Document(

                        page_content=line,

                        metadata=metadata

                    )

                )

                block_id += 1

        return analyzed

    # ------------------------------------------------------------

    @staticmethod
    def get_block_type(text: str):

        if DocumentAnalyzer.is_title(text):
            return "title"

        if DocumentAnalyzer.is_numbered_heading(text):
            return "numbered_heading"

        if DocumentAnalyzer.is_heading(text):
            return "heading"

        if DocumentAnalyzer.is_bullet(text):
            return "bullet"

        return "paragraph"

    # ------------------------------------------------------------

    @staticmethod
    def is_title(text: str):

        words = text.split()

        if len(words) > 15:
            return False

        if DocumentAnalyzer._TITLE_PREFIX_RE.match(text):
            return True

        if text.isupper() and len(words) <= 10:
            return True

        return False

    # ------------------------------------------------------------

    @staticmethod
    def is_numbered_heading(text: str):

        return bool(DocumentAnalyzer._NUMBERED_HEADING_RE.match(text))

    # ------------------------------------------------------------

    @staticmethod
    def is_heading(text: str):

        words = text.split()

        if len(words) == 0:
            return False

        if len(words) > 12:
            return False

        if text.endswith("."):
            return False

        if DocumentAnalyzer.is_bullet(text):
            return False

        if DocumentAnalyzer._STARTS_WITH_DIGIT_RE.match(text):
            return False

        uppercase_words = sum(

            word[0].isupper()

            for word in words

            if word

        )

        return uppercase_words >= max(1, len(words) * 0.6)

    # ------------------------------------------------------------

    @staticmethod
    def is_bullet(text: str):

        return bool(DocumentAnalyzer._BULLET_RE.match(text))