# """
# Semantic Chunk Builder
# ======================

# Purpose
# -------
# Build meaningful document sections from analyzed blocks.

# Input:
# ------
# Analyzer Output (list[Document])

# Output:
# -------
# Semantic Sections (list[Document])

# Rules
# -----
# - Title belongs to the first section.
# - Heading starts a new section.
# - Numbered heading starts a new section.
# - Paragraphs belong to current section.
# - Bullets belong to current section.
# """

# from langchain_core.documents import Document


# class SemanticChunker:

#     @staticmethod
#     def build(documents: list[Document]):

#         chunks = []

#         current_lines = []

#         current_metadata = None

#         current_title = None

#         current_section = "Untitled"

#         block_count = 0

#         for doc in documents:

#             block_type = doc.metadata["block_type"]

#             text = doc.page_content

#             # ---------------------------------------
#             # Store page title
#             # ---------------------------------------

#             if block_type == "title":

#                 current_title = text

#                 continue

#             # ---------------------------------------
#             # New section starts
#             # ---------------------------------------

#             if block_type in ("heading", "numbered_heading"):

#                 # Save previous section

#                 if current_lines:

#                     metadata = current_metadata.copy()

#                     metadata["section"] = current_section
#                     metadata["title"] = current_title
#                     metadata["block_count"] = block_count

#                     chunks.append(

#                         Document(

#                             page_content="\n".join(current_lines),

#                             metadata=metadata

#                         )

#                     )

#                 # Start new section

#                 current_section = text

#                 current_lines = []

#                 block_count = 0

#                 current_metadata = doc.metadata.copy()

#                 # Attach page title once

#                 if current_title:

#                     current_lines.append(current_title)

#                     current_lines.append("")

#                 current_lines.append(text)

#                 block_count += 1

#                 continue

#             # ---------------------------------------
#             # Paragraph / Bullet
#             # ---------------------------------------

#             if current_metadata is None:

#                 current_metadata = doc.metadata.copy()

#             current_lines.append(text)

#             block_count += 1

#         # ---------------------------------------
#         # Save last section
#         # ---------------------------------------

#         if current_lines:

#             metadata = current_metadata.copy()

#             metadata["section"] = current_section
#             metadata["title"] = current_title
#             metadata["block_count"] = block_count

#             chunks.append(

#                 Document(

#                     page_content="\n".join(current_lines),

#                     metadata=metadata

#                 )

#             )

#         return chunks



"""
Semantic Chunk Builder
======================

Purpose
-------
Build meaningful document sections from analyzed blocks.

Input:
------
Analyzer Output (list[Document])

Output:
-------
Semantic Sections (list[Document])

Rules
-----
- Title belongs to the first section.
- Heading starts a new section.
- Numbered heading starts a new section.
- Paragraphs belong to current section.
- Bullets belong to current section.

Why page tracking changed
--------------------------
Previously, a section's metadata (including `page`) was captured ONCE --
either from the heading that started the section, or from the first
paragraph if there was no heading -- and never updated again as more
lines were appended. For any section that spans more than one PDF page
(the common case, since headings are sparse), every chunk from that
section reported a single stale page number that only matched the FIRST
line, even though later paragraphs in the same chunk came from a
different page. This version tracks every page actually seen while
building a section and reports the true start/end page.
"""

from langchain_core.documents import Document


class SemanticChunker:

    @staticmethod
    def build(documents: list[Document]):

        chunks = []

        current_lines = []

        current_metadata = None

        current_title = None

        current_section = "Untitled"

        block_count = 0

        # Every page number actually seen while building the CURRENT section.
        current_pages: set = set()

        def _finalize_page_metadata(metadata: dict) -> dict:
            """Replace the single stale `page` with the true range seen."""

            pages = sorted(p for p in current_pages if p is not None)

            if not pages:
                metadata["page"] = metadata.get("page", 0)
            elif len(pages) == 1:
                metadata["page"] = pages[0]
            else:
                # Section spans multiple pages -- report the range rather
                # than silently picking one.
                metadata["page"] = pages[0]
                metadata["page_end"] = pages[-1]

            return metadata

        for doc in documents:

            block_type = doc.metadata["block_type"]

            text = doc.page_content

            # ---------------------------------------
            # Store page title
            # ---------------------------------------

            if block_type == "title":

                current_title = text

                continue

            # ---------------------------------------
            # New section starts
            # ---------------------------------------

            if block_type in ("heading", "numbered_heading"):

                # Save previous section

                if current_lines:

                    metadata = current_metadata.copy()

                    metadata["section"] = current_section
                    metadata["title"] = current_title
                    metadata["block_count"] = block_count
                    metadata = _finalize_page_metadata(metadata)

                    chunks.append(

                        Document(

                            page_content="\n".join(current_lines),

                            metadata=metadata

                        )

                    )

                # Start new section

                current_section = text

                current_lines = []

                block_count = 0

                current_metadata = doc.metadata.copy()
                current_pages = {doc.metadata.get("page")}

                # Attach page title once

                if current_title:

                    current_lines.append(current_title)

                    current_lines.append("")

                current_lines.append(text)

                block_count += 1

                continue

            # ---------------------------------------
            # Paragraph / Bullet
            # ---------------------------------------

            if current_metadata is None:

                current_metadata = doc.metadata.copy()

            current_pages.add(doc.metadata.get("page"))

            current_lines.append(text)

            block_count += 1

        # ---------------------------------------
        # Save last section
        # ---------------------------------------

        if current_lines:

            metadata = current_metadata.copy()

            metadata["section"] = current_section
            metadata["title"] = current_title
            metadata["block_count"] = block_count
            metadata = _finalize_page_metadata(metadata)

            chunks.append(

                Document(

                    page_content="\n".join(current_lines),

                    metadata=metadata

                )

            )

        return chunks