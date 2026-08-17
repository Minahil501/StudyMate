from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import (
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    SEPARATORS,
)


class RecursiveSplitter:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(

            chunk_size=CHUNK_SIZE,

            chunk_overlap=CHUNK_OVERLAP,

            separators=SEPARATORS,

            length_function=len,

            is_separator_regex=False,

        )

    def split(self, documents):

        final_chunks = []

        chunk_id = 1

        for doc in documents:

            # ------------------------------------
            # Tables are never split, regardless of size.
            # Breaking a Markdown table mid-row corrupts it (the LLM
            # can no longer tell which cell belongs to which column),
            # so a whole table is always kept as a single chunk.
            # ------------------------------------

            if doc.metadata.get("content_type") == "table":

                metadata = doc.metadata.copy()

                metadata["chunk_id"] = chunk_id
                metadata["chunk_size"] = len(doc.page_content)
                metadata["is_split"] = False

                doc.metadata = metadata

                final_chunks.append(doc)

                chunk_id += 1

                continue

            # ------------------------------------
            # Small semantic section
            # ------------------------------------

            if len(doc.page_content) <= CHUNK_SIZE:

                metadata = doc.metadata.copy()

                metadata["chunk_id"] = chunk_id
                metadata["chunk_size"] = len(doc.page_content)
                metadata["is_split"] = False

                doc.metadata = metadata

                final_chunks.append(doc)

                chunk_id += 1

                continue

            # ------------------------------------
            # Large semantic section
            # ------------------------------------

            pieces = self.splitter.split_documents([doc])

            for piece in pieces:

                metadata = piece.metadata.copy()

                metadata["chunk_id"] = chunk_id
                metadata["chunk_size"] = len(piece.page_content)
                metadata["is_split"] = True

                piece.metadata = metadata

                final_chunks.append(piece)

                chunk_id += 1

        total = len(final_chunks)

        for chunk in final_chunks:

            chunk.metadata["total_chunks"] = total

        return final_chunks


splitter = RecursiveSplitter()