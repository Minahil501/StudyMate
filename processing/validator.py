from langchain_core.documents import Document

from config import CHUNK_SIZE, MAX_TABLE_CHUNK_SIZE


class ChunkValidator:

    @staticmethod
    def validate(chunks: list[Document]) -> int:
        """Print a validation report and return the number of errors found.

        Returns an int (rather than only printing) so tests/callers can
        assert `ChunkValidator.validate(chunks) == 0` instead of scraping
        stdout.
        """

        print("\n" + "=" * 80)
        print("VALIDATING CHUNKS")
        print("=" * 80)

        errors = 0

        for index, chunk in enumerate(chunks, start=1):

            is_table = chunk.metadata.get("content_type") == "table"

            # Empty chunk
            if not chunk.page_content.strip():
                print(f"[FAIL] Chunk {index} is empty.")
                errors += 1

            # Too large. Tables are intentionally allowed to exceed
            # CHUNK_SIZE (they are never split — see processing/splitters.py)
            # so they're checked against a much looser ceiling instead.
            size_limit = MAX_TABLE_CHUNK_SIZE if is_table else CHUNK_SIZE

            if len(chunk.page_content) > size_limit:
                print(
                    f"[FAIL] Chunk {index} exceeds "
                    f"{size_limit} characters "
                    f"({len(chunk.page_content)})."
                )
                errors += 1

            # Missing source
            if "source" not in chunk.metadata:
                print(f"[FAIL] Chunk {index} missing source metadata.")
                errors += 1

            # Missing page
            if "page" not in chunk.metadata:
                print(f"[FAIL] Chunk {index} missing page metadata.")
                errors += 1

        print()

        if errors == 0:
            print("[OK] All chunks passed validation.")
        else:
            print(f"[WARN] Found {errors} issue(s).")

        print("=" * 80)

        return errors