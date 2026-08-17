import re
from copy import deepcopy
from langchain_core.documents import Document


class HeadingSplitter:

    heading_pattern = re.compile(
        r"^(Week\s+\d+|Chapter\s+\d+|Unit\s+\d+|Section\s+\d+|[A-Z][A-Za-z0-9\s]{2,40})$",
        re.MULTILINE,
    )

    @classmethod
    def split(cls, documents):

        split_docs = []

        for doc in documents:

            text = doc.page_content

            matches = list(cls.heading_pattern.finditer(text))

            if not matches:
                split_docs.append(doc)
                continue

            for i, match in enumerate(matches):

                start = match.start()

                end = (
                    matches[i + 1].start()
                    if i + 1 < len(matches)
                    else len(text)
                )

                chunk = text[start:end].strip()

                metadata = deepcopy(doc.metadata)
                metadata["heading"] = match.group().strip()

                split_docs.append(
                    Document(
                        page_content=chunk,
                        metadata=metadata,
                    )
                )

        return split_docs