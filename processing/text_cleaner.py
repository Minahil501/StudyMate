import re
from copy import deepcopy


class TextCleaner:
    """
    Cleans extracted document text while preserving metadata.
    """

    @staticmethod
    def clean(documents):

        cleaned_documents = []

        for doc in documents:

            new_doc = deepcopy(doc)

            text = new_doc.page_content

            # Remove excessive spaces
            text = re.sub(r"[ \t]+", " ", text)

            # Remove 3+ consecutive blank lines
            text = re.sub(r"\n{3,}", "\n\n", text)

            # Remove leading/trailing spaces
            text = text.strip()

            new_doc.page_content = text

            cleaned_documents.append(new_doc)

        return cleaned_documents