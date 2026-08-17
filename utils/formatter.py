from langchain_core.documents import Document


def format_docs(docs: list[Document]):

    formatted = []


    for doc in docs:

        page = doc.metadata.get(
            "page",
            "unknown"
        )

        source = doc.metadata.get(
            "source",
            "unknown"
        )

        formatted.append(

            f"""
Source: {source}
Page: {page}

Content:
{doc.page_content}

"""

        )


    return "\n\n-----------------\n\n".join(formatted)