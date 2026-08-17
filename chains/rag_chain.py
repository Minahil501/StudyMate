# from operator import itemgetter


# from langchain_core.output_parsers import StrOutputParser


# from prompts.prompt import RAG_PROMPT

# from utils.formatter import format_docs



# def build_rag_chain(
#         retriever,
#         llm
# ):


#     rag_chain = (

#         {

#             "context":
#                 itemgetter("question")
#                 |
#                 retriever
#                 |
#                 format_docs,


#             "question":
#                 itemgetter("question")

#         }

#         |

#         RAG_PROMPT

#         |

#         llm

#         |

#         StrOutputParser()

#     )


#     return rag_chain

"""
rag_chain.py

Why this changed
-----------------
The previous version took a `retriever` and called it AS PART OF the chain
(`itemgetter("question") | retriever | format_docs`). That meant every
caller who needed the retrieved docs for anything else (e.g. showing
"Sources" in the chat UI) had to call the retriever a second time
separately -- which is exactly what app.py was doing on every chat message:
one retrieval to build the sources list, then a second, fully redundant
retrieval inside `rag_chain.invoke()` for the same question. With a
hybrid (BM25 + FAISS) retriever, that's 2x BM25 scoring, 2x embed_query()
calls to Ollama, 2x FAISS search, and 2x ensemble merge -- every message.

This version takes already-retrieved documents as input instead of a
retriever, so retrieval happens exactly ONCE per question, and the caller
reuses that same result both for the LLM context and for the sources UI.
"""

from operator import itemgetter

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

from prompts.prompt import RAG_PROMPT
from utils.formatter import format_docs


def build_rag_chain(llm):
    """
    Returns a chain that expects:
        {"question": str, "context": list[Document]}
    i.e. the caller retrieves once, then passes the retrieved docs straight
    in -- no retriever object needed here anymore.
    """

    rag_chain = (
        {
            "context": itemgetter("context") | RunnableLambda(format_docs),
            "question": itemgetter("question"),
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    return rag_chain