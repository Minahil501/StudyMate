"""
study_chains.py

Chain builders for the generative study features: Flashcards, Quiz, and
Notes/Topic-Summaries. Kept separate from chains/rag_chain.py because
these are a different shape of problem:

- rag_chain.py answers ONE question from the TOP-K most relevant chunks
  (precision-focused retrieval).
- These chains generate comprehensive study material from the (roughly)
  FULL set of chunks for a document, since a flashcard set that only
  covers the top-4 retrieved chunks would miss most of the material.

Known limitation (documented on purpose, not hidden): for very large
documents, "full context" can exceed the model's context window. The
`max_chars` truncation below is a simple safeguard for a one-day build;
a production version would map-reduce over chunk groups instead of
truncating. This is a good "next step" to mention in interviews.
"""

from config import (
    DEFAULT_FLASHCARD_COUNT,
    DEFAULT_NOTES_STYLE,
    DEFAULT_QUIZ_DIFFICULTY,
    DEFAULT_QUIZ_QUESTION_COUNT,
    GENERATIVE_TEMPERATURE,
)
from prompts.prompt import FLASHCARD_PROMPT, NOTES_PROMPT, QUIZ_PROMPT
from schemas.study_schemas import FlashcardSet, NotesOutput, QuizSet
from utils.formatter import format_docs


def _build_context(chunks, max_chars: int = 8000) -> str:
    """
    Join chunks into one context string for generation, with a hard
    character cap so we don't blow past the model's context window on a
    large document. See the module docstring for why this is a
    known/documented simplification rather than a silent limitation.
    """

    context = format_docs(chunks)

    if len(context) > max_chars:
        context = context[:max_chars] + "\n\n[... additional content truncated ...]"

    return context


def build_flashcard_chain(llm):
    """Returns a callable: (chunks, num_flashcards=...) -> FlashcardSet"""

    structured_llm = llm.with_structured_output(FlashcardSet)
    chain = FLASHCARD_PROMPT | structured_llm

    def run(chunks, num_flashcards: int = DEFAULT_FLASHCARD_COUNT) -> FlashcardSet:
        return chain.invoke(
            {
                "context": _build_context(chunks),
                "num_flashcards": num_flashcards,
            }
        )

    return run


def build_quiz_chain(llm):
    """Returns a callable: (chunks, num_questions=..., difficulty=...) -> QuizSet"""

    structured_llm = llm.with_structured_output(QuizSet)
    chain = QUIZ_PROMPT | structured_llm

    def run(
        chunks,
        num_questions: int = DEFAULT_QUIZ_QUESTION_COUNT,
        difficulty: str = DEFAULT_QUIZ_DIFFICULTY,
    ) -> QuizSet:
        return chain.invoke(
            {
                "context": _build_context(chunks),
                "num_questions": num_questions,
                "difficulty": difficulty,
            }
        )

    return run


def build_notes_chain(llm):
    """Returns a callable: (chunks, style=...) -> NotesOutput

    style="topic_summary" is how the "Topic Summaries" feature is served --
    it's the same chain with a different prompt instruction rather than a
    separate pipeline, since the underlying task (organize context into
    labeled sections) is identical.
    """

    structured_llm = llm.with_structured_output(NotesOutput)
    chain = NOTES_PROMPT | structured_llm

    def run(chunks, style: str = DEFAULT_NOTES_STYLE, document_name: str = "Selected document") -> NotesOutput:
        return chain.invoke(
            {
                "context": _build_context(chunks),
                "style": style,
                "document_name": document_name,
            }
        )

    return run
