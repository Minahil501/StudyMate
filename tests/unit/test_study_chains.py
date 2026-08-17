from langchain_core.documents import Document

from chains.study_chains import build_flashcard_chain, build_notes_chain, build_quiz_chain
from schemas.study_schemas import (
    Flashcard,
    FlashcardSet,
    NoteSection,
    NotesOutput,
    QuizQuestion,
    QuizSet,
)
from tests.unit.conftest import FakeStructuredLLM

SAMPLE_CHUNKS = [
    Document(
        page_content="FIRST and FOLLOW sets are used in LL(1) parsing.",
        metadata={"source": "compiler.pdf", "page": 1},
    )
]


class TestFlashcardChain:

    def test_returns_flashcard_set(self):
        expected = FlashcardSet(
            flashcards=[Flashcard(question="q1", answer="a1", topic="LL(1) parsing")]
        )
        run = build_flashcard_chain(FakeStructuredLLM(expected))

        result = run(SAMPLE_CHUNKS, num_flashcards=1)

        assert isinstance(result, FlashcardSet)
        assert result.flashcards[0].question == "q1"

    def test_context_gets_truncated_for_huge_document_sets(self):
        """
        Verifies the max_chars safeguard in study_chains._build_context
        actually kicks in, since we can't validate real LLM behavior on
        oversized input without a live model.
        """
        from chains.study_chains import _build_context

        huge_chunks = [
            Document(page_content="word " * 10000, metadata={"source": "x", "page": 1})
        ]
        context = _build_context(huge_chunks, max_chars=500)
        assert len(context) <= 550  # small buffer for the truncation marker
        assert "truncated" in context


class TestQuizChain:

    def test_returns_quiz_set_with_correct_shape(self):
        expected = QuizSet(
            questions=[
                QuizQuestion(
                    question="What is FIRST(E)?",
                    options=["a", "b", "c", "d"],
                    correct_answer_index=2,
                    explanation="Because...",
                    difficulty="medium",
                )
            ]
        )
        run = build_quiz_chain(FakeStructuredLLM(expected))

        result = run(SAMPLE_CHUNKS, num_questions=1, difficulty="medium")

        assert isinstance(result, QuizSet)
        assert len(result.questions[0].options) == 4
        assert 0 <= result.questions[0].correct_answer_index < 4


class TestNotesChain:

    def test_returns_notes_output(self):
        expected = NotesOutput(
            title="Compiler Theory",
            sections=[NoteSection(heading="LL(1) Parsing", bullet_points=["point 1"])],
            key_takeaways=["Know FIRST/FOLLOW sets"],
        )
        run = build_notes_chain(FakeStructuredLLM(expected))

        result = run(SAMPLE_CHUNKS, style="concise")

        assert isinstance(result, NotesOutput)
        assert result.title == "Compiler Theory"

    def test_topic_summary_style_uses_same_chain(self):
        """
        Topic Summaries is served by the Notes chain with style="topic_summary"
        rather than a separate pipeline -- this just confirms that style value
        flows through without error.
        """
        expected = NotesOutput(title="t", sections=[], key_takeaways=[])
        run = build_notes_chain(FakeStructuredLLM(expected))

        result = run(SAMPLE_CHUNKS, style="topic_summary")

        assert isinstance(result, NotesOutput)
