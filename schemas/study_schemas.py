"""
study_schemas.py

Pydantic models used as structured-output targets for the LLM.

Why this matters
-----------------
Asking an LLM to "generate 5 quiz questions" and printing whatever prose
comes back is fragile -- you can't reliably render it as interactive UI
cards, and any formatting drift breaks the app. Instead, every generative
feature (Flashcards / Quiz / Notes) binds the LLM to one of these schemas
via `llm.with_structured_output(Schema)`, so the response is guaranteed
to be a validated Python object, not a string to regex apart.
"""

from typing import Literal

from pydantic import BaseModel, Field


class Flashcard(BaseModel):
    question: str = Field(description="A short, focused question testing exactly one concept.")
    answer: str = Field(description="A concise, accurate answer to the question.")
    topic: str = Field(description="The topic/section this flashcard belongs to.")


class FlashcardSet(BaseModel):
    flashcards: list[Flashcard]


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(description="Exactly 4 answer options, in order.")
    correct_answer_index: int = Field(
        description="0-based index into `options` pointing at the correct answer."
    )
    explanation: str = Field(
        description="1-2 sentence explanation of why the correct answer is correct."
    )
    difficulty: Literal["easy", "medium", "hard"]


class QuizSet(BaseModel):
    questions: list[QuizQuestion]


class NoteSection(BaseModel):
    heading: str
    bullet_points: list[str]


class NotesOutput(BaseModel):
    title: str
    sections: list[NoteSection]
    key_takeaways: list[str] = Field(
        description="3-5 one-line takeaways summarizing the most important points."
    )
