"""
conftest.py

Shared fixtures for the test suite.

Why fakes instead of real Ollama calls
----------------------------------------
These tests need to run in CI / on a machine without Ollama installed or
running. Everything that depends on real model output (embedding quality,
generated flashcard wording, etc.) is out of scope for unit tests here --
that's an integration/manual-QA concern. What IS tested automatically:
loading, table extraction, chunking rules, retriever wiring, chain
composition, and schema validation.
"""

import sys
from pathlib import Path

import pytest

# Make the project root importable when running `pytest` from the repo root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"


@pytest.fixture
def sample_pdf_path():
    return str(FIXTURES_DIR / "sample.pdf")


@pytest.fixture
def sample_docx_path():
    return str(FIXTURES_DIR / "sample.docx")


@pytest.fixture
def sample_txt_path():
    return str(FIXTURES_DIR / "sample.txt")


@pytest.fixture
def all_sample_paths(sample_pdf_path, sample_docx_path, sample_txt_path):
    return [sample_pdf_path, sample_docx_path, sample_txt_path]


from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """
    Deterministic fake embedding model: same text always maps to the same
    vector, different text maps to different vectors (via hashing), so
    FAISS similarity search behaves consistently across test runs without
    ever calling a real Ollama server.

    Must subclass `Embeddings` (not just duck-type embed_documents/
    embed_query) -- langchain_community's FAISS wrapper does an explicit
    `isinstance(self.embedding_function, Embeddings)` check and silently
    falls back to treating anything else as a bare callable.
    """

    def __init__(self, dim: int = 16):
        self.dim = dim

    def embed_documents(self, texts):
        return [self._vector(t) for t in texts]

    def embed_query(self, text):
        return self._vector(text)

    def _vector(self, text: str):
        import numpy as np

        seed = abs(hash(text)) % (2**32)
        return np.random.RandomState(seed).rand(self.dim).tolist()


@pytest.fixture
def fake_embeddings():
    return FakeEmbeddings()


class FakeStructuredLLM:
    """
    Stands in for `llm.with_structured_output(Schema)`. Always returns the
    `fake_output` object it was constructed with, regardless of the prompt
    it receives -- used to test that study_chains.py wires prompts and
    schemas together correctly without needing a real model call.
    """

    def __init__(self, fake_output):
        self.fake_output = fake_output

    def with_structured_output(self, schema):
        from langchain_core.runnables import RunnableLambda

        return RunnableLambda(lambda _prompt_value: self.fake_output)
