"""
huggingface_llm.py

This module initializes the Large Language Model (LLM)
used throughout StudyMate AI.

The rest of the project should import the LLM from here
instead of creating multiple LLM instances.

Why this was changed
---------------------
The original implementation talked to a locally-running Ollama server.
That works for local development, but Ollama has nothing to talk to on
platforms that only host the Streamlit process itself (e.g. Streamlit
Community Cloud) -- there's no way to also run a model server alongside
the app there.

This version talks to Hugging Face's hosted Inference API instead, via
`ChatHuggingFace` wrapping `HuggingFaceEndpoint`. `ChatHuggingFace` is a
real LangChain `BaseChatModel`, so `.with_structured_output(Schema)`
(used throughout chains/study_chains.py for Flashcards/Quiz/Notes) keeps
working unchanged -- no other file needed to change for this swap.

Requires a free Hugging Face access token set as the
HUGGINGFACEHUB_API_TOKEN environment variable (locally: a .env file;
on Streamlit Cloud: the app's Secrets). Get one at
https://huggingface.co/settings/tokens -- `huggingface_hub` (a
dependency of `langchain-huggingface`) picks it up automatically, no
code needed to read it explicitly.

Caveat worth knowing: unlike Ollama's native structured-output support,
`.with_structured_output()` here depends on the chosen model/provider
actually supporting tool calling on Hugging Face's Inference API. If you
swap LLM_MODEL (config.py) for a model without solid tool-calling
support, Flashcards/Quiz/Notes generation may degrade or fail even
though plain chat still works -- test those three features after
changing it.

Provider note: unlike embeddings/huggingface_embeddings.py, this does
NOT pin `provider="hf-inference"`. As of writing, HF's own hf-inference
serverless tier hosts essentially no text-generation/conversational
models at all -- check any model's providers with
`GET https://huggingface.co/api/models/<repo_id>?expand[]=inferenceProviderMapping`
before assuming hf-inference will work for a chat model. LLM_MODEL
below (Qwen2.5-7B-Instruct) is confirmed live on `featherless-ai`,
pinned explicitly rather than left as `provider=None`/"auto" so a
future model swap fails loudly (wrong-provider error) instead of
silently routing to a provider that requires separate billing on your
HF account. If you change LLM_MODEL, check its inferenceProviderMapping
first and update `provider` to match.
"""

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from config import GENERATIVE_TEMPERATURE, LLM_MODEL

# Hugging Face Inference API's context window and rate limits are governed
# by the model/provider, not something we configure client-side the way
# Ollama's num_ctx was. max_new_tokens caps the *output* length so a long
# generation (e.g. a full Notes page) doesn't get cut off mid-way.
MAX_NEW_TOKENS = 2048


class LLMService:
    """
    Creates and manages the Hugging Face-backed chat LLM.
    """

    def __init__(self, temperature: float = 0.2):
        endpoint = HuggingFaceEndpoint(
            repo_id=LLM_MODEL,
            provider="featherless-ai",
            temperature=temperature,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        self.llm = ChatHuggingFace(llm=endpoint)

    def get_llm(self):
        """
        Returns the initialized LLM.
        """
        return self.llm


# Singleton instances.
# - `llm`: deterministic (low temperature), used for Q&A where we want
#   consistent, grounded answers.
# - `generative_llm`: slightly higher temperature, used for Flashcards /
#   Quiz / Notes where some variety in phrasing is fine and even helpful
#   (e.g. avoiding near-identical flashcard wording).
llm = LLMService(temperature=0.2).get_llm()
generative_llm = LLMService(temperature=GENERATIVE_TEMPERATURE).get_llm()
