"""Manual smoke test for the LLM only."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from llm.huggingface_llm import llm


def main():
    prompt = "Introduce yourself in one sentence."
    response = llm.invoke(prompt)

    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("=" * 80)
    print(response.content)
    print("=" * 80)


if __name__ == "__main__":
    main()
