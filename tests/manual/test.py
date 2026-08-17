from llm.ollama_llm import llm

response = llm.invoke("Introduce yourself in one sentence.")

print(response.content)