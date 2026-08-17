# tests/manual/

These are exploratory, module-by-module smoke-test scripts written during
early development (no assertions -- they just print output for manual
inspection). They were originally directly inside `tests/`, where pytest's
default discovery (`test*.py`) would pick them up and execute them as if
they were real tests -- including running the full pipeline against
`uploads/sample.pdf` and hitting a live Ollama server as a side effect of
just *importing* the module during test collection.

They're kept here for reference (some show how individual pieces were
first wired together) but are **not** part of the automated suite. The
automated suite lives in `tests/unit/` and uses fake embeddings/LLMs so it
runs without a live Ollama server.

To run one of these manually (requires `ollama serve` running locally
with the models from `config.py` pulled):

```
python tests/manual/testc.py
```
