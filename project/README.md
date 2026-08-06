# RAG Retriever Prototype

A minimal retrieval-augmented generation (RAG) prototype for Dutch influencer marketing regulations.

## Structure

- `data/raw/` - raw cleaned Markdown documents downloaded from the website.
- `data/processed/` - processed chunk JSON outputs.
- `vector_db/` - ChromaDB persistence folder.
- `src/` - implementation modules.
- `src/main.py` - entry point.

## Requirements

- Python 3.11+
- Install dependencies:

```bash
python -m pip install -r requirements.txt
```

## Usage

```bash
cd project
python src/main.py
```

The first run will download the example webpage, clean it, chunk it, translate the indexed content into English for embedding, generate embeddings, and build the vector store.

Subsequent runs are much faster because the vector store is reused if it already exists.

The retrieval step accepts questions in any language, translates the user query into English before embedding, and returns the top 5 chunks with both the original Dutch paragraph and its English translation.

The chatbot step builds a prompt from the most relevant chunks and sends it to an LLM through LiteLLM. This makes it easy to switch providers later by changing the `LLM_MODEL` environment variable.

## Environment

- To use OpenAI with LiteLLM, set `OPENAI_API_KEY` and optionally `LLM_MODEL=openai/gpt-3.5-turbo`.
- To use another LiteLLM-backed provider, set `LLM_MODEL` to the appropriate provider-specific model string.
- `LLM_API_BASE_URL` can also be used to override the endpoint for the chosen provider.

## Tests

Run pytest from the `project` directory:

```bash
python -m pytest -q tests
```

Run the small unit test suite with:

```bash
python -m pytest -q tests
```

## Notes

- Retrieval is based only on the downloaded page content.
- No LLM is invoked.
- The pipeline is modular and each stage is independent.
