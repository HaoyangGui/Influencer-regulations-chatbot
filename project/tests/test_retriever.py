from pathlib import Path
from typing import Any

from unittest.mock import MagicMock

from src.retriever import Retriever


class FakeCollection:
    def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str]) -> dict[str, Any]:
        return {
            "metadatas": [[{"chunk_id": "chunk-1", "heading": "Test heading", "paragraph_index": 1, "source_name": "test", "source_url": "https://example.com"}]],
            "documents": [["Original paragraph text."]],
            "embeddings": [[[0.1, 0.0, 0.0]]],
        }


def test_retriever_returns_parsed_results(monkeypatch: Any) -> None:
    fake_embedding_model = MagicMock()
    fake_embedding_model.embed_text.return_value = [0.1, 0.0, 0.0]

    fake_vector_store = MagicMock()
    fake_vector_store.get_collection.return_value = FakeCollection()

    fake_translator = MagicMock()
    fake_translator.translate.return_value = "What is the rule?"

    fake_llm_client = MagicMock()
    fake_llm_client.generate_answer.return_value = "Fake answer."

    monkeypatch.setattr("src.retriever.EmbeddingModel", lambda: fake_embedding_model)
    monkeypatch.setattr("src.retriever.VectorStore", lambda *args, **kwargs: fake_vector_store)
    monkeypatch.setattr("src.retriever.TranslationModel", lambda: fake_translator)
    monkeypatch.setattr("src.retriever.LLMClient", lambda: fake_llm_client)

    retriever = Retriever(vector_dir=Path("vector_db"))
    results = retriever.retrieve("What is the rule?", top_k=1)

    assert len(results) == 1
    result = results[0]
    assert result.chunk_id == "chunk-1"
    assert result.heading == "Test heading"
    assert result.paragraph_index == 1
    assert result.original_paragraph == "Original paragraph text."
    assert abs(result.similarity - 0.01) < 1e-6


def test_retriever_answer_uses_llm(monkeypatch: Any) -> None:
    fake_embedding_model = MagicMock()
    fake_embedding_model.embed_text.return_value = [0.1, 0.0, 0.0]

    fake_vector_store = MagicMock()
    fake_vector_store.get_collection.return_value = FakeCollection()

    fake_translator = MagicMock()
    fake_translator.translate.return_value = "What is the rule?"

    fake_llm_client = MagicMock()
    fake_llm_client.generate_answer.return_value = "This is a final answer."

    monkeypatch.setattr("src.retriever.EmbeddingModel", lambda: fake_embedding_model)
    monkeypatch.setattr("src.retriever.VectorStore", lambda *args, **kwargs: fake_vector_store)
    monkeypatch.setattr("src.retriever.TranslationModel", lambda: fake_translator)
    monkeypatch.setattr("src.retriever.LLMClient", lambda: fake_llm_client)

    retriever = Retriever(vector_dir=Path("vector_db"))
    answer_text, results = retriever.answer("What is the rule?", top_k=1)

    assert answer_text == "This is a final answer."
    assert len(results) == 1
    fake_llm_client.generate_answer.assert_called_once()
