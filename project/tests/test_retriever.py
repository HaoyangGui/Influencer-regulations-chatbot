from pathlib import Path
from typing import Any

from unittest.mock import MagicMock
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever, RetrievalResult

import json as _json


class FakeCollection:
    def query(self, query_embeddings: list[list[float]], n_results: int, include: list[str], **kwargs: Any) -> dict[str, Any]:
        FakeCollection.last_kwargs = kwargs
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


def _make_retriever(monkeypatch: Any) -> Retriever:
    fake_embedding_model = MagicMock()
    fake_embedding_model.embed_text.return_value = [0.1, 0.0, 0.0]

    fake_vector_store = MagicMock()
    fake_vector_store.get_collection.return_value = FakeCollection()

    monkeypatch.setattr("src.retriever.EmbeddingModel", lambda: fake_embedding_model)
    monkeypatch.setattr("src.retriever.VectorStore", lambda *args, **kwargs: fake_vector_store)
    monkeypatch.setattr("src.retriever.TranslationModel", lambda: MagicMock())
    monkeypatch.setattr("src.retriever.LLMClient", lambda: MagicMock())
    return Retriever(vector_dir=Path("vector_db"))


def test_retrieve_without_document_filter_sends_no_where(monkeypatch: Any) -> None:
    FakeCollection.last_kwargs = {}
    retriever = _make_retriever(monkeypatch)
    retriever.retrieve("Question?", top_k=1)
    assert "where" not in FakeCollection.last_kwargs


def test_retrieve_with_document_filter_filters_by_document_name(monkeypatch: Any) -> None:
    """The RAG system can restrict retrieval to one individual document."""
    FakeCollection.last_kwargs = {}
    retriever = _make_retriever(monkeypatch)
    results = retriever.retrieve("Question?", top_k=1, document_name="Legal brief 2")
    assert FakeCollection.last_kwargs == {"where": {"document_name": "Legal brief 2"}}
    assert len(results) == 1


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
    answer_text, results, language = retriever.answer("What is the rule?", top_k=1)

    assert answer_text == "This is a final answer."
    assert language == "en"
    assert len(results) == 1
    fake_llm_client.generate_answer.assert_called_once()


def test_answer_reuses_provided_results(monkeypatch: Any) -> None:
    """answer() must reuse caller-supplied RAG chunks instead of re-fetching."""
    fake_llm_client = MagicMock()
    fake_llm_client.generate_answer.return_value = "Answer."

    fake_vector_store = MagicMock()
    fake_vector_store.get_collection.return_value = FakeCollection()

    def _no_retrieve(*args, **kwargs) -> None:
        raise AssertionError("retrieve() should not be called when results are supplied")

    monkeypatch.setattr("src.retriever.EmbeddingModel", lambda: MagicMock(embed_text=lambda t: [0.1, 0.0, 0.0]))
    monkeypatch.setattr("src.retriever.VectorStore", lambda *args, **kwargs: fake_vector_store)
    monkeypatch.setattr("src.retriever.TranslationModel", lambda: MagicMock())
    monkeypatch.setattr("src.retriever.LLMClient", lambda: fake_llm_client)

    retriever = Retriever(vector_dir=Path("vector_db"))
    retriever.retrieve = _no_retrieve  # type: ignore[method-assign]

    chunk = RetrievalResult(
        similarity=0.99,
        chunk_id="chunk-7",
        heading="Heading",
        paragraph_index=2,
        original_paragraph="Original Dutch text.",
        translated_paragraph="",
        source_name="source",
        source_url="https://example.com",
        quotation="Original Dutch text.",
        quotation_language="nl",
    )

    answer_text, results, language = retriever.answer("Wat is the rule?", top_k=1, results=[chunk], language="nl")

    assert answer_text == "Answer."
    assert language == "nl"
    assert results == [chunk]
    # The prompt built from the supplied chunk, so the LLM sees the same RAG basis.
    prompt = fake_llm_client.generate_answer.call_args.args[0]
    assert "chunk-7" in prompt
    assert "Dutch" in prompt  # dynamic target-language name


def test_select_quotation_prefers_exact_user_language() -> None:
    metadata = {
        "source_language": "nl",
        "translations": _json.dumps({"nl": "Originele nederlands", "en": "Original english", "zh": "中文原始"}, ensure_ascii=False),
    }
    text, lang = Retriever._select_quotation(metadata, "Originele nederlands", "zh")
    assert lang == "zh"
    assert text == "中文原始"


def test_select_quotation_does_not_force_english() -> None:
    # A French user query with only a source-language original + English fallback
    # must get the authentic source text, not an English bridge.
    metadata = {
        "source_language": "de",
        "translations": _json.dumps({"de": "Das deutsche Original.", "en": "The English translation."}, ensure_ascii=False),
    }
    text, lang = Retriever._select_quotation(metadata, "Das deutsche Original.", "fr")
    assert lang == "de"
    assert text == "Das deutsche Original."


def test_select_quotation_english_users_still_get_english() -> None:
    metadata = {
        "source_language": "nl",
        "translations": _json.dumps({"nl": "Dutch text.", "en": "English text."}, ensure_ascii=False),
    }
    text, lang = Retriever._select_quotation(metadata, "Nederland orig.", "en")
    assert lang == "en"
    assert text == "English text."


def test_select_quotation_legacy_english_no_translations_map() -> None:
    # Legacy persisted chunks used ``translated_paragraph``/``embedding_text_en``
    # with no ``translations`` map. An English user must get the English quote
    # (this is the exact stale-data bug: quote was coming back Dutch).
    metadata = {
        "source_language": "nl",
        "translated_paragraph": "Are you a video uploader? Then you need to keep to the rules.",
        "embedding_text_en": "Heading Are you a video uploader?",
    }
    text, lang = Retriever._select_quotation(metadata, "Original dutch paragraph.", "en")
    assert lang == "en"
    assert text == "Are you a video uploader? Then you need to keep to the rules."


def test_select_quotation_legacy_prefers_translated_paragraph_over_embedding() -> None:
    # ``translated_paragraph`` is the cleaner English source; it wins over the
    # heading-prefixed ``embedding_text_en``.
    metadata = {
        "source_language": "nl",
        "translated_paragraph": "Clean english translation.",
        "embedding_text_en": "Heading Clean english translation.",
    }
    text, lang = Retriever._select_quotation(metadata, "Nederland orig.", "en")
    assert lang == "en"
    assert text == "Clean english translation."


def test_select_quotation_legacy_non_english_not_forced_to_english() -> None:
    # A French user on legacy data (no translations map) must NOT receive the
    # stored English text; they get the authentic source-language original.
    metadata = {
        "source_language": "de",
        "translated_paragraph": "The English translation.",
        "embedding_text_en": "The English translation.",
    }
    text, lang = Retriever._select_quotation(metadata, "Das deutsche Original.", "fr")
    assert lang == "de"
    assert text == "Das deutsche Original."
