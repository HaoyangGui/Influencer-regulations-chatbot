from unittest.mock import MagicMock

import pytest

from src import embedding
from src.embedding import EmbeddingModel


def test_embedding_model_uses_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = [
        MagicMock(tolist=MagicMock(return_value=[0.1, 0.2])),
        MagicMock(tolist=MagicMock(return_value=[0.3, 0.4])),
    ]

    def fake_sentence_transformer(model_name: str):
        assert model_name == "test-model"
        return fake_model

    monkeypatch.setattr(embedding, "SentenceTransformer", fake_sentence_transformer)

    model = EmbeddingModel(model_name="test-model")
    embeddings = model.embed_texts(["first", "second"])

    assert embeddings == [[0.1, 0.2], [0.3, 0.4]]
    fake_model.encode.assert_called_once()


def test_embedding_model_reuses_shared_sentence_transformer(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_model = MagicMock()
    fake_model.encode.return_value = [MagicMock(tolist=MagicMock(return_value=[0.1, 0.2]))]

    def fake_sentence_transformer(model_name: str):
        return fake_model

    monkeypatch.setattr(embedding, "SentenceTransformer", fake_sentence_transformer)
    EmbeddingModel._shared_model = None

    first_instance = EmbeddingModel(model_name="test-model")
    second_instance = EmbeddingModel(model_name="test-model")

    assert first_instance.model is second_instance.model
