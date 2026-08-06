from unittest.mock import MagicMock

from src import translator
from src.translator import TranslationModel


def test_translation_model_calls_transformers_pipeline(monkeypatch) -> None:
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [{"translation_text": "Hello"}]
    monkeypatch.setattr(translator, "pipeline", lambda task, model, device: fake_pipeline)

    model = TranslationModel(model_name="test-model")
    result = model.translate("Hallo")

    assert result == "Hello"
    fake_pipeline.assert_called_once_with("Hallo", max_length=512)


def test_translation_model_reuses_shared_pipeline(monkeypatch) -> None:
    fake_pipeline = MagicMock()
    fake_pipeline.return_value = [{"translation_text": "Hello"}]
    monkeypatch.setattr(translator, "pipeline", lambda task, model, device: fake_pipeline)

    TranslationModel._shared_pipeline = None

    first_instance = TranslationModel(model_name="test-model")
    second_instance = TranslationModel(model_name="test-model")

    assert first_instance.pipeline is second_instance.pipeline
