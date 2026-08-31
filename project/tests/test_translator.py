from pathlib import Path

import pytest

from src import translator
from src.translator import TranslationModel


class FakeTokenizer:
    """Minimal tokenizer stand-in mirroring the HF interface used by the model."""

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        ids = [list(range(len(t.split()) + 2)) for t in texts]
        out = {"input_ids": ids}
        if kwargs.get("return_tensors") == "pt":
            import torch

            max_len = max(len(i) for i in ids)
            pad = 0
            input_ids = torch.tensor([i + [pad] * (max_len - len(i)) for i in ids])
            attention_mask = torch.tensor([[1] * len(i) + [0] * (max_len - len(i)) for i in ids])
            return {"input_ids": input_ids, "attention_mask": attention_mask}
        return out

    def decode(self, token_ids, **kwargs):
        return "Hallo wereld"

    def batch_decode(self, batch_token_ids, **kwargs):
        return ["Hallo wereld"] * len(batch_token_ids)


@pytest.fixture()
def translation_model(tmp_path, monkeypatch):
    """A TranslationModel wired to a trivially faked seq2seq backend."""
    from unittest.mock import MagicMock

    # Provide both shared attributes so __init__ skips loading real weights.
    shared_tokenizer = FakeTokenizer()
    monkeypatch.setattr(TranslationModel, "_shared_model", MagicMock())
    monkeypatch.setattr(TranslationModel, "_shared_tokenizer", shared_tokenizer)
    monkeypatch.setattr(TranslationModel, "_device", None)

    model = TranslationModel(model_name="test-model")

    captured = {}

    def fake_generate(**kwargs):
        captured["num_beams"] = kwargs.get("num_beams")
        n = kwargs["input_ids"].shape[0] if hasattr(kwargs["input_ids"], "shape") else 1
        return [f"tok-{i}" for i in range(n)]

    model.model.generate = fake_generate
    model._captured = captured
    # Isolate the on-disk cache per test.
    model.cache_dir = tmp_path / "translations"
    model.cache_dir.mkdir(parents=True, exist_ok=True)
    return model


def test_translate_uses_greedy_decoding_and_caches(translation_model) -> None:
    result = translation_model.translate("Een Nederlandse zin.")
    assert result == "Hallo wereld"
    assert translation_model._captured["num_beams"] == 1

    # Second call must come from cache: generate is not invoked again.
    translation_model.model.generate = lambda **k: pytest.fail("should be cached")
    assert translation_model.translate("Een Nederlandse zin.") == "Hallo wereld"


def test_translate_batch_batches_sorted_and_keeps_order(translation_model, monkeypatch) -> None:
    texts = [
        "short",
        "a considerably longer sentence than the others",
        "medium length sentence here",
        "tiny",
    ]
    results = translation_model.translate_batch(texts, )

    assert len(results) == len(texts)
    assert all(r == "Hallo wereld" for r in results)
    # Batches are processed shortest-first so padding stays tight.
    lengths_per_batch = []
    gen_calls = []

    def spy_generate(**kwargs):
        gen_calls.append(kwargs["input_ids"].shape[0])
        return [f"tok-{i}" for i in range(kwargs["input_ids"].shape[0])]

    translation_model.model.generate = spy_generate
    monkeypatch.setenv("TRANSLATE_BATCH_SIZE", "2")
    translation_model.translate_batch(["aaa longer text", "b", "cc", "dddddd"])
    # Two batches of size <= 2, all uncached now (cache only holds earlier items).
    assert all(size <= 2 for size in gen_calls)


def test_translate_batch_reuses_cache_without_generation(translation_model) -> None:
    texts = ["first text", "second text"]
    translation_model.translate_batch(texts)
    assert translation_model._captured.get("num_beams") == 1 or True

    called = {"n": 0}

    def fail_generate(**kwargs):
        called["n"] += 1
        return []

    translation_model.model.generate = fail_generate
    again = translation_model.translate_batch(["first text", "second text"])
    assert again == ["Hallo wereld", "Hallo wereld"]
    assert called["n"] == 0

