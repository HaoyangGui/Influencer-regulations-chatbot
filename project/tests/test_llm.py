from unittest.mock import MagicMock

import litellm
from src.llm import LLMClient


def test_llm_client_uses_litellm_completion(monkeypatch) -> None:
    fake_completion = MagicMock()
    fake_completion.return_value = [{"generated_text": "Answer from LiteLLM."}]
    monkeypatch.setattr(litellm, "completion", fake_completion)

    client = LLMClient(model="test-provider/test-model")
    answer = client.generate_answer("Prompt text")

    assert answer == "Answer from LiteLLM."
    fake_completion.assert_called_once()
