from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import litellm


class LLMClient:
    _metadata_log_path = Path(__file__).resolve().parent.parent / "logs" / "llm_responses.jsonl"

    def __init__(self, model: str = "openai/gpt-3.5-turbo") -> None:
        self.model = os.getenv("LLM_MODEL", model)
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.base_url = os.getenv("LLM_API_BASE_URL")
        self._ensure_log_dir()

    def _ensure_log_dir(self) -> None:
        self._metadata_log_path.parent.mkdir(parents=True, exist_ok=True)

    def _extract_content(self, result: Any) -> str:
        if isinstance(result, str):
            return result.strip()

        if hasattr(result, "choices"):
            choices = getattr(result, "choices", None)
            if choices:
                first_choice = choices[0]
                message = getattr(first_choice, "message", None)
                if message is not None:
                    content = getattr(message, "content", None)
                    if content:
                        return content.strip()
                content = getattr(first_choice, "generated_text", None)
                if content:
                    return content.strip()
                if isinstance(first_choice, dict):
                    return first_choice.get("message", {}).get("content", "") or first_choice.get("generated_text", "") or ""

        if isinstance(result, list) and result:
            first = result[0]
            if isinstance(first, dict):
                return (
                    first.get("message", {}).get("content")
                    or first.get("generated_text")
                    or first.get("text")
                    or ""
                )

        if isinstance(result, dict):
            choices = result.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    return (
                        first.get("message", {}).get("content")
                        or first.get("generated_text")
                        or first.get("text")
                        or ""
                    )
            return result.get("generated_text", "").strip()

        return str(result).strip()

    def _normalize_metadata(self, result: Any) -> Any:
        if hasattr(result, "to_dict"):
            try:
                return result.to_dict()
            except Exception:
                pass
        if isinstance(result, dict) or isinstance(result, list):
            return result
        return {"repr": repr(result)}

    def _log_response_metadata(self, metadata: Any) -> None:
        entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": self.model,
            "response": metadata,
        }
        with self._metadata_log_path.open("a", encoding="utf-8") as handle:
            json.dump(entry, handle, ensure_ascii=False)
            handle.write("\n")

    def generate_answer(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        messages = [
            {"role": "system", "content": "You answer questions based only on the provided source documents."},
            {"role": "user", "content": prompt},
        ]

        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_completion_tokens": max_tokens,
            "stream": False,
        }

        if self.api_key:
            request_kwargs["api_key"] = self.api_key
        if self.base_url:
            request_kwargs["base_url"] = self.base_url

        result = litellm.completion(**request_kwargs)
        metadata = self._normalize_metadata(result)
        self._log_response_metadata(metadata)
        return self._extract_content(result)
