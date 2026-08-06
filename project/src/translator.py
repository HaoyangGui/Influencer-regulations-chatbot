from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, List, ClassVar

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class TranslationModel:
    _shared_model: ClassVar[AutoModelForSeq2SeqLM | None] = None
    _shared_tokenizer: ClassVar[AutoTokenizer | None] = None
    _device: ClassVar[torch.device | None] = None

    def __init__(self, model_name: str = "Helsinki-NLP/opus-mt-mul-en") -> None:
        self.model_name = model_name
        self._loaded_from_local = False
        if TranslationModel._shared_model is None or TranslationModel._shared_tokenizer is None:
            self._load_model()
        self.model = TranslationModel._shared_model
        self.tokenizer = TranslationModel._shared_tokenizer
        self.device = TranslationModel._device
        self.cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache" / "translations"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_model(self) -> None:
        auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if auth_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", auth_token)
        tokenizer = self._from_pretrained_local_first(AutoTokenizer, self.model_name)
        model = self._from_pretrained_local_first(AutoModelForSeq2SeqLM, self.model_name)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        if device.type == "cuda":
            # Use half precision on GPU whenever possible to reduce latency and memory use.
            model = model.half()
            model.to(device)
        TranslationModel._shared_model = model
        TranslationModel._shared_tokenizer = tokenizer
        TranslationModel._device = device

    def ensure_local_cache(self) -> None:
        auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if auth_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", auth_token)
        try:
            AutoTokenizer.from_pretrained(self.model_name, local_files_only=True)
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=True)
            print("Translation model is already cached locally.")
        except Exception:
            print("Downloading translation model to local cache...")
            AutoTokenizer.from_pretrained(self.model_name, local_files_only=False)
            AutoModelForSeq2SeqLM.from_pretrained(self.model_name, local_files_only=False)

    def _from_pretrained_local_first(self, cls, model_name: str, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs.pop("use_auth_token")
        try:
            result = cls.from_pretrained(model_name, local_files_only=True, **kwargs)
            self._loaded_from_local = True
            return result
        except Exception:
            self._loaded_from_local = False
            return cls.from_pretrained(model_name, local_files_only=False, **kwargs)

    def _cache_path_for_text(self, text: str) -> Path:
        cache_key = {
            "model_name": self.model_name,
            "text": text,
        }
        text_hash = hashlib.sha256(json.dumps(cache_key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"{text_hash}.json"

    def _read_cached_translation(self, text: str) -> str | None:
        cache_path = self._cache_path_for_text(text)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, str):
                return payload
        except Exception:
            return None
        return None

    def _write_cached_translation(self, text: str, translation: str) -> None:
        cache_path = self._cache_path_for_text(text)
        try:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(translation, handle)
        except Exception:
            return

    def translate(self, text: str, max_length: int = 512) -> str:
        if not text:
            return ""
        cached = self._read_cached_translation(text)
        if cached is not None:
            return cached
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=4)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self._write_cached_translation(text, decoded)
        return decoded

    def translate_batch(self, texts: Iterable[str], max_length: int = 512) -> List[str]:
        items = [t for t in texts if t]
        if not items:
            return []

        cached_results: List[str] = []
        uncached_items: List[str] = []
        for item in items:
            cached = self._read_cached_translation(item)
            if cached is not None:
                cached_results.append(cached)
            else:
                uncached_items.append(item)

        if uncached_items:
            inputs = self.tokenizer(uncached_items, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_length=max_length, num_beams=4)
            decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for item, translation in zip(uncached_items, decoded):
                self._write_cached_translation(item, translation)
                cached_results.append(translation)

        # Reconstruct original order
        ordered_results: List[str] = []
        for item in items:
            translated = self._read_cached_translation(item)
            if translated is not None:
                ordered_results.append(translated)
        return ordered_results

    def detect_language(self, text: str) -> str:
        normalized = text.strip()
        if not normalized:
            return "en"

        try:
            from langdetect import detect

            return detect(normalized)
        except Exception:
            pass

        if any("\u4e00" <= ch <= "\u9fff" for ch in normalized):
            return "zh"
        if any(ch in normalized for ch in ["á", "é", "í", "ó", "ú", "ü", "ö", "ä", "ß"]):
            return "nl"
        return "en"
