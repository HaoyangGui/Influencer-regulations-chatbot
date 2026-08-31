from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, ClassVar

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
            # Greedy decoding: for RAG quotation purposes beam search adds
            # 3-4x latency on CPU with negligible quality difference.
            outputs = self.model.generate(**inputs, max_length=max_length, num_beams=1)
        decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self._write_cached_translation(text, decoded)
        return decoded

    def translate_batch(self, texts: Iterable[str], max_length: int = 512) -> List[str]:
        items = [t for t in texts if t]
        if not items:
            return []

        results: Dict[str, str] = {}
        uncached_items: List[str] = []
        for item in items:
            cached = self._read_cached_translation(item)
            if cached is not None:
                results[item] = cached
            else:
                uncached_items.append(item)

        if uncached_items:
            batch_size = max(1, int(os.getenv("TRANSLATE_BATCH_SIZE", "12")))
            # Length-sorting keeps every padded mini-batch as tight as possible:
            # a single huge padded batch pads short texts up to the longest one,
            # wasting most of the decoder compute on padding.
            ordered_uncached = sorted(dict.fromkeys(uncached_items), key=len)
            for start in range(0, len(ordered_uncached), batch_size):
                batch = ordered_uncached[start : start + batch_size]
                inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, max_length=max_length, num_beams=1)
                decoded = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                for item, translation in zip(batch, decoded):
                    results[item] = translation
                    self._write_cached_translation(item, translation)

        # Reconstruct original order from memory (no per-item disk re-reads).
        return [results.get(item, "") for item in items]


def normalize_language_code(lang: str) -> str:
    """Normalize a detected language label/full name to a short ISO-ish code.

    e.g. "dutch" -> "nl", "german" -> "de", "fr" -> "fr". Falls back to the
    first two characters of the label, or "en" when nothing can be inferred.
    """
    label = (lang or "").strip().lower()
    full = {
        "dutch": "nl", "netherlands": "nl", "flemish": "nl",
        "english": "en", "french": "fr", "german": "de", "germanic": "de",
        "spanish": "es", "italian": "it", "portuguese": "pt",
        "chinese": "zh", "japanese": "ja", "korean": "ko", "javanese": "jv",
        "afrikaans": "af", "arabic": "ar", "czech": "cs", "danish": "da",
        "greek": "el", "finnish": "fi", "hindi": "hi", "croatian": "hr",
        "hungarian": "hu", "indonesian": "id", "icelandic": "is",
        "malay": "ms", "norwegian": "no", "polish": "pl", "romanian": "ro",
        "russian": "ru", "swedish": "sv", "thai": "th", "turkish": "tr",
        "vietnamese": "vi", "persian": "fa", "hebrew": "he",
        "mandarin": "zh", "cantonese": "zh", "ukrainian": "uk",
    }
    if label in full:
        return full[label]
    if len(label) >= 2 and label[:2].isalpha():
        return label[:2]
    return "en"


def detect_language(text: str) -> str:
    normalized = text.strip()
    if not normalized:
        return "en"

    try:
        from langdetect import detect

        return normalize_language_code(detect(normalized))
    except Exception:
        pass
    return "en"


_LANGUAGE_NAMES: dict[str, str] = {
    "en": "English", "nl": "Dutch", "fr": "French", "de": "German",
    "es": "Spanish", "it": "Italian", "pt": "Portuguese", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "ru": "Russian", "ar": "Arabic",
    "hi": "Hindi", "tr": "Turkish", "pl": "Polish", "sv": "Swedish",
    "da": "Danish", "no": "Norwegian", "fi": "Finnish", "cs": "Czech",
    "hu": "Hungarian", "ro": "Romanian", "el": "Greek", "id": "Indonesian",
    "ms": "Malay", "th": "Thai", "vi": "Vietnamese", "uk": "Ukrainian",
    "fa": "Persian", "he": "Hebrew", "af": "Afrikaans", "hr": "Croatian",
    "is": "Icelandic", "jv": "Javanese",
}


def language_name(code: str) -> str:
    """Human-readable name for a language code/label.

    Deliberately not restricted to a fixed set: the codebase supports an
    arbitrary number of languages, so unknown codes fall back to the raw
    label (or the normalised code) instead of being dropped or assumed.
    """
    label = (code or "").strip()
    normalized = normalize_language_code(label)
    known = _LANGUAGE_NAMES.get(normalized)
    if known:
        return known
    # A readable full label (e.g. "Dutch", "Mandarin") beats an internal code.
    if len(label) >= 2:
        return label
    return normalized or "en"
