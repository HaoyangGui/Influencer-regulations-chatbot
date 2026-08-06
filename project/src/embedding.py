from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Iterable, List

import torch
from sentence_transformers import SentenceTransformer


@dataclass
class EmbeddingModel:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    model: SentenceTransformer | None = None
    batch_size: int = 32
    device: str | None = None
    cache_dir: str | None = None

    _shared_model: ClassVar[SentenceTransformer | None] = None
    _shared_device: ClassVar[str | None] = None

    def __post_init__(self) -> None:
        if EmbeddingModel._shared_model is None:
            resolved_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
            if auth_token:
                os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", auth_token)
            model = self._load_sentence_transformer(self.model_name, resolved_device)
            model.eval()
            EmbeddingModel._shared_model = model
            EmbeddingModel._shared_device = resolved_device
        self.model = EmbeddingModel._shared_model
        self.device = EmbeddingModel._shared_device
        self.cache_dir = Path(self.cache_dir or Path(__file__).resolve().parent.parent / "data" / "cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _load_sentence_transformer(self, model_name: str, device: str) -> SentenceTransformer:
        try:
            return SentenceTransformer(model_name, device=device, local_files_only=True)
        except Exception:
            return SentenceTransformer(model_name, device=device)

    def ensure_local_cache(self) -> None:
        auth_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
        if auth_token:
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", auth_token)
        try:
            SentenceTransformer(self.model_name, device=self.device or ("cuda" if torch.cuda.is_available() else "cpu"), local_files_only=True)
            print("Embedding model is already cached locally.")
        except Exception:
            print("Downloading embedding model to local cache...")
            try:
                SentenceTransformer(self.model_name, device=self.device or ("cuda" if torch.cuda.is_available() else "cpu"), local_files_only=False)
            except TypeError:
                SentenceTransformer(self.model_name, device=self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _cache_path_for_text(self, text: str) -> Path:
        cache_key = {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "device": self.device,
            "text": text,
        }
        text_hash = hashlib.sha256(json.dumps(cache_key, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / f"embedding_{text_hash}.json"

    def _read_cached_embedding(self, text: str) -> List[float] | None:
        cache_path = self._cache_path_for_text(text)
        if not cache_path.exists():
            return None
        try:
            with cache_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, list):
                return payload
        except Exception:
            return None
        return None

    def _write_cached_embedding(self, text: str, embedding: List[float]) -> None:
        cache_path = self._cache_path_for_text(text)
        try:
            with cache_path.open("w", encoding="utf-8") as handle:
                json.dump(embedding, handle)
        except Exception:
            return

    def embed_texts(self, texts: Iterable[str]) -> List[List[float]]:
        if self.model is None:
            raise RuntimeError("Embedding model is not loaded.")

        text_list = [text for text in texts if text]
        if not text_list:
            return []

        uncached_texts: List[str] = []
        results_by_text: dict[str, List[float]] = {}
        for text in text_list:
            cached = self._read_cached_embedding(text)
            if cached is None:
                uncached_texts.append(text)
            else:
                results_by_text[text] = cached

        if uncached_texts:
            with torch.inference_mode():
                embeddings = self.model.encode(
                    uncached_texts,
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    device=self.device,
                )

            for text, embedding in zip(uncached_texts, embeddings):
                if hasattr(embedding, "tolist"):
                    normalized_embedding = embedding.tolist()
                else:
                    normalized_embedding = list(embedding)
                self._write_cached_embedding(text, normalized_embedding)
                results_by_text[text] = normalized_embedding

        return [results_by_text[text] for text in text_list]

    def embed_text(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]
