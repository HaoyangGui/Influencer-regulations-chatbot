from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import chromadb
from chromadb.config import Settings
from chromadb.errors import NotFoundError

from src.chunker import Chunk


class VectorStore:
    def __init__(self, persist_directory: Path, collection_name: str = "rag_chunks") -> None:
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        self.client = chromadb.Client(
            settings=Settings(is_persistent=True, persist_directory=str(persist_directory))
        )
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        try:
            return self.client.get_collection(name=self.collection_name)
        except (ValueError, NotFoundError):
            return self.client.create_collection(name=self.collection_name)

    def store_chunks(self, chunks: List[Chunk], embeddings: List[List[float]], processed_file: str | None = None) -> None:
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.original_paragraph for chunk in chunks]
        metadatas = []
        for chunk in chunks:
            meta = {
                "chunk_id": chunk.chunk_id,
                "heading": chunk.heading,
                "source_name": chunk.source_name,
                "source_url": chunk.source_url,
                "paragraph_index": chunk.paragraph_index,
                "translated_paragraph": chunk.translated_paragraph,
                "embedding_text_en": chunk.embedding_text_en,
            }
            if processed_file:
                meta["processed_file"] = processed_file
            metadatas.append(meta)

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def has_documents(self) -> bool:
        try:
            return self.collection.count() > 0
        except Exception:
            return False

    def clear(self) -> None:
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(name=self.collection_name)

    def get_collection(self):
        return self.collection
