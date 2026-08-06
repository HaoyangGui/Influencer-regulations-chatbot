from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.embedding import EmbeddingModel
from src.llm import LLMClient
from src.translator import TranslationModel
from src.vectorstore import VectorStore


@dataclass
class RetrievalResult:
    similarity: float
    chunk_id: str
    heading: str
    paragraph_index: int
    original_paragraph: str
    translated_paragraph: str
    source_name: str
    source_url: str
    processed_file: str | None = None


class Retriever:
    def __init__(
        self,
        vector_dir: Path,
        collection_name: str = "rag_chunks",
        translator: Optional[TranslationModel] = None,
        embedding_model: Optional[EmbeddingModel] = None,
        llm: Optional[LLMClient] = None,
    ) -> None:
        self.translator = translator or TranslationModel()
        self.embedding_model = embedding_model or EmbeddingModel()
        self.llm = llm or LLMClient()
        self.vector_store = VectorStore(persist_directory=vector_dir, collection_name=collection_name)
        self.collection = self.vector_store.get_collection()

    @staticmethod
    def _dot_product(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _compute_similarity(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        return [self._dot_product(query_embedding, doc_emb) for doc_emb in document_embeddings]

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        query_text_en = self.translator.translate(question)
        query_embedding = self.embedding_model.embed_text(query_text_en)
        query_results: Dict[str, Any] = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "embeddings"],
        )

        metadatas = query_results.get("metadatas", [[]])[0]
        documents = query_results.get("documents", [[]])[0]
        document_embeddings = query_results.get("embeddings", [[]])[0]

        if not metadatas or not documents:
            return []

        similarities = self._compute_similarity(query_embedding, document_embeddings)

        results: List[RetrievalResult] = []
        for similarity, metadata, document_text in zip(similarities, metadatas, documents):
            results.append(
                RetrievalResult(
                    similarity=similarity,
                    chunk_id=metadata.get("chunk_id", ""),
                    heading=metadata.get("heading", ""),
                    paragraph_index=int(metadata.get("paragraph_index", 0)),
                    original_paragraph=document_text,
                    translated_paragraph=metadata.get("translated_paragraph", ""),
                    source_name=metadata.get("source_name", ""),
                    source_url=metadata.get("source_url", ""),
                    processed_file=metadata.get("processed_file"),
                )
            )
        return results

    def _truncate_text(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def build_prompt(self, question: str, chunks: List[RetrievalResult]) -> str:
        max_chunk_length = 3000
        question_language = self.translator.detect_language(question)
        language_instruction = f"Answer in the same language as the question. Detected language code: {question_language}."
        prompt_lines = [
            "You are a helpful assistant that answers questions about Dutch influencer marketing regulations using only the provided source material.",
            "Do not invent any information. If the answer cannot be found in the provided text, say you do not know.",
            language_instruction,
            "CRITICAL CITATION RULES:",
            "- For every factual claim, include a short exact quote in quotation marks with a citation number in square brackets.",
            "- Number citations starting from 1 based on the order chunks appear below: Chunk 1 = [1], Chunk 2 = [2], etc.",
            "- Only cite chunks that are actually listed below.",
            "Do not add any extra explanation beyond the answer and the evidence quotes. Only include evidence quotes that are actually used to support your answer.",
            "Do not invent quotes. Use only the source text provided below.",
            "",
            "Sources:",
        ]

        for index, chunk in enumerate(chunks, start=1):
            prompt_lines.extend([
                f"Chunk {index}:",
                f"Heading: {chunk.heading}",
                f"Source URL: {chunk.source_url}",
                f"Original text: {self._truncate_text(chunk.original_paragraph, max_chunk_length)}",
                f"English translation: {self._truncate_text(chunk.translated_paragraph, max_chunk_length)}",
                "",
            ])

        prompt_lines.extend([
            "Question:",
            question,
            "",
            "Answer:",
        ])
        return "\n".join(prompt_lines)

    def answer(self, question: str, top_k: int = 5, max_tokens: int = 256) -> tuple[str, List[RetrievalResult]]:
        results = self.retrieve(question, top_k=top_k)
        prompt = self.build_prompt(question, results)
        answer_text = self.llm.generate_answer(prompt, max_tokens=max_tokens)
        return answer_text, results
