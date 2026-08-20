from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.embedding import EmbeddingModel
from src.llm import LLMClient
from src.translator import TranslationModel, detect_language, language_name, normalize_language_code
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
    quotation: str = ""
    quotation_language: str = ""


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
        self.last_language = "en"

    @staticmethod
    def _detect_language_code(lang: str) -> str:
        """Normalise a detected language label to a short code (e.g. nl, en)."""
        return normalize_language_code(lang)

    @staticmethod
    def _select_quotation(metadata: Dict[str, Any], source_text: str, lang: str) -> tuple[str, str]:
        """Return the chunk text best aligned with the requested language.

        The preference order is data-driven and deliberately NOT tied to any
        specific language pair (the pipeline must support whatever languages
        are used in the future):
          1. an exact translation for the detected user language (from the
             ``translations`` map), if present,
          2. the authentic source-language original of this chunk,
          3. legacy backward-compatibility: older persisted chunks stored only
             an English text (``translated_paragraph`` / ``embedding_text_en``)
             with no ``translations`` map. Those are used ONLY when the user
             actually requested English, so other languages are never silently
             forced into English,
          4. otherwise the raw document text in its own (source) language.
        Reports the language code of whatever text it actually returned.
        """
        lang_code = normalize_language_code(lang)
        source_lang = normalize_language_code(metadata.get("source_language") or "")

        translations = metadata.get("translations")
        if isinstance(translations, str):
            try:
                import json as _json

                parsed = _json.loads(translations)
                translations = parsed if isinstance(parsed, dict) else None
            except Exception:
                translations = None
        if isinstance(translations, dict):
            # 1. Exact translation in the user's language, whatever it is.
            if translations.get(lang_code):
                return translations[lang_code], lang_code
            # 2. Authentic source-language original (kept as the primary basis
            #    so the returned RAG evidence stays faithful to the document).
            if source_lang and translations.get(source_lang):
                return translations[source_lang], source_lang

        # 3. Backward-compatible with legacy persisted chunks that have no
        #    ``translations`` map but do store English text. Only English
        #    requests may use it (so a French/German/Chinese user still gets
        #    the authentic source text instead of an English bridge).
        if lang_code == "en":
            legacy_english = metadata.get("translated_paragraph") or metadata.get("embedding_text_en") or ""
            if legacy_english and legacy_english.strip():
                return legacy_english, "en"

        # 4. Last resort: the raw document text in its source language.
        return source_text, source_lang

    @staticmethod
    def _dot_product(a: List[float], b: List[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def _compute_similarity(self, query_embedding: List[float], document_embeddings: List[List[float]]) -> List[float]:
        return [self._dot_product(query_embedding, doc_emb) for doc_emb in document_embeddings]

    def retrieve(self, question: str, top_k: int = 5) -> List[RetrievalResult]:
        # Compute the query embedding in the user's own language (no forced
        # English translation) so retrieval/similarity follows the question's
        # language. The multilingual embedding model shares one vector space
        # with the stored chunk vectors.
        self.last_language = self._detect_language_code(detect_language(question))
        query_embedding = self.embedding_model.embed_text(question)
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
            quotation, quotation_language = self._select_quotation(metadata, document_text, self.last_language)
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
                    quotation=quotation,
                    quotation_language=quotation_language,
                )
            )
        return results

    def _truncate_text(self, text: str, max_length: int) -> str:
        if len(text) <= max_length:
            return text
        return text[: max_length - 3].rstrip() + "..."

    def build_prompt(self, question: str, chunks: List[RetrievalResult], language: Optional[str] = None) -> str:
        max_chunk_length = 3000
        language = self._detect_language_code(language or self.last_language or detect_language(question))
        # Keep the language the model must answer/quote in aligned with the
        # question. The name is resolved at query time so any supported language
        # works (the fallback set is not locked to a fixed list).
        target_lang_name = language_name(language) or language
        language_instruction = (
            f"Answer in the same language ({target_lang_name}, ISO code: {language}) as the question.\n"
            f"The answer, every quotation, and every citation must be written in {target_lang_name} ({language})."
        )
        prompt_lines = [
            "You are a helpful assistant that answers questions about Dutch influencer marketing regulations using only the provided source material.",
            "Do not invent any information. If the answer cannot be found in the provided text, say you do not know.",
            language_instruction,
            "CRITICAL CITATION RULES:",
            "- For every factual claim, include a short exact quote in quotation marks with a citation number in square brackets.",
            "- Number citations starting from 1 based on the order chunks appear below: Chunk 1 = [1], Chunk 2 = [2], etc.",
            "- Only cite chunks that are actually listed below.",
            f"- Quotes must be kept in the same language ({target_lang_name}); if the provided text is not yet in that language, render it in the question's language.",
            "Do not add any extra explanation beyond the answer and the evidence quotes. Only include evidence quotes that are actually used to support your answer.",
            "Do not invent quotes. Use only the source text provided below.",
            "",
            "Sources:",
        ]

        for index, chunk in enumerate(chunks, start=1):
            quote_lang = chunk.quotation_language
            prompt_lines.extend([
                f"Chunk {index} (chunk_id: {chunk.chunk_id}):",
                f"Heading: {chunk.heading}",
                f"Source URL: {chunk.source_url}",
                f"Text (in {language_name(quote_lang) or quote_lang}): {self._truncate_text(chunk.quotation, max_chunk_length)}",
                f"Original text: {self._truncate_text(chunk.original_paragraph, max_chunk_length)}",
                "",
            ])

        prompt_lines.extend([
            "Question:",
            question,
            "",
            "Answer:",
        ])
        return "\n".join(prompt_lines)

    def answer(
        self,
        question: str,
        top_k: int = 5,
        max_tokens: int = 256,
        results: Optional[List[RetrievalResult]] = None,
        language: Optional[str] = None,
    ) -> tuple[str, List[RetrievalResult], str]:
        # Reuse pre-retrieved chunks when the caller already has them. This keeps
        # the single retrieval consistent: the same RAG chunks feed both the LLM
        # prompt and the returned evidence (so the answer basis is reproducible).
        if results is None:
            results = self.retrieve(question, top_k=top_k)
        lang = self._detect_language_code(language or self.last_language or detect_language(question))
        prompt = self.build_prompt(question, results, language=lang)
        answer_text = self.llm.generate_answer(prompt, max_tokens=max_tokens)
        return answer_text, results, lang
