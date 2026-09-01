"""Load and index the corpus of legal briefs for evaluation dataset generation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ChunkMetadata:
    """Metadata for a chunk from the corpus."""
    chunk_id: str
    heading: str
    source_name: str
    document_name: str  # Name of individual PDF (e.g., "Legal brief 1")
    paragraph_index: int
    original_paragraph: str
    page: Optional[int] = None
    section: Optional[str] = None


class CorpusLoader:
    """Load and index the corpus of legal briefs."""
    
    def __init__(self, project_root: Path):
        self.project_root = Path(project_root)
        self.chunks: List[ChunkMetadata] = []
        self.chunk_texts: Dict[str, str] = {}  # chunk_id -> full text
        self.document_names: List[str] = []
        
    def load_chunks(self) -> None:
        """Load all chunks from the cached chunk JSON files."""
        cache_dir = self.project_root / "data" / "cache"
        if not cache_dir.exists():
            print(f"Warning: cache directory not found at {cache_dir}")
            return
        
        # Find all _chunks.json files
        chunks_files = list(cache_dir.rglob("*_chunks.json"))
        if not chunks_files:
            print(f"Warning: no chunks files found in {cache_dir}")
            return
        
        for chunks_file in sorted(chunks_files):
            try:
                with open(chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                if not isinstance(chunks_data, list):
                    chunks_data = [chunks_data]
                
                for chunk_data in chunks_data:
                    if isinstance(chunk_data, dict):
                        chunk_meta = ChunkMetadata(
                            chunk_id=chunk_data.get("chunk_id", ""),
                            heading=chunk_data.get("heading", ""),
                            source_name=chunk_data.get("source_name", ""),
                            document_name=chunk_data.get("document_name", ""),
                            paragraph_index=chunk_data.get("paragraph_index", 0),
                            original_paragraph=chunk_data.get("original_paragraph", ""),
                            page=chunk_data.get("page"),
                            section=chunk_data.get("section"),
                        )
                        
                        if chunk_meta.chunk_id and chunk_meta.original_paragraph:
                            self.chunks.append(chunk_meta)
                            self.chunk_texts[chunk_meta.chunk_id] = chunk_meta.original_paragraph
                            
                            # Track unique document names
                            if chunk_meta.document_name and chunk_meta.document_name not in self.document_names:
                                self.document_names.append(chunk_meta.document_name)
            except Exception as e:
                print(f"Warning: failed to load chunks from {chunks_file}: {e}")
        
        print(f"Loaded {len(self.chunks)} chunks from {len(chunks_files)} files")
        print(f"Document names: {self.document_names}")
    
    def get_chunks_by_document(self, document_name: str) -> List[ChunkMetadata]:
        """Get all chunks from a specific document."""
        return [c for c in self.chunks if c.document_name == document_name]
    
    def search_chunks(self, keywords: List[str], limit: int = 5) -> List[ChunkMetadata]:
        """Simple keyword search in chunks."""
        keyword_lower = [kw.lower() for kw in keywords]
        scored_chunks = []
        
        for chunk in self.chunks:
            text = chunk.original_paragraph.lower()
            heading = chunk.heading.lower()
            
            # Score based on keyword matches
            score = 0
            for kw in keyword_lower:
                score += text.count(kw)
                score += heading.count(kw) * 2  # Boost heading matches
            
            if score > 0:
                scored_chunks.append((score, chunk))
        
        # Sort by score descending
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        return [chunk for _, chunk in scored_chunks[:limit]]
    
    def get_chunk_text(self, chunk_id: str) -> Optional[str]:
        """Get the full text of a chunk."""
        return self.chunk_texts.get(chunk_id)
    
    def get_all_chunks(self) -> List[ChunkMetadata]:
        """Get all loaded chunks."""
        return self.chunks
    
    def get_documents(self) -> List[str]:
        """Get list of all document names."""
        return sorted(self.document_names)
