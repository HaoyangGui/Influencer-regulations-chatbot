"""Question generation implementation based on corpus analysis."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from evaluation.corpus_loader import CorpusLoader, ChunkMetadata


class CorpusAnalyzer:
    """Analyze the corpus to identify relevant content for question generation."""
    
    def __init__(self, corpus: CorpusLoader):
        self.corpus = corpus
        self.topic_keywords = self._build_topic_keywords()
    
    def _build_topic_keywords(self) -> Dict[str, List[str]]:
        """Build keyword sets for each topic."""
        return {
            "advertising_disclosure": [
                "disclosure", "advertising", "advertisement", "clearly marked",
                "commercial", "sponsored", "influencer", "relationship", "advertiser"
            ],
            "trader_status": [
                "trader", "consumer", "business", "commercial activity", "frequency",
                "VAT", "registration", "company", "professional"
            ],
            "free_products": [
                "free", "product", "gift", "consideration", "barter",
                "compensation", "exchange", "receive"
            ],
            "affiliate_marketing": [
                "affiliate", "commission", "link", "payment", "percentage",
                "sales", "earning"
            ],
            "copyright_ip": [
                "copyright", "intellectual property", "photograph", "image",
                "music", "design", "rights", "permission", "authorized"
            ],
            "consumer_protection": [
                "consumer", "protection", "rights", "compliance", "law",
                "EU", "European", "directive", "regulation"
            ],
            "direct_sales": [
                "seller", "sales", "website", "direct", "product information",
                "consumer information", "address", "contact"
            ],
            "content_monetization": [
                "monetization", "sponsorship", "payment", "money", "earnings",
                "income", "revenue"
            ]
        }
    
    def find_chunks_by_topic(self, topic: str, limit: int = 10) -> List[ChunkMetadata]:
        """Find chunks relevant to a specific topic."""
        keywords = self.topic_keywords.get(topic, [])
        return self.corpus.search_chunks(keywords, limit=limit)
    
    def extract_key_sentences(self, text: str, max_sentences: int = 5) -> List[str]:
        """Extract key sentences from a chunk of text."""
        # Simple sentence splitting (not perfect, but works for most cases)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        # Filter out very short sentences and take the first N
        key_sentences = [s.strip() for s in sentences if len(s.strip()) > 20][:max_sentences]
        return key_sentences
    
    def find_quotable_sentences(self, chunk: ChunkMetadata, keywords: List[str]) -> List[Tuple[str, float]]:
        """Find sentences in a chunk that are good candidates for quotations."""
        text = chunk.original_paragraph
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        scored_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 20:  # Too short
                continue
            
            # Score based on keyword matches
            score = sum(1 for kw in keywords if kw.lower() in sentence.lower())
            if score > 0:
                scored_sentences.append((sentence, score))
        
        # Sort by score and return
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        return scored_sentences[:5]


class QuestionTemplateEngine:
    """Generate questions using templates and corpus data."""
    
    def __init__(self, corpus_analyzer: CorpusAnalyzer):
        self.analyzer = corpus_analyzer
    
    def generate_advertising_disclosure_questions(self) -> List[Dict]:
        """Generate questions about advertising disclosure requirements."""
        questions = []
        
        # Find relevant chunks
        chunks = self.analyzer.find_chunks_by_topic("advertising_disclosure")
        
        if chunks:
            chunk = chunks[0]
            
            # Question 1: Simple fact retrieval
            q1 = {
                "id": "C01",
                "category": "C. Advertising and Commercial Disclosure",
                "difficulty": "easy",
                "question_type": "fact_retrieval",
                "question": "When does influencer content constitute advertising according to EU consumer law?",
                "gold_answer_summary": chunk.heading,
                "gold_answer_key_points": [chunk.original_paragraph[:200]],
                "chunk_id": chunk.chunk_id,
                "document": chunk.document_name,
            }
            questions.append(q1)
        
        return questions
    
    def generate_trader_status_questions(self) -> List[Dict]:
        """Generate questions about trader status and consumer qualification."""
        questions = []
        
        chunks = self.analyzer.find_chunks_by_topic("trader_status")
        
        if chunks:
            chunk = chunks[0]
            
            q1 = {
                "id": "B01",
                "category": "B. Trader Status",
                "difficulty": "medium",
                "question_type": "scenario_application",
                "question": "I have 30,000 followers but I am not registered as a business. I occasionally receive free products from brands. Could I still be considered a trader under EU law?",
                "gold_answer_summary": "Trader status depends on the nature and frequency of commercial activity, not just business registration.",
                "gold_answer_key_points": ["Trader status depends on commercial activity", "Registration status is not the only factor"],
                "chunk_id": chunk.chunk_id,
                "document": chunk.document_name,
            }
            questions.append(q1)
        
        return questions
    
    def generate_free_products_questions(self) -> List[Dict]:
        """Generate questions about free products and consideration."""
        questions = []
        
        chunks = self.analyzer.find_chunks_by_topic("free_products")
        
        if chunks:
            chunk = chunks[0]
            
            q1 = {
                "id": "D01",
                "category": "D. Content Monetisation",
                "difficulty": "easy",
                "question_type": "conceptual",
                "question": "Does receiving free products from a brand constitute 'consideration' in the legal sense?",
                "gold_answer_summary": "Free products can constitute consideration under consumer law and may trigger disclosure obligations.",
                "gold_answer_key_points": ["Free products can be consideration", "Disclosure may be required"],
                "chunk_id": chunk.chunk_id,
                "document": chunk.document_name,
            }
            questions.append(q1)
        
        return questions
    
    def generate_copyright_questions(self) -> List[Dict]:
        """Generate questions about intellectual property and copyright."""
        questions = []
        
        chunks = self.analyzer.find_chunks_by_topic("copyright_ip")
        
        if chunks:
            chunk = chunks[0]
            
            q1 = {
                "id": "G01",
                "category": "G. Intellectual Property",
                "difficulty": "medium",
                "question_type": "scenario_application",
                "question": "An influencer downloads a professional photographer's image from Instagram and uses it to promote a brand. What legal issue may arise?",
                "gold_answer_summary": "Using copyrighted images without permission violates copyright law, even if the image is public on social media.",
                "gold_answer_key_points": ["Copyright protection applies", "Permission is required", "Public availability doesn't grant rights"],
                "chunk_id": chunk.chunk_id,
                "document": chunk.document_name,
            }
            questions.append(q1)
        
        return questions
    
    def generate_unanswerable_questions(self) -> List[Dict]:
        """Generate questions that should not be answerable from the corpus."""
        questions = [
            {
                "id": "UN01",
                "category": "Unanswerable / Insufficient-Evidence Questions",
                "difficulty": "hard",
                "question_type": "unanswerable",
                "question": "What is the exact maximum fine an influencer can receive in Germany for failing to disclose an advertisement?",
                "unanswerable": True,
                "reason": "The corpus does not contain jurisdiction-specific fine amounts for Germany."
            },
            {
                "id": "UN02",
                "category": "Unanswerable / Insufficient-Evidence Questions",
                "difficulty": "hard",
                "question_type": "unanswerable",
                "question": "What tax must an influencer pay in the Netherlands when receiving free products?",
                "unanswerable": True,
                "reason": "Tax treatment is jurisdiction-specific and not covered in this consumer law corpus."
            }
        ]
        return questions


def generate_question_templates():
    """Generate a set of question templates based on the specification."""
    
    templates = {
        "advertising_disclosure": [
            "When does influencer content constitute advertising?",
            "What are the disclosure requirements for sponsored content?",
            "How clear must a disclosure be to satisfy EU law?",
            "What is considered 'clear and unambiguous' disclosure?",
            "Must an influencer use specific hashtags or labels?",
        ],
        "trader_status": [
            "What makes an influencer a 'trader' under EU law?",
            "Does an influencer need business registration to be a trader?",
            "How is trader status determined in influencer marketing?",
            "What factors determine if someone qualifies as a trader?",
            "Can an influencer without a company still be a trader?",
        ],
        "free_products": [
            "Do free products constitute consideration?",
            "When must an influencer disclose receiving free products?",
            "How should free product exchanges be labeled?",
            "Is receiving a free sample the same as commercial consideration?",
        ],
        "affiliate_marketing": [
            "What is affiliate marketing in the context of influencer law?",
            "Do affiliate commissions require disclosure?",
            "How should affiliate relationships be disclosed?",
            "What is the legal difference between affiliate and sponsored content?",
        ],
        "consumer_protection": [
            "What are the main principles of EU consumer protection law?",
            "How does consumer law apply to influencer marketing?",
            "What consumer rights are relevant to influencer content?",
        ],
        "copyright": [
            "What copyright issues arise in influencer marketing?",
            "Can an influencer use third-party images without permission?",
            "What are the copyright implications of using music in sponsored content?",
        ]
    }
    
    return templates
