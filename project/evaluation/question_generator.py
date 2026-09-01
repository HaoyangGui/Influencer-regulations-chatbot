"""Generate evaluation questions for the Influencer Regulation RAG chatbot."""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional

from evaluation.corpus_loader import ChunkMetadata, CorpusLoader
from evaluation.corpus_analyzer import CorpusAnalyzer, QuestionTemplateEngine, generate_question_templates


class Difficulty(Enum):
    """Question difficulty levels."""
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class QuestionType(Enum):
    """Types of evaluation questions."""
    FACT_RETRIEVAL = "fact_retrieval"
    CONCEPTUAL = "conceptual"
    SCENARIO_APPLICATION = "scenario_application"
    COMPARISON = "comparison"
    CASE_LAW = "case_law"
    DECISION_SUPPORT = "decision_support"
    MULTI_HOP = "multi_hop"
    UNANSWERABLE = "unanswerable"


class Category(Enum):
    """Question categories based on the specification."""
    GENERAL_EU_CONSUMER_LAW = "A. General EU Consumer Law"
    TRADER_STATUS = "B. Trader Status"
    ADVERTISING_COMMERCIAL_DISCLOSURE = "C. Advertising and Commercial Disclosure"
    CONTENT_MONETISATION = "D. Content Monetisation"
    INFLUENCERS_AS_SELLERS = "E. Influencers as Sellers"
    CONSUMER_CONTRACTS_AND_RIGHTS = "F. Consumer Contracts and Consumer Rights"
    INTELLECTUAL_PROPERTY = "G. Intellectual Property"
    CASE_LAW_AND_ADVANCED_REASONING = "H. Case Law and Advanced Legal Reasoning"
    UNANSWERABLE = "Unanswerable / Insufficient-Evidence Questions"
    MULTI_HOP = "Multi-Hop / Cross-Document Questions"
    DECISION_SUPPORT = "Decision-Support Questions"


@dataclass
class GoldQuotation:
    """A quotation from the corpus that supports an answer."""
    quote_id: str
    quote: str
    document: str  # PDF name, e.g., "Legal brief 1.pdf"
    section: Optional[str] = None
    page: Optional[int] = None
    chunk_id: Optional[str] = None
    supports: List[str] = field(default_factory=list)  # Key points supported
    quotation_quality: str = "direct_support"  # direct_support, partial_support, insufficient_support


@dataclass
class ClaimEvidenceMapping:
    """Maps a claim to supporting quotations."""
    claim: str
    supporting_quote_ids: List[str]


@dataclass
class GoldAnswer:
    """The gold standard answer to a question."""
    summary: str
    key_points: List[str]
    legal_caution: Optional[str] = None
    decision: Optional[str] = None  # yes, no, potentially, depends, insufficient_information


@dataclass
class DecisionSupportExpectations:
    """Expectations for decision-support questions."""
    should_identify: List[str] = field(default_factory=list)
    should_warn_about: List[str] = field(default_factory=list)
    should_not_claim: List[str] = field(default_factory=list)


@dataclass
class EvaluationQuestion:
    """A complete evaluation question with gold annotations."""
    id: str
    category: str
    difficulty: str  # easy, medium, hard
    question_type: str  # from QuestionType enum
    question: str
    gold_answer: GoldAnswer
    gold_quotations: List[GoldQuotation]
    claim_evidence_mapping: List[ClaimEvidenceMapping]
    gold_sources: List[Dict[str, str]]
    requires_multi_hop: bool = False
    unanswerable: bool = False
    decision_support_expectations: DecisionSupportExpectations = field(default_factory=DecisionSupportExpectations)


class EvaluationDatasetGenerator:
    """Generate evaluation questions from the corpus."""
    
    def __init__(self, project_root: Path, random_seed: int = 42):
        self.project_root = Path(project_root)
        self.corpus = CorpusLoader(self.project_root)
        self.random_seed = random_seed
        random.seed(random_seed)
        self.questions: List[EvaluationQuestion] = []
        self.quote_counter = 0
    
    def load_corpus(self) -> None:
        """Load the corpus."""
        self.corpus.load_chunks()
    
    def generate_next_quote_id(self) -> str:
        """Generate a unique quote ID."""
        self.quote_counter += 1
        return f"Q{self.quote_counter}"
    
    def generate_questions(self) -> List[EvaluationQuestion]:
        """Generate all evaluation questions."""
        print("Starting question generation...")
        
        # Initialize analyzers
        analyzer = CorpusAnalyzer(self.corpus)
        template_engine = QuestionTemplateEngine(analyzer)
        
        # Generate questions by category
        self._generate_advertising_disclosure_questions(analyzer)
        self._generate_trader_status_questions(analyzer)
        self._generate_free_products_questions(analyzer)
        self._generate_affiliate_marketing_questions(analyzer)
        self._generate_copyright_ip_questions(analyzer)
        self._generate_consumer_protection_questions(analyzer)
        self._generate_direct_sales_questions(analyzer)
        self._generate_content_monetization_questions(analyzer)
        self._generate_unanswerable_questions()
        self._generate_multi_hop_questions(analyzer)
        self._generate_decision_support_questions(analyzer)
        
        print(f"Generated {len(self.questions)} questions")
        return self.questions
    
    def _generate_advertising_disclosure_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about advertising disclosure (Category C)."""
        # Find relevant chunks about advertising and disclosure
        chunks = analyzer.find_chunks_by_topic("advertising_disclosure", limit=5)
        
        for idx, chunk in enumerate(chunks[:3]):
            q_id = f"C{(idx + 1):02d}"
            
            # Extract a key sentence for quotation
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.ADVERTISING_COMMERCIAL_DISCLOSURE.value,
                difficulty=Difficulty.EASY.value if idx == 0 else Difficulty.MEDIUM.value,
                question_type=QuestionType.FACT_RETRIEVAL.value if idx == 0 else QuestionType.SCENARIO_APPLICATION.value,
                question=f"What are the disclosure requirements for {chunk.heading.lower()}?" if idx == 0 
                         else f"How should an influencer handle disclosure requirements related to {chunk.heading.lower()}?",
                gold_answer=GoldAnswer(
                    summary=f"According to {chunk.heading}: {chunk.original_paragraph[:200]}",
                    key_points=[f"Key point from {chunk.heading}", "Disclosure is required"],
                    legal_caution="Ensure compliance with EU consumer protection regulations",
                    decision="yes" if idx == 0 else "depends"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=[f"Key point from {chunk.heading}"],
                        quotation_quality="direct_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim=f"Key point from {chunk.heading}",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_trader_status_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about trader status (Category B)."""
        chunks = analyzer.find_chunks_by_topic("trader_status", limit=5)
        
        for idx, chunk in enumerate(chunks[:3]):
            q_id = f"B{(idx + 1):02d}"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.TRADER_STATUS.value,
                difficulty=Difficulty.MEDIUM.value,
                question_type=QuestionType.SCENARIO_APPLICATION.value,
                question=f"When might an influencer be considered a trader based on {chunk.heading.lower()}?",
                gold_answer=GoldAnswer(
                    summary=f"Trader status is determined by: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Trader status depends on commercial activity nature and frequency",
                        "Registration status alone does not determine trader classification"
                    ],
                    legal_caution="Multiple factors should be considered, not just one criterion",
                    decision="depends"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Trader status determination factors"],
                        quotation_quality="direct_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Trader status depends on commercial activity",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_free_products_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about free products and consideration (Category D)."""
        chunks = analyzer.find_chunks_by_topic("free_products", limit=5)
        
        for idx, chunk in enumerate(chunks[:2]):
            q_id = f"D{(idx + 1):02d}"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.CONTENT_MONETISATION.value,
                difficulty=Difficulty.EASY.value,
                question_type=QuestionType.CONCEPTUAL.value,
                question="Do free products constitute consideration under consumer law?",
                gold_answer=GoldAnswer(
                    summary=f"Free products can constitute consideration: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Free products can be considered valuable consideration",
                        "This may trigger disclosure obligations"
                    ],
                    legal_caution="The nature and context of the free products matter",
                    decision="yes"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Free products as consideration"],
                        quotation_quality="direct_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Free products can be consideration",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_affiliate_marketing_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about affiliate marketing (Category D)."""
        chunks = analyzer.find_chunks_by_topic("affiliate_marketing", limit=5)
        
        if chunks:
            chunk = chunks[0]
            q_id = "D03"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.CONTENT_MONETISATION.value,
                difficulty=Difficulty.MEDIUM.value,
                question_type=QuestionType.COMPARISON.value,
                question="What is the legal difference between receiving a free product, receiving a direct payment, and receiving an affiliate commission?",
                gold_answer=GoldAnswer(
                    summary=f"Affiliate marketing differs from direct payment and gifts: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Each monetization model has different legal implications",
                        "All may require disclosure",
                        "The nature of consideration varies"
                    ],
                    legal_caution="All forms of consideration may trigger consumer law obligations",
                    decision="depends"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Affiliate marketing vs other monetization"],
                        quotation_quality="partial_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Different monetization models have different implications",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_copyright_ip_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about intellectual property (Category G)."""
        chunks = analyzer.find_chunks_by_topic("copyright_ip", limit=5)
        
        for idx, chunk in enumerate(chunks[:2]):
            q_id = f"G{(idx + 1):02d}"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.INTELLECTUAL_PROPERTY.value,
                difficulty=Difficulty.MEDIUM.value,
                question_type=QuestionType.SCENARIO_APPLICATION.value,
                question="An influencer downloads a professional photographer's image from Instagram and uses it to promote a brand. What legal issue may arise?",
                gold_answer=GoldAnswer(
                    summary=f"Copyright issues arise with unauthorized use: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Copyright protection applies to all creative works",
                        "Public availability does not grant usage rights",
                        "Permission is required for commercial use"
                    ],
                    legal_caution="Assume copyright protection even for public content",
                    decision="yes"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Copyright in influencer marketing"],
                        quotation_quality="direct_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Copyright protection applies to creative works",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_consumer_protection_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about consumer protection (Category A)."""
        chunks = analyzer.find_chunks_by_topic("consumer_protection", limit=5)
        
        for idx, chunk in enumerate(chunks[:2]):
            q_id = f"A{(idx + 1):02d}"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.GENERAL_EU_CONSUMER_LAW.value,
                difficulty=Difficulty.EASY.value,
                question_type=QuestionType.FACT_RETRIEVAL.value,
                question="What is the purpose of EU consumer law and how does it apply to influencers?",
                gold_answer=GoldAnswer(
                    summary=f"EU consumer law protects consumers: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Consumer protection is a fundamental principle",
                        "Influencers are covered when acting commercially"
                    ],
                    legal_caution="Consumer protection applies broadly to commercial activity",
                    decision="yes"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Consumer protection principles"],
                        quotation_quality="direct_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Consumer protection applies to influencers",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_direct_sales_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate questions about influencers as sellers (Category E)."""
        chunks = analyzer.find_chunks_by_topic("direct_sales", limit=5)
        
        for idx, chunk in enumerate(chunks[:2]):
            q_id = f"E{(idx + 1):02d}"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.INFLUENCERS_AS_SELLERS.value,
                difficulty=Difficulty.MEDIUM.value,
                question_type=QuestionType.SCENARIO_APPLICATION.value,
                question="An influencer sells clothing directly through their website but provides only an Instagram username as contact information. What consumer-law issue could arise?",
                gold_answer=GoldAnswer(
                    summary=f"Missing contact information violates seller obligations: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Sellers must provide proper contact information",
                        "Instagram username alone is insufficient",
                        "Consumer remedies may be affected"
                    ],
                    legal_caution="Proper seller information is required by law",
                    decision="yes"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Seller information requirements"],
                        quotation_quality="direct_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Sellers must provide proper contact information",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_content_monetization_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate additional content monetization questions (Category D)."""
        chunks = analyzer.find_chunks_by_topic("content_monetization", limit=5)
        
        for idx, chunk in enumerate(chunks[:2]):
            q_id = f"D{(idx + 4):02d}"
            
            sentences = analyzer.extract_key_sentences(chunk.original_paragraph, max_sentences=1)
            quote = sentences[0] if sentences else chunk.original_paragraph[:100]
            
            q = EvaluationQuestion(
                id=q_id,
                category=Category.CONTENT_MONETISATION.value,
                difficulty=Difficulty.MEDIUM.value,
                question_type=QuestionType.COMPARISON.value,
                question=f"How does {chunk.heading.lower()} differ in terms of legal obligations?",
                gold_answer=GoldAnswer(
                    summary=f"Different monetization models have varying obligations: {chunk.original_paragraph[:150]}",
                    key_points=[
                        "Each model may have different disclosure requirements",
                        "Legal obligations depend on the nature of consideration",
                        "Context matters in determining obligations"
                    ],
                    legal_caution="Carefully analyze the specific arrangement",
                    decision="depends"
                ),
                gold_quotations=[
                    GoldQuotation(
                        quote_id=self.generate_next_quote_id(),
                        quote=quote,
                        document=chunk.document_name,
                        chunk_id=chunk.chunk_id,
                        supports=["Monetization models"],
                        quotation_quality="partial_support"
                    )
                ],
                claim_evidence_mapping=[
                    ClaimEvidenceMapping(
                        claim="Monetization models have different obligations",
                        supporting_quote_ids=["Q1"]
                    )
                ],
                gold_sources=[{
                    "document": chunk.document_name,
                    "chunk_id": chunk.chunk_id,
                }],
                requires_multi_hop=False,
                unanswerable=False,
            )
            self.questions.append(q)
    
    def _generate_unanswerable_questions(self) -> None:
        """Generate questions that should not be answerable from the corpus."""
        unanswerable_qs = [
            {
                "id": "UN01",
                "question": "What is the exact maximum fine an influencer can receive in Germany for failing to disclose an advertisement?",
                "decision": "insufficient_information",
            },
            {
                "id": "UN02",
                "question": "What tax must an influencer pay in the Netherlands when receiving free products?",
                "decision": "insufficient_information",
            },
            {
                "id": "UN03",
                "question": "What is the exact deadline for appealing an influencer-marketing fine in France?",
                "decision": "insufficient_information",
            },
            {
                "id": "UN04",
                "question": "Will this particular influencer definitely be fined by the Dutch authorities?",
                "decision": "insufficient_information",
            },
        ]
        
        for q_data in unanswerable_qs:
            q = EvaluationQuestion(
                id=q_data["id"],
                category="Unanswerable / Insufficient-Evidence Questions",
                difficulty=Difficulty.HARD.value,
                question_type=QuestionType.UNANSWERABLE.value,
                question=q_data["question"],
                gold_answer=GoldAnswer(
                    summary="This question cannot be answered with the available sources.",
                    key_points=[
                        "The required information is not contained in the corpus",
                        "Jurisdiction-specific details are not available"
                    ],
                    legal_caution="Do not hallucinate information beyond the corpus",
                    decision=q_data["decision"]
                ),
                gold_quotations=[],
                claim_evidence_mapping=[],
                gold_sources=[],
                requires_multi_hop=False,
                unanswerable=True,
            )
            self.questions.append(q)
    
    def _generate_multi_hop_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate multi-hop questions requiring cross-document reasoning."""
        q1 = EvaluationQuestion(
            id="MH01",
            category="Multi-Hop / Cross-Document Questions",
            difficulty=Difficulty.HARD.value,
            question_type=QuestionType.MULTI_HOP.value,
            question="An influencer receives free clothing from a brand, posts about the clothing on Instagram, and also sells their own clothing through a website. What different consumer-law obligations could potentially arise?",
            gold_answer=GoldAnswer(
                summary="Multiple legal areas could apply: advertising disclosure, trader status, and direct sales obligations.",
                key_points=[
                    "Advertising disclosure requirements apply to the free product post",
                    "Trader status must be evaluated for the sales activity",
                    "Seller information obligations apply to direct sales",
                    "Different legal frameworks may overlap"
                ],
                legal_caution="Carefully evaluate each aspect separately and in combination",
                decision="potentially"
            ),
            gold_quotations=[],  # Would be filled with actual quotes
            claim_evidence_mapping=[],
            gold_sources=[],
            requires_multi_hop=True,
            unanswerable=False,
        )
        self.questions.append(q1)
    
    def _generate_decision_support_questions(self, analyzer: CorpusAnalyzer) -> None:
        """Generate decision-support questions."""
        decisions_questions = [
            {
                "id": "DS01",
                "question": "I am planning to promote a product that I received for free. What should I do before publishing the post?",
                "should_identify": [
                    "Need for disclosure",
                    "Consumer protection obligations",
                    "What constitutes 'clear and unambiguous' disclosure"
                ],
                "should_warn_about": [
                    "Regulatory penalties for non-disclosure",
                    "Consumer complaints"
                ],
                "should_not_claim": [
                    "Specific fines or penalties",
                    "Jurisdiction-specific requirements"
                ]
            },
            {
                "id": "DS02",
                "question": "I want to use another photographer's image in a sponsored Instagram post. What should I check first?",
                "should_identify": [
                    "Copyright and intellectual property rights",
                    "Need for permission",
                    "Types of licenses"
                ],
                "should_warn_about": [
                    "Copyright infringement risks",
                    "Potential legal liability"
                ],
                "should_not_claim": [
                    "Specific damages",
                    "Exact penalties"
                ]
            },
        ]
        
        for q_data in decisions_questions:
            q = EvaluationQuestion(
                id=q_data["id"],
                category="Decision-Support Questions",
                difficulty=Difficulty.MEDIUM.value,
                question_type=QuestionType.DECISION_SUPPORT.value,
                question=q_data["question"],
                gold_answer=GoldAnswer(
                    summary="Consider the legal obligations before taking action.",
                    key_points=q_data["should_identify"],
                    legal_caution="Consult legal expertise for specific situations",
                    decision="depends"
                ),
                gold_quotations=[],
                claim_evidence_mapping=[],
                gold_sources=[],
                requires_multi_hop=False,
                unanswerable=False,
                decision_support_expectations=DecisionSupportExpectations(
                    should_identify=q_data["should_identify"],
                    should_warn_about=q_data["should_warn_about"],
                    should_not_claim=q_data["should_not_claim"]
                )
            )
            self.questions.append(q)
    
    def _add_question(self, 
                     question_id: str,
                     category: Category,
                     difficulty: Difficulty,
                     question_type: QuestionType,
                     question_text: str,
                     gold_answer: GoldAnswer,
                     gold_quotations: List[GoldQuotation],
                     claim_evidence_mapping: List[ClaimEvidenceMapping],
                     gold_sources: List[Dict[str, str]],
                     requires_multi_hop: bool = False,
                     unanswerable: bool = False,
                     decision_support_expectations: Optional[DecisionSupportExpectations] = None) -> None:
        """Add a question to the dataset."""
        
        question = EvaluationQuestion(
            id=question_id,
            category=category.value,
            difficulty=difficulty.value,
            question_type=question_type.value,
            question=question_text,
            gold_answer=gold_answer,
            gold_quotations=gold_quotations,
            claim_evidence_mapping=claim_evidence_mapping,
            gold_sources=gold_sources,
            requires_multi_hop=requires_multi_hop,
            unanswerable=unanswerable,
            decision_support_expectations=decision_support_expectations or DecisionSupportExpectations(),
        )
        self.questions.append(question)
    
    def get_statistics(self) -> Dict[str, any]:
        """Get statistics about the generated dataset."""
        stats = {
            "total_questions": len(self.questions),
            "by_category": {},
            "by_difficulty": {},
            "by_question_type": {},
            "unanswerable_count": sum(1 for q in self.questions if q.unanswerable),
            "multi_hop_count": sum(1 for q in self.questions if q.requires_multi_hop),
            "decision_support_count": sum(1 for q in self.questions if q.decision_support_expectations.should_identify),
            "with_multiple_quotes": sum(1 for q in self.questions if len(q.gold_quotations) > 1),
        }
        
        # Count by category
        for q in self.questions:
            cat = q.category
            stats["by_category"][cat] = stats["by_category"].get(cat, 0) + 1
        
        # Count by difficulty
        for q in self.questions:
            diff = q.difficulty
            stats["by_difficulty"][diff] = stats["by_difficulty"].get(diff, 0) + 1
        
        # Count by question type
        for q in self.questions:
            qt = q.question_type
            stats["by_question_type"][qt] = stats["by_question_type"].get(qt, 0) + 1
        
        return stats
    
    def save_to_json(self, output_path: Path) -> None:
        """Save the dataset as JSON."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = []
        for q in self.questions:
            q_dict = {
                "id": q.id,
                "category": q.category,
                "difficulty": q.difficulty,
                "question_type": q.question_type,
                "question": q.question,
                "gold_answer": {
                    "summary": q.gold_answer.summary,
                    "key_points": q.gold_answer.key_points,
                    "legal_caution": q.gold_answer.legal_caution,
                    "decision": q.gold_answer.decision,
                },
                "gold_quotations": [
                    {
                        "quote_id": quat.quote_id,
                        "quote": quat.quote,
                        "document": quat.document,
                        "section": quat.section,
                        "page": quat.page,
                        "chunk_id": quat.chunk_id,
                        "supports": quat.supports,
                        "quotation_quality": quat.quotation_quality,
                    }
                    for quat in q.gold_quotations
                ],
                "claim_evidence_mapping": [
                    {
                        "claim": cem.claim,
                        "supporting_quote_ids": cem.supporting_quote_ids,
                    }
                    for cem in q.claim_evidence_mapping
                ],
                "gold_sources": q.gold_sources,
                "requires_multi_hop": q.requires_multi_hop,
                "unanswerable": q.unanswerable,
                "decision_support_expectations": {
                    "should_identify": q.decision_support_expectations.should_identify,
                    "should_warn_about": q.decision_support_expectations.should_warn_about,
                    "should_not_claim": q.decision_support_expectations.should_not_claim,
                },
            }
            data.append(q_dict)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved JSON to {output_path}")
    
    def save_to_csv(self, output_path: Path) -> None:
        """Save a flattened version as CSV."""
        import csv
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                "id",
                "category",
                "difficulty",
                "question_type",
                "question",
                "gold_answer_summary",
                "key_points_count",
                "key_points",
                "legal_caution",
                "decision",
                "quotations_count",
                "quotation_snippets",
                "requires_multi_hop",
                "unanswerable",
            ])
            
            # Data
            for q in self.questions:
                writer.writerow([
                    q.id,
                    q.category,
                    q.difficulty,
                    q.question_type,
                    q.question,
                    q.gold_answer.summary,
                    len(q.gold_answer.key_points),
                    " | ".join(q.gold_answer.key_points),
                    q.gold_answer.legal_caution or "",
                    q.gold_answer.decision or "",
                    len(q.gold_quotations),
                    " | ".join(f'"{quat.quote[:50]}..."' for quat in q.gold_quotations),
                    q.requires_multi_hop,
                    q.unanswerable,
                ])
        
        print(f"Saved CSV to {output_path}")
    
    def save_summary(self, output_path: Path) -> None:
        """Save a summary markdown file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        stats = self.get_statistics()
        
        summary = f"""# Evaluation Dataset Summary

Generated: {datetime.now().isoformat()}

## Overview

- **Total Questions**: {stats['total_questions']}
- **Unanswerable Questions**: {stats['unanswerable_count']}
- **Multi-hop Questions**: {stats['multi_hop_count']}
- **Decision-support Questions**: {stats['decision_support_count']}
- **Questions with Multiple Quotations**: {stats['with_multiple_quotes']}

## Distribution by Category

"""
        for cat, count in sorted(stats["by_category"].items()):
            pct = (count / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
            summary += f"- {cat}: {count} ({pct:.1f}%)\n"
        
        summary += "\n## Distribution by Difficulty\n\n"
        for diff in ["easy", "medium", "hard"]:
            count = stats["by_difficulty"].get(diff, 0)
            pct = (count / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
            summary += f"- {diff.capitalize()}: {count} ({pct:.1f}%)\n"
        
        summary += "\n## Distribution by Question Type\n\n"
        for qt, count in sorted(stats["by_question_type"].items()):
            pct = (count / stats['total_questions'] * 100) if stats['total_questions'] > 0 else 0
            summary += f"- {qt}: {count} ({pct:.1f}%)\n"
        
        summary += "\n## Source Document Coverage\n\n"
        summary += f"- Documents in corpus: {len(self.corpus.get_documents())}\n"
        summary += f"- Document names: {', '.join(self.corpus.get_documents())}\n"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        
        print(f"Saved summary to {output_path}")
