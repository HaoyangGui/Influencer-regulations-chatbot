"""Test/demo evaluation dataset generator for testing without full corpus."""

from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.question_generator import (
    EvaluationDatasetGenerator,
    Category,
    Difficulty,
    QuestionType,
    GoldAnswer,
    GoldQuotation,
    ClaimEvidenceMapping,
    DecisionSupportExpectations,
)
from evaluation.corpus_loader import ChunkMetadata


def create_mock_corpus_data():
    """Create mock corpus data for testing."""
    mock_chunks = [
        ChunkMetadata(
            chunk_id="chunk_adv_001",
            heading="Advertising Disclosure Requirements",
            source_name="Influencer Legal Hub",
            document_name="Legal brief 1.pdf",
            paragraph_index=0,
            original_paragraph="Commercial content must be clearly marked as advertising. Influencers have an obligation to disclose when they have a commercial relationship with a brand. This disclosure must be unambiguous and immediately obvious to consumers.",
            page=1,
            section="Disclosure"
        ),
        ChunkMetadata(
            chunk_id="chunk_free_001",
            heading="Free Products as Consideration",
            source_name="Influencer Legal Hub",
            document_name="Legal brief 2.pdf",
            paragraph_index=5,
            original_paragraph="Free products, samples, or services received by an influencer can constitute consideration under consumer protection law. If an influencer receives free products and is expected to promote them, this creates a commercial relationship that requires disclosure.",
            page=3,
            section="Monetization"
        ),
        ChunkMetadata(
            chunk_id="chunk_trader_001",
            heading="Trader Status Determination",
            source_name="Influencer Legal Hub",
            document_name="Legal brief 3.pdf",
            paragraph_index=2,
            original_paragraph="An influencer may be considered a trader based on the frequency, nature, and circumstances of their commercial activities, not solely on formal business registration. The CJEU case Kamenova established that regular commercial activity can classify someone as a trader.",
            page=2,
            section="Legal Status"
        ),
        ChunkMetadata(
            chunk_id="chunk_copyright_001",
            heading="Copyright and Influencer Marketing",
            source_name="Influencer Legal Hub",
            document_name="Legal brief 4.pdf",
            paragraph_index=8,
            original_paragraph="Influencers must ensure they have appropriate rights or permissions to use third-party content including photographs, music, and designs. Unauthorized use of copyrighted material in promotional content can expose influencers to liability.",
            page=4,
            section="Intellectual Property"
        ),
        ChunkMetadata(
            chunk_id="chunk_seller_001",
            heading="Influencer as Direct Seller",
            source_name="Influencer Legal Hub",
            document_name="Legal brief 5.pdf",
            paragraph_index=3,
            original_paragraph="When influencers directly sell products to consumers through websites or other channels, they must provide required seller information including a verifiable business name, address, and contact details. An Instagram username alone is insufficient.",
            page=5,
            section="Direct Sales"
        ),
    ]
    return mock_chunks


def generate_demo_dataset(output_dir: Path = None):
    """Generate a demo evaluation dataset without full corpus processing."""
    
    if output_dir is None:
        output_dir = project_root / "evaluation" / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Initializing evaluation dataset generator (DEMO mode)...")
    generator = EvaluationDatasetGenerator(project_root)
    
    # Add mock corpus data
    print("Creating mock corpus data for demonstration...")
    mock_chunks = create_mock_corpus_data()
    for chunk in mock_chunks:
        generator.corpus.chunks.append(chunk)
        generator.corpus.chunk_texts[chunk.chunk_id] = chunk.original_paragraph
    
    print(f"Generated {len(mock_chunks)} mock chunks")
    
    # Generate questions using the system
    print("\nGenerating evaluation questions...")
    
    # Add a few example questions manually
    q1 = generator._add_question(
        "C01",
        Category.ADVERTISING_COMMERCIAL_DISCLOSURE,
        Difficulty.EASY,
        QuestionType.FACT_RETRIEVAL,
        "When does influencer content constitute advertising under EU consumer law?",
        GoldAnswer(
            summary="Commercial content must be clearly marked as advertising when there is a commercial relationship between the influencer and brand.",
            key_points=[
                "Commercial relationships must be disclosed",
                "Disclosure must be clear and unambiguous",
                "Disclosure must be immediately obvious to consumers"
            ],
            legal_caution="Ambiguous or unclear disclosures may not satisfy legal requirements",
            decision="yes"
        ),
        [GoldQuotation(
            quote_id="Q1",
            quote="Commercial content must be clearly marked as advertising. Influencers have an obligation to disclose when they have a commercial relationship with a brand.",
            document="Legal brief 1.pdf",
            chunk_id="chunk_adv_001",
            supports=["Commercial content requires disclosure"],
            quotation_quality="direct_support"
        )],
        [ClaimEvidenceMapping(
            claim="Commercial content requires disclosure",
            supporting_quote_ids=["Q1"]
        )],
        [{"document": "Legal brief 1.pdf", "chunk_id": "chunk_adv_001"}]
    )
    
    q2 = generator._add_question(
        "D01",
        Category.CONTENT_MONETISATION,
        Difficulty.EASY,
        QuestionType.CONCEPTUAL,
        "Do free products from brands constitute consideration under consumer law?",
        GoldAnswer(
            summary="Yes, free products can constitute consideration and may trigger disclosure obligations.",
            key_points=[
                "Free products can be valuable consideration",
                "This may create a commercial relationship",
                "Disclosure obligations may apply"
            ],
            legal_caution="Consider all forms of consideration, not just cash payments",
            decision="yes"
        ),
        [GoldQuotation(
            quote_id="Q2",
            quote="Free products, samples, or services received by an influencer can constitute consideration under consumer protection law.",
            document="Legal brief 2.pdf",
            chunk_id="chunk_free_001",
            supports=["Free products as consideration"],
            quotation_quality="direct_support"
        )],
        [ClaimEvidenceMapping(
            claim="Free products constitute consideration",
            supporting_quote_ids=["Q2"]
        )],
        [{"document": "Legal brief 2.pdf", "chunk_id": "chunk_free_001"}]
    )
    
    q3 = generator._add_question(
        "B01",
        Category.TRADER_STATUS,
        Difficulty.MEDIUM,
        QuestionType.SCENARIO_APPLICATION,
        "I have 30,000 followers but am not registered as a business. I occasionally receive free products. Could I still be considered a trader?",
        GoldAnswer(
            summary="Trader status depends on the nature and frequency of commercial activity, not just business registration.",
            key_points=[
                "Regular commercial activity can establish trader status",
                "Business registration is not the determining factor",
                "Frequency and nature of activity matter"
            ],
            legal_caution="Analyze all circumstances, not just formal registration status",
            decision="depends"
        ),
        [GoldQuotation(
            quote_id="Q3",
            quote="An influencer may be considered a trader based on the frequency, nature, and circumstances of their commercial activities, not solely on formal business registration.",
            document="Legal brief 3.pdf",
            chunk_id="chunk_trader_001",
            supports=["Trader status determination"],
            quotation_quality="direct_support"
        )],
        [ClaimEvidenceMapping(
            claim="Trader status depends on commercial activity",
            supporting_quote_ids=["Q3"]
        )],
        [{"document": "Legal brief 3.pdf", "chunk_id": "chunk_trader_001"}]
    )
    
    q4 = generator._add_question(
        "G01",
        Category.INTELLECTUAL_PROPERTY,
        Difficulty.MEDIUM,
        QuestionType.SCENARIO_APPLICATION,
        "An influencer downloads a professional photographer's image from Instagram and uses it to promote a brand. What legal issue may arise?",
        GoldAnswer(
            summary="Unauthorized use of copyrighted images violates copyright law, even if they are publicly available online.",
            key_points=[
                "Copyright protections apply to all creative works",
                "Public availability does not grant usage rights",
                "Permission is required for commercial use",
                "Liability can be significant"
            ],
            legal_caution="Assume copyright protection for all creative content",
            decision="yes"
        ),
        [GoldQuotation(
            quote_id="Q4",
            quote="Influencers must ensure they have appropriate rights or permissions to use third-party content including photographs, music, and designs. Unauthorized use of copyrighted material can expose influencers to liability.",
            document="Legal brief 4.pdf",
            chunk_id="chunk_copyright_001",
            supports=["Copyright requirements", "Permission necessity"],
            quotation_quality="direct_support"
        )],
        [ClaimEvidenceMapping(
            claim="Copyright protection applies to all creative works",
            supporting_quote_ids=["Q4"]
        )],
        [{"document": "Legal brief 4.pdf", "chunk_id": "chunk_copyright_001"}]
    )
    
    q5 = generator._add_question(
        "E01",
        Category.INFLUENCERS_AS_SELLERS,
        Difficulty.MEDIUM,
        QuestionType.SCENARIO_APPLICATION,
        "An influencer sells clothing directly through their website but provides only an Instagram username as contact information. What consumer-law issue could arise?",
        GoldAnswer(
            summary="Missing proper seller information violates consumer protection requirements.",
            key_points=[
                "Sellers must provide verifiable business information",
                "An Instagram username is insufficient",
                "Physical address and proper contact details are required",
                "This affects consumer remedies and complaints"
            ],
            legal_caution="All seller information requirements must be met",
            decision="yes"
        ),
        [GoldQuotation(
            quote_id="Q5",
            quote="When influencers directly sell products to consumers, they must provide required seller information including a verifiable business name, address, and contact details.",
            document="Legal brief 5.pdf",
            chunk_id="chunk_seller_001",
            supports=["Seller information requirements"],
            quotation_quality="direct_support"
        )],
        [ClaimEvidenceMapping(
            claim="Proper seller information is required",
            supporting_quote_ids=["Q5"]
        )],
        [{"document": "Legal brief 5.pdf", "chunk_id": "chunk_seller_001"}]
    )
    
    # Add unanswerable questions
    generator._generate_unanswerable_questions()
    
    print(f"Generated {len(generator.questions)} questions")
    
    # Generate statistics
    print("\nDataset Statistics:")
    stats = generator.get_statistics()
    print(f"Total questions: {stats['total_questions']}")
    print(f"By category:")
    for cat, count in sorted(stats["by_category"].items()):
        print(f"  - {cat}: {count}")
    
    # Save outputs
    print("\nSaving outputs...")
    
    json_path = output_dir / "evaluation_dataset_demo.json"
    csv_path = output_dir / "evaluation_dataset_demo.csv"
    summary_path = output_dir / "evaluation_summary_demo.md"
    
    generator.save_to_json(json_path)
    generator.save_to_csv(csv_path)
    generator.save_summary(summary_path)
    
    print(f"\nDemo evaluation dataset generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    print(f"- Summary: {summary_path}")
    
    return True


if __name__ == "__main__":
    success = generate_demo_dataset()
    sys.exit(0 if success else 1)
