"""Main script to generate evaluation dataset for Influencer Regulation RAG chatbot."""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from evaluation.corpus_loader import CorpusLoader
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


def generate_evaluation_dataset(output_dir: Path = None):
    """Generate the complete evaluation dataset."""
    
    if output_dir is None:
        output_dir = project_root / "evaluation" / "output"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize generator
    print("Initializing evaluation dataset generator...")
    generator = EvaluationDatasetGenerator(project_root)
    
    # Load corpus
    print("Loading corpus...")
    generator.load_corpus()
    
    if len(generator.corpus.get_all_chunks()) == 0:
        print("\nWARNING: No chunks found in corpus!")
        print("The corpus needs to be processed first using the main project pipeline.")
        print("Run: python project/start_server.py")
        print("Then access the admin interface to process the PDFs.")
        return False
    
    print(f"Loaded {len(generator.corpus.get_all_chunks())} chunks")
    print(f"Documents: {generator.corpus.get_documents()}")
    
    # Generate questions
    print("\nGenerating evaluation questions...")
    questions = generator.generate_questions()
    
    if len(questions) == 0:
        print("WARNING: No questions were generated!")
        print("This is because the question generation needs to be completed in question_generator.py")
        return False
    
    # Generate statistics
    print("\nDataset Statistics:")
    stats = generator.get_statistics()
    print(f"Total questions: {stats['total_questions']}")
    print(f"Unanswerable: {stats['unanswerable_count']}")
    print(f"Multi-hop: {stats['multi_hop_count']}")
    print(f"Decision-support: {stats['decision_support_count']}")
    
    # Save outputs
    print("\nSaving outputs...")
    
    json_path = output_dir / "evaluation_dataset.json"
    csv_path = output_dir / "evaluation_dataset.csv"
    summary_path = output_dir / "evaluation_summary.md"
    
    generator.save_to_json(json_path)
    generator.save_to_csv(csv_path)
    generator.save_summary(summary_path)
    
    print(f"\nEvaluation dataset generated successfully!")
    print(f"Output directory: {output_dir}")
    print(f"- JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    print(f"- Summary: {summary_path}")
    
    return True


if __name__ == "__main__":
    success = generate_evaluation_dataset()
    sys.exit(0 if success else 1)
