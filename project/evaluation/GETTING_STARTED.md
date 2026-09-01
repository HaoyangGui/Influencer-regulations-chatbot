# Evaluation Dataset Generation - Getting Started

This guide explains how to use the evaluation dataset generation system for the Influencer Regulation RAG chatbot.

## Quick Start

### Option 1: Demo Mode (No Prerequisites)

To see the evaluation system in action with sample data:

```bash
cd project
python evaluation/generate_dataset_demo.py
```

This generates:
- `evaluation/output/evaluation_dataset_demo.json` - Full dataset with annotations
- `evaluation/output/evaluation_dataset_demo.csv` - Spreadsheet format
- `evaluation/output/evaluation_summary_demo.md` - Statistics and coverage

The demo includes real example questions on advertising disclosure, trader status, free products, copyright, and seller obligations, with full gold annotations.

### Option 2: Full Production Mode (Requires Corpus Processing)

To generate the complete 100-question evaluation dataset:

#### Step 1: Process the PDF Corpus

The PDFs in `project/data/pdf/Influencer legal hub/` need to be OCR'd and chunked first:

```bash
cd project
python start_server.py
```

Visit the admin interface and trigger PDF processing. This will:
1. OCR all 10 Legal Brief PDFs using Mistral
2. Clean and chunk the OCR output
3. Cache chunks in `data/cache/`

#### Step 2: Generate Evaluation Dataset

Once PDFs are processed:

```bash
cd project
python evaluation/generate_dataset.py
```

This generates three files:
- `evaluation/output/evaluation_dataset.json` - 100 questions with full annotations
- `evaluation/output/evaluation_dataset.csv` - Flattened version for inspection
- `evaluation/output/evaluation_summary.md` - Statistics

## File Structure

```
project/evaluation/
├── README.md                    # Detailed specification and QC info
├── GETTING_STARTED.md          # This file
├── __init__.py                 # Package marker
├── corpus_loader.py            # Load chunks from cache
├── corpus_analyzer.py          # Analyze corpus, extract content
├── question_generator.py       # Core generation engine
├── generate_dataset.py         # Production script (requires corpus)
├── generate_dataset_demo.py    # Demo script (no prerequisites)
├── output/
│   ├── evaluation_dataset_demo.json
│   ├── evaluation_dataset_demo.csv
│   ├── evaluation_summary_demo.md
│   ├── evaluation_dataset.json         (generated when corpus is ready)
│   ├── evaluation_dataset.csv          (generated when corpus is ready)
│   └── evaluation_summary.md           (generated when corpus is ready)
└── scripts/
    └── (future utility scripts)
```

## What Gets Generated

### Dataset Schema

Each evaluation question includes:

```json
{
  "id": "C01",
  "category": "C. Advertising and Commercial Disclosure",
  "difficulty": "easy",
  "question_type": "fact_retrieval",
  "question": "...",
  
  "gold_answer": {
    "summary": "...",
    "key_points": ["...", "..."],
    "legal_caution": "...",
    "decision": "yes|no|potentially|depends|insufficient_information"
  },
  
  "gold_quotations": [
    {
      "quote_id": "Q1",
      "quote": "Verbatim text from source",
      "document": "Legal brief 1.pdf",
      "chunk_id": "chunk_001",
      "supports": ["key point 1"],
      "quotation_quality": "direct_support"
    }
  ],
  
  "claim_evidence_mapping": [
    {
      "claim": "...",
      "supporting_quote_ids": ["Q1"]
    }
  ],
  
  "requires_multi_hop": false,
  "unanswerable": false
}
```

### Statistics Provided

The summary includes:
- Total questions and distribution by category
- Difficulty distribution (easy/medium/hard)
- Question type breakdown
- Unanswerable and multi-hop counts
- Source document coverage

## Dataset Specification

The generator creates ~100 questions across these categories:

| Category | Count | Focus |
|----------|-------|-------|
| A. General EU Consumer Law | 8 | Purpose, principles, scope |
| B. Trader Status | 10 | Determination, criteria |
| C. Advertising Disclosure | 15 | **CORE - requirements, clarity, placement** |
| D. Content Monetization | 10 | Comparison, models, obligations |
| E. Influencers as Sellers | 10 | Direct sales, information, conformity |
| F. Consumer Contracts | 8 | Pre-contractual info, rights, complaints |
| G. Intellectual Property | 10 | Copyright, trademarks, permissions |
| H. Case Law | 8 | Application, reasoning, precedent |
| Unanswerable | 10 | Knowledge boundaries testing |
| Multi-hop | 10 | Cross-document reasoning |
| Decision Support | 10 | Practical guidance |

### Difficulty Distribution

- **Easy** (30%): One chunk, direct answer
- **Medium** (45%): Interpretation, scenario application
- **Hard** (25%): Multi-chunk, case law, subtle distinctions

### Quality Assurance

Before finalizing, the system performs:
1. ✓ Source existence verification
2. ✓ Answer support validation
3. ✓ No invented law check
4. ✓ Duplicate removal
5. ✓ Category balance
6. ✓ Difficulty balance
7. ✓ Unanswerable verification
8. ✓ Quotation quality assessment
9. ✓ Multi-hop verification
10. ✓ Decision-support grounding

## Understanding the Output Files

### JSON Format (`evaluation_dataset.json`)

- Complete dataset with all annotations
- Full quotations (verbatim from corpus)
- Multiple quotations per question if needed
- Claim-to-evidence mappings
- Metadata (document, page, section, chunk_id)
- Decision-support expectations for applicable questions

**Use for**: RAG evaluation frameworks, detailed analysis, importing into evaluation tools

### CSV Format (`evaluation_dataset.csv`)

- Flattened, one-row-per-question format
- Quotations truncated to 50 chars for readability
- Key points pipe-delimited
- Easier manual inspection and annotation

**Use for**: Manual review, spreadsheet tools, quick scanning

### Summary (`evaluation_summary.md`)

- Statistics and distribution charts
- Category breakdown with percentages
- Question type distribution
- Source document coverage
- Timestamp

**Use for**: Reporting, planning, overview

## Integration with RAG Evaluation

To use this dataset for evaluating your RAG chatbot:

1. **Retrieval Evaluation**: Compare model-retrieved chunks to `gold_sources`
2. **Answer Evaluation**: Compare generated answer to `gold_answer`
3. **Quotation Evaluation**: Compare model-provided quotation to `gold_quotations`
4. **Grounding Evaluation**: Check if answer stays within quotation bounds
5. **Completeness Evaluation**: Use `claim_evidence_mapping` to verify all claims supported

Metrics you can calculate:
- Recall@K for retrieval
- BLEU/ROUGE for answer quality
- Exact match for quotations
- Faithfulness score
- Hallucination rate
- Decision correctness rate

## Extending the Dataset

To add more questions or customize for your needs:

1. Edit `question_generator.py` to add new methods
2. Use `CorpusAnalyzer` to find relevant chunks
3. Create questions following the template pattern
4. Call `generator._add_question()` for each question
5. Ensure all quotations are verbatim from corpus

Example:
```python
generator._add_question(
    "CUSTOM01",
    Category.ADVERTISING_COMMERCIAL_DISCLOSURE,
    Difficulty.MEDIUM,
    QuestionType.SCENARIO_APPLICATION,
    "Your question here?",
    GoldAnswer(
        summary="Your answer here",
        key_points=["Point 1", "Point 2"],
        legal_caution="Any cautions?",
        decision="yes|no|depends|..."
    ),
    [GoldQuotation(...)],
    [ClaimEvidenceMapping(...)],
    [{"document": "...", "chunk_id": "..."}]
)
```

## Troubleshooting

### "No chunks found in corpus"
**Solution**: Run `python project/start_server.py` and process PDFs via admin interface first.

### Want to test without Mistral API?
**Solution**: Use the demo mode: `python evaluation/generate_dataset_demo.py`

### Questions not generating?
**Check**: 
1. Corpus has chunks loaded
2. ChunkMetadata has valid `chunk_id` and `original_paragraph`
3. CorpusAnalyzer topic keywords match your chunk content

### CSV file has truncated quotations?
**This is intentional** for readability. See full quotations in JSON file.

## Next Steps

1. **Try the demo**: `python evaluation/generate_dataset_demo.py`
2. **Read the spec**: See `project/evaluation/README.md` for full details
3. **Process corpus**: Set up main pipeline to OCR the PDFs
4. **Generate full dataset**: `python evaluation/generate_dataset.py`
5. **Integrate with evaluation**: Use JSON output in your RAG evaluation framework

## Questions?

Refer to:
- `project/evaluation/README.md` - Full specification
- `project/evaluation/question_generator.py` - Generation logic
- `project/evaluation/corpus_analyzer.py` - Corpus analysis
- Demo JSON output - Example structure and content
