# Evaluation Dataset Generation

This folder contains the code to generate an evaluation dataset for the Influencer Regulation RAG chatbot, based on the specification in the task document.

## Workflow

### Step 1: Ensure Corpus is Processed

The evaluation dataset generation requires that the PDF corpus has been processed into chunks. This is done by the main project pipeline:

```bash
cd project
python start_server.py
```

Then access the admin interface to trigger PDF processing. Once complete, chunks will be cached in `data/cache/sources/*/.../_chunks.json`.

### Step 2: Generate Evaluation Dataset

Once the corpus is processed:

```bash
cd project
python evaluation/generate_dataset.py
```

This will:
1. Load all processed chunks from the corpus
2. Analyze the corpus structure and content
3. Generate ~100 evaluation questions following the specification
4. Output three files:
   - `evaluation/output/evaluation_dataset.json` - Complete dataset with gold annotations
   - `evaluation/output/evaluation_dataset.csv` - Flattened CSV for manual inspection
   - `evaluation/output/evaluation_summary.md` - Statistics and coverage analysis

## Dataset Specification

The evaluation dataset includes:

### Question Categories (as per spec)
- **A. General EU Consumer Law** (~8 questions)
- **B. Trader Status** (~10 questions)
- **C. Advertising and Commercial Disclosure** (~15 questions) - CORE
- **D. Content Monetisation** (~10 questions)
- **E. Influencers as Sellers** (~10 questions)
- **F. Consumer Contracts and Consumer Rights** (~8 questions)
- **G. Intellectual Property** (~10 questions)
- **H. Case Law and Advanced Legal Reasoning** (~8 questions)
- **Unanswerable Questions** (~10 questions)
- **Multi-Hop Questions** (~10 questions)
- **Decision-Support Questions** (~10 questions)

### Difficulty Distribution
- **Easy** (30%): Directly answered by one chunk/document
- **Medium** (45%): Requires interpretation or application to simple scenario
- **Hard** (25%): Multiple chunks, legal concepts, case-law, or subtle distinctions

### Question Types
- `fact_retrieval`: Direct factual information retrieval
- `conceptual`: Conceptual understanding
- `scenario_application`: Apply rule to a scenario
- `comparison`: Compare legal concepts
- `case_law`: Case law reasoning
- `decision_support`: Practical decision support
- `multi_hop`: Cross-document reasoning
- `unanswerable`: Intentionally unanswerable

### Gold Annotations

Each question includes:

1. **Gold Answer**: Summary, key points, legal caution, decision
2. **Gold Quotations**: Supporting quotations from the corpus with:
   - Exact text (verbatim from source)
   - Document name and metadata
   - Quality label (direct_support, partial_support, insufficient_support)
   - Mapping to key points supported

3. **Claim-Evidence Mapping**: Maps each claim to supporting quotation IDs

4. **Metadata**: Document, section, page, chunk_id references

## Quality Control

Before finalizing, the generator performs:

1. ✓ Source existence verification
2. ✓ Answer support validation
3. ✓ No invented law check
4. ✓ Duplicate question removal
5. ✓ Category balance
6. ✓ Difficulty balance
7. ✓ Unanswerable verification
8. ✓ Quotation quality assessment
9. ✓ Multi-hop verification
10. ✓ Decision-support grounding

## File Structure

```
project/evaluation/
├── README.md (this file)
├── corpus_loader.py          # Load chunks from cache
├── corpus_analyzer.py        # Analyze corpus and extract content
├── question_generator.py     # Core question generation engine
├── generate_dataset.py       # Main orchestration script
├── output/
│   ├── evaluation_dataset.json    # Complete dataset
│   ├── evaluation_dataset.csv     # Flattened version
│   └── evaluation_summary.md      # Statistics
└── scripts/
    └── (future: utility scripts)
```

## Development Notes

The question generator uses:
- **Corpus analyzer** to identify relevant chunks by topic
- **Question templates** to generate questions systematically
- **Chunk metadata** to provide exact citations and metadata
- **Validation** to ensure all quotations are verbatim from the corpus

The implementation emphasizes:
- Grounding in actual corpus content (no invented law)
- Realistic scenarios (not just dictionary definitions)
- Comprehensive coverage of all legal areas
- Clear distinction between supported and unsupported claims
