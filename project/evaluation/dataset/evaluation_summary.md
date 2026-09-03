# Evaluation Dataset Summary

## Dataset
- **Total questions:** 100
- **Corpus:** European Commission Influencer Legal Hub, Legal Briefs #1–#10
- **Purpose:** RAG retrieval, answer correctness, quotation/evidence grounding, and decision-support evaluation.

## Category distribution
- General Consumer Law: 8
- Trader Status: 12
- Advertising Disclosure: 16
- Content Monetisation: 10
- Influencers as Sellers: 12
- Consumer Contracts: 10
- Intellectual Property: 12
- Case Law: 6
- Media Law & Self-Regulation: 4
- Multi-hop: 5
- Unanswerable: 5

## Difficulty distribution
- easy: 18
- medium: 36
- hard: 46

## Question-type distribution
- fact_retrieval: 18
- conceptual: 13
- scenario_application: 35
- comparison: 10
- decision_support: 9
- case_law: 5
- multi_hop: 5
- unanswerable: 5

## Special evaluation sets
- Multi-hop questions: 5
- Unanswerable / insufficient-evidence questions: 5
- Decision-support questions: 9
- Questions with multiple gold quotations: 5

## Recommended scoring

### Retrieval
- Recall@1
- Recall@3
- Recall@5
- MRR

### Answer
- Correctness
- Completeness
- Relevance

### Grounding
- Faithfulness
- Unsupported-claim rate
- Contradiction rate

### Quotation / evidence
- Exact quotation match or semantic similarity
- Quotation relevance
- Claim-to-quotation entailment
- Evidence completeness
- Source/document accuracy

### Decision support
- Decision correctness
- Evidence-supported decision
- Appropriate uncertainty
- Actionability

### Robustness
- Unanswerable-question detection
- Hallucination rate
- Overconfidence rate

## Important evaluation rule

A quotation should not receive credit merely because it contains similar keywords. It should directly support the legal claim made in the answer.

For unanswerable questions, the correct behaviour is to acknowledge that the available corpus is insufficient rather than inventing an exact penalty, tax amount, procedural deadline, or guaranteed enforcement outcome.

## Corpus scope

The European Commission describes the Hub as a collection of legal briefs and related materials covering consumer protection, advertising, selling, consumer contracts and intellectual property. The Hub also notes that the selected legal materials are not an exhaustive overview of every rule relevant to influencers.

## Source files
- Legal Brief #1 — European consumer law and influencer marketing
- Legal Brief #2 — What is European consumer law?
- Legal Brief #3 — When is an influencer a 'trader'?
- Legal Brief #4 — Consumer law, media law and self-regulation
- Legal Brief #5 — Content monetisation business models and best practices
- Legal Brief #6 — How to disclose advertising on social media
- Legal Brief #7 — Influencers as sellers: non-conformity
- Legal Brief #8 — Influencers as sellers: information duties and consumer contracts
- Legal Brief #9 — IP and copyright
- Legal Brief #10 — Trade marks and designs
