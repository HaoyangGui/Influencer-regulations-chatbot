# LLM-as-a-Judge Summary

Prompt version: `V1`

## Overall

| Metric | Value |
|---|---:|
| number_of_questions | 100 |
| mean_correctness | 1.13 |
| mean_completeness | 1.01 |
| mean_evidence_support | 1.19 |
| mean_instruction_following | 1.87 |
| mean_total_score | 0.65 |
| number_of_answerable_questions | 95 |
| number_of_partially_answerable_questions | 0 |
| number_of_unanswerable_questions | 5 |

## Strengths

- instruction_following was strongest (mean 1.87/2).

## Weaknesses

- completeness was weakest (mean 1.01/2).

## Common Errors

- Incomplete answer: 34
- Incorrect legal conclusion: 17
- Incorrect citation: 17

## Recommendations

- Keep requiring claims to cite retrieved chunks and quotations to match verbatim.
- Strengthen prompt guidance for completeness.
