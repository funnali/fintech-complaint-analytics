# RAG Pipeline Evaluation Results — Before/After Improvement Sprint

## Summary

| Metric | Before (original pipeline) | After (improved pipeline) |
|---|---|---|
| Average quality score | 2.1 / 5 | **3.1 / 5** |
| Retrieval category mismatches | 2 of 8 questions | **0 of 8 questions** |
| Prompt-template leakage in answers | 4 of 8 questions | **0 of 8 questions** |

Two targeted fixes closed both concrete gaps the original evaluation
identified:
1. **Category-aware retrieval** — the pipeline now detects the product
   category implied by a question and filters ChromaDB results toward
   it, eliminating cross-category source mismatches (previously affected
   Q2 and Q4).
2. **Answer post-processing** — strips prompt-template leakage artifacts
   ("Excerpt N:", "Answer N:", repeated questions) from every generated
   answer before it reaches the user (previously affected Q6 and Q8).

## Detailed Results (after improvements)

| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |
|---|---|---|---|---|
| Why are customers unhappy with credit cards? | they've done a lot and trying to do more and more for their consumers to win business and win reputation, it's so funny that i see a credit card company couldn't wait to close their c | [Credit Card, complaint 3775797]; [Credit Card, complaint 4378171] | 2 | Unchanged — still cuts off mid-sentence, no true synthesis. Remaining weakness is the generator model's capacity, not retrieval or post-processing. |
| What issues do people report with money transfers? | hacked and money was transferred to a unidentified individual. account | [Money Transfer, complaint 11549638]; [Money Transfer, complaint 5738094] | 3 | **Improved from 2** — both sources are now correctly Money Transfer (previously one was mismatched Savings Account). Answer is a rough fragment but on-topic. |
| What problems do customers have with savings accounts? | They are getting drastically below market rates of interest unless they call in to switch. this is just not right and consistent with a company that should be acting in the best interest of their customers. | [Savings Account, complaint 6188958]; [Savings Account, complaint 2170710] | 4 | Unchanged — consistently the strongest result: coherent, on-topic, correctly matched sources. |
| Why do personal loan customers file complaints? | They do not understand that people face hardships and are not customer friendly at all. they do not try to work out arrangements they scream at you and tell you you should have made the payment on time then. | [Personal Loan, complaint 1627780]; [Personal Loan, complaint 1384640] | 4 | **Improved from 2** — both sources now correctly Personal Loan (previously one was mismatched Credit Card). Answer is now two clean, coherent sentences with no leakage. |
| Are there complaints about unauthorized transactions? | i believe these practices are unfair and respectfully request an investigation into my account and the specific transactions in question. | [Savings Account, complaint 7710396]; [Savings Account, complaint 2890090] | 3 | Unchanged — accurate and correctly sourced, though verbatim rather than synthesized. |
| Do customers report problems with customer service response times? | No. Customer service behaviors are very bizarre and disturbing. The resolution i asked is the restoration of my account and funds without delay. | [Savings Account, complaint 5579808]; [Savings Account, complaint 4376303] | 4 | **Improved from 1** — the "Answer 3:" template-leakage artifact is fully stripped now; answer is coherent and on-topic. |
| What billing issues are commonly reported for credit cards? | against them and xxxx of those complaints are billing collection issues | [Credit Card, complaint 1544179]; [Credit Card, complaint 1742437] | 2 | Unchanged — still a short verbatim fragment, though no leakage artifacts. |
| Are there complaints about fraud or scams? | fraud and their decision is unacceptable and want answers. thank you | [Money Transfer, complaint 4228595]; [Money Transfer, complaint 6482545] | 3 | **Improved from 1** — no longer repeats the question back verbatim; ending is a little abrupt but the leakage bug is fixed. |

**New average score: 3.1 / 5** (up from 2.1 / 5 — a ~48% relative improvement)

## Remaining Known Limitation

The generator model (`flan-t5-small`, 77M parameters) still produces
verbatim or fragment-style answers rather than true synthesis in about
half of test cases (Q1, Q7 in particular). This is a model-capacity
limitation, not a retrieval or engineering bug — the two targeted fixes
in this sprint addressed the specific, diagnosable issues (category
mismatch, template leakage) that were fixable with better engineering.
Further quality gains would require a larger or API-based generator
model, noted as a future improvement.
