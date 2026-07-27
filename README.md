# CrediTrust Complaint Insights

![Unit Tests](https://github.com/funnali/fintech-complaint-analytics/actions/workflows/unittests.yml/badge.svg)

A Retrieval-Augmented Generation (RAG) chatbot that lets internal teams at
a fintech company ask plain-English questions about customer complaints
and get evidence-backed answers grounded in real complaint data — no data
analyst required.

## Business Problem

CrediTrust Financial receives thousands of customer complaints every
month across credit cards, personal loans, savings accounts, and money
transfers. Product, Support, and Compliance teams currently have no way
to understand complaint trends without manually reading complaints one by
one — a process that takes days and keeps teams reactive instead of
proactive. This tool lets any team member type a question like *"why are
customers unhappy with credit cards?"* and get a synthesized answer
backed by real complaint excerpts in seconds.

## Solution Overview

A four-stage pipeline: (1) clean and categorize 480K+ real CFPB complaint
records, (2) chunk and embed a representative sample into a ChromaDB
vector store, (3) retrieve the most relevant chunks for a given question
and generate a grounded answer with a local LLM, (4) serve it all through
an interactive Gradio chat interface that shows its sources for
verification.

## Key Results

- **Answer quality improved 48%** (2.1/5 → 3.1/5 average) after two
  targeted engineering fixes — see `evaluation_results.md` for the full
  before/after breakdown.
- **Retrieval category mismatches eliminated**: 2 of 8 test questions
  originally retrieved sources from the wrong product category; 0 after
  adding category-aware retrieval filtering.
- **480,580 complaints** cleaned and categorized from a 9.6M-row raw
  dataset; **35,171 chunks** indexed for semantic search.
- **16 passing unit tests**, enforced automatically on every push via CI.

## Quick Start

```bash
git clone https://github.com/funnali/fintech-complaint-analytics
cd fintech-complaint-analytics
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# Run the test suite
pytest tests/ -v

# Launch the chatbot (requires vector_store/ to be built first — see below)
python app.py
```

To rebuild the pipeline from scratch: place the raw CFPB CSV at
`data/raw/complaints.csv`, then run `python src/task1_eda_preprocessing.py`
→ `python src/task2_chunk_embed_index.py` → `python app.py`.

## Project Structure

```
rag-complaint-chatbot/
├── .github/workflows/unittests.yml   # CI: runs pytest on every push
├── data/
│   ├── raw/                          # place complaints.csv here
│   └── filtered_complaints.csv       # cleaned output of Task 1
├── vector_store/                     # persisted ChromaDB index
├── src/
│   ├── utils.py                      # typed config + reusable core logic
│   ├── task1_eda_preprocessing.py    # EDA, cleaning, category mapping
│   ├── task2_chunk_embed_index.py    # chunking, embedding, indexing
│   ├── task3_rag_pipeline.py         # retriever + prompt + generator
│   └── task3_evaluation.py           # runs eval questions, scores table
├── tests/
│   └── test_utils.py                 # 16 unit tests, no model downloads needed
├── app.py                            # Gradio chat interface
├── evaluation_results.md             # before/after quality evaluation
├── REPORT.md / FINAL_REPORT.md       # written project reports
└── requirements.txt
```

## Demo

*(Screenshot of the Gradio interface — question box, generated answer,
and source excerpts shown for verification.)*

## Technical Details

- **Data**: CFPB consumer complaint dataset, 9.6M raw rows filtered to
  480,580 complaints across 4 product categories (Credit Card, Personal
  Loan, Savings Account, Money Transfer), via an explicit mapping from 11
  overlapping raw CFPB product labels.
- **Chunking**: LangChain `RecursiveCharacterTextSplitter`, 500 characters
  / 50 overlap, matched to the spec of the reference full-scale vector
  store for comparability.
- **Embedding**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim,
  CPU-friendly).
- **Vector store**: ChromaDB, persisted locally, with category-aware
  retrieval filtering.
- **Generator**: `google/flan-t5-small`, with a grounded prompt template
  and an extractive fallback for degenerate outputs.
- **Evaluation**: 8 representative questions, manually scored 1-5 for
  relevance/coherence; see `evaluation_results.md` for full results and
  before/after analysis.

## Future Improvements

- Swap the generator for a larger or API-based model (`flan-t5-base`,
  GPT-4o-mini, or Claude) — the current small local model is the primary
  remaining quality bottleneck.
- Scale from the current 12,000-complaint / 35,171-chunk sample to the
  full 464K-complaint / 1.37M-chunk dataset.
- Add response streaming to the UI.
- Expand the evaluation set beyond 8 questions with multi-rater scoring
  for reliability.

## Author

funnali — [GitHub](https://github.com/funnali)
