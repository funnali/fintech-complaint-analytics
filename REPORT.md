# Interim Report — Intelligent Complaint Analysis for Financial Services

**Project:** RAG-Powered Complaint Chatbot for CrediTrust Financial
**Author:** funnali
**Covers:** Task 1 (EDA & Preprocessing) and Task 2 (Chunking, Embedding, Indexing)

---

## Task 1: Exploratory Data Analysis and Preprocessing

### Key Findings

The raw CFPB complaint dataset contains **9,609,797 records** spanning many
product categories, only a subset of which fall under CrediTrust's four
lines of business (Credit Card, Personal Loan, Savings Account, Money
Transfer). A first pass over the `Product` field showed that CFPB has
renamed and split these categories multiple times over the years — for
example, credit card complaints appear under both `"Credit card"` and
`"Credit card or prepaid card"`, and money transfer complaints appear under
three different labels. We addressed this by building an explicit mapping
from each of the raw CFPB product labels onto one of our four clean target
categories, rather than filtering on a single exact string match.

Narrative coverage is a major factor in how much usable data exists for a
RAG pipeline: of the 9.6M total complaints, only **2,980,756 (31%)** include
a free-text consumer narrative, while the remaining **6,629,041 (69%)** are
metadata-only records (e.g., routed complaints with no accompanying
written description). Since a semantic-search chatbot depends entirely on
narrative text, these narrative-less records are of no use to the RAG
pipeline and were dropped.

Among complaints that do have a narrative, word counts are wide-ranging: the
mean is **175.6 words**, but the median is much lower at **114 words**,
indicating a right-skewed distribution with a long tail of very detailed
complaints (up to 6,469 words) alongside many short one- or two-sentence
entries (minimum 1 word). The 25th–75th percentile range of **59–209 words**
guided our chunking strategy in Task 2: most narratives are short enough
that they don't need many chunks, but a meaningful minority are long enough
that single-vector embedding would lose important detail.

After filtering to the four target categories and removing empty
narratives, the cleaned dataset contains **480,580 complaints**, distributed
as follows:

| Product Category | Count | Share |
|---|---|---|
| Credit Card | 189,334 | 39.4% |
| Savings Account | 155,204 | 32.3% |
| Money Transfer | 98,701 | 20.5% |
| Personal Loan | 37,341 | 7.8% |

This shows a clear class imbalance — Credit Card complaints outnumber
Personal Loan complaints by roughly 5:1. This imbalance was carried forward
proportionally into the Task 2 sample (rather than a naive equal split
across categories) so that Task 3's retrieval results reflect real-world
complaint volume, though it's worth flagging that Personal Loan questions
may retrieve from a comparatively small evidence base.

### Preprocessing Steps Applied

- Loaded the raw CSV in chunks of 200,000 rows (the full file is too large
  to load into memory at once) and loaded only the 8 columns needed
  downstream, to keep memory usage manageable.
- Mapped 11 raw CFPB `Product` labels onto the four target categories.
- Dropped rows with missing or empty narratives.
- Lowercased all narrative text, stripped special characters, and removed
  common boilerplate openers (e.g., "I am writing to file a complaint...").
- Saved the result to `data/filtered_complaints.csv` (480,580 rows).

---

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

### Sampling Strategy

From the cleaned 480,580-row dataset, we drew a **stratified sample of
12,000 complaints**, sized within the required 10,000–15,000 range and
proportional to each category's real-world share:

| Product Category | Sampled Rows |
|---|---|
| Credit Card | 4,728 |
| Savings Account | 3,875 |
| Money Transfer | 2,465 |
| Personal Loan | 932 |

Sampling proportionally (rather than equally) keeps the sample
representative of CrediTrust's actual complaint mix, so that patterns
learned or retrieved at this stage generalize to the full dataset used in
Tasks 3–4.

### Chunking Approach

We used LangChain's `RecursiveCharacterTextSplitter` with **chunk_size=500
characters** and **chunk_overlap=50 characters**. This choice was
deliberately matched to the specification of the pre-built vector store
provided for Tasks 3–4, so that our own Task 2 pipeline and the full-scale
pre-built index are directly comparable rather than using two incompatible
chunking schemes.

Given the narrative length distribution from Task 1 (median 114 words,
which is roughly 600–700 characters), a 500-character chunk size means most
narratives split into just 1–2 chunks, while longer narratives in the upper
quartile (200+ words) split into several overlapping chunks. The 50-character
overlap helps preserve context across chunk boundaries (e.g., a sentence
describing a specific transaction date or dollar amount isn't cut cleanly
in half). Applying this to our 12,000-complaint sample produced **35,171
total chunks** — roughly 2.9 chunks per complaint on average.

### Embedding Model Choice

We used **`sentence-transformers/all-MiniLM-L6-v2`**, as specified in the
challenge brief. This model was chosen because it:

- Produces compact 384-dimensional embeddings (~80MB model size), making it
  fast enough to run on CPU-only hardware for both indexing and query-time
  embedding — important since this pipeline needs to scale to the full
  1.37M-chunk pre-built store in later tasks.
- Is widely used for semantic similarity tasks and performs well on
  general-domain English text, which matches the informal, first-person
  writing style of CFPB narratives.
- Is the same model used to build the provided pre-built vector store,
  ensuring our Task 2 embeddings and the Task 3/4 pre-built embeddings live
  in the same vector space and are directly comparable.

### Vector Store

All 35,171 chunks were embedded and persisted into a local **ChromaDB**
collection (`complaint_chunks`) under `vector_store/`. Each chunk's metadata
includes its source `complaint_id`, `product_category`, `chunk_index`, and
`total_chunks`, so any retrieved chunk can be traced back to its original
complaint for verification — a requirement carried through to the Task 4
UI's "show sources" feature.

---

## Next Steps (Task 3–4)

- Build the retriever + prompt template + LLM generation pipeline against
  the full pre-built vector store (1.37M chunks).
- Run and score 5–10 representative evaluation questions.
- Build the Gradio/Streamlit interface with source display and streaming.
