# RAG Complaint Chatbot — CrediTrust Financial

Intelligent Complaint Analysis for Financial Services: a Retrieval-Augmented
Generation (RAG) chatbot that lets internal teams (Product, Support,
Compliance) ask plain-English questions about customer complaints across
Credit Cards, Personal Loans, Savings Accounts, and Money Transfers.

## Project Structure

```
rag-complaint-chatbot/
├── .vscode/
├── .github/workflows/unittests.yml
├── data/
│   ├── raw/            # place complaints.csv here (gitignored)
│   └── processed/
├── vector_store/        # persisted FAISS/ChromaDB index (gitignored)
├── notebooks/
├── src/
├── tests/
├── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Tasks

- **Task 1** — EDA & preprocessing: `src/task1_eda_preprocessing.py`
- **Task 2** — Chunking, embedding, vector store indexing
- **Task 3** — RAG core logic and evaluation
- **Task 4** — Gradio/Streamlit interactive interface (`app.py`)

## Status

🚧 In progress — see branch history for task-by-task development.
