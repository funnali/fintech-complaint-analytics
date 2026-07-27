"""
Task 3: Qualitative Evaluation of the RAG Pipeline
CrediTrust Complaint Analysis - RAG Chatbot Project

Runs a fixed set of representative questions through the RAG pipeline and
writes a Markdown evaluation table (Question, Generated Answer, Retrieved
Sources, Quality Score placeholder, Comments placeholder) to
evaluation_results.md for inclusion in the final report.

NOTE: Quality Score and Comments are left as placeholders — reviewing and
scoring each answer for accuracy/relevance is a manual judgment call you
make by reading the output, per the assignment instructions.

Run from the project root:
    python src/task3_evaluation.py
"""

from task3_rag_pipeline import ComplaintRAGPipeline

EVAL_QUESTIONS = [
    "Why are customers unhappy with credit cards?",
    "What issues do people report with money transfers?",
    "What problems do customers have with savings accounts?",
    "Why do personal loan customers file complaints?",
    "Are there complaints about unauthorized transactions?",
    "Do customers report problems with customer service response times?",
    "What billing issues are commonly reported for credit cards?",
    "Are there complaints about fraud or scams?",
]

OUTPUT_PATH = "evaluation_results.md"


def format_source(chunk: dict) -> str:
    meta = chunk["metadata"]
    snippet = chunk["text"][:200].replace("\n", " ")
    return (
        f"[{meta['product_category']}, complaint {meta['complaint_id']}] {snippet}..."
    )


def main():
    rag = ComplaintRAGPipeline()

    rows = []
    for q in EVAL_QUESTIONS:
        print(f"\nRunning: {q}")
        result = rag.answer(q)
        sources = [format_source(c) for c in result["sources"][:2]]
        rows.append(
            {
                "question": q,
                "answer": result["answer"],
                "sources": sources,
            }
        )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("# RAG Pipeline Evaluation Results\n\n")
        f.write(
            "| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |\n"
        )
        f.write("|---|---|---|---|---|\n")
        for row in rows:
            question = row["question"].replace("|", "\\|")
            answer = row["answer"].replace("|", "\\|").replace("\n", " ")
            sources = "<br>".join(s.replace("|", "\\|") for s in row["sources"])
            f.write(f"| {question} | {answer} | {sources} | _fill in_ | _fill in_ |\n")

    print(f"\nSaved evaluation table to {OUTPUT_PATH}")
    print(
        "Open it and fill in the Quality Score and Comments columns by reviewing each answer."
    )


if __name__ == "__main__":
    main()
