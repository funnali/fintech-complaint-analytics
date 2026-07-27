"""
Task 4: Interactive Chat Interface
CrediTrust Complaint Analysis - RAG Chatbot Project

A Gradio app that lets non-technical users ask questions about customer
complaints and see both the generated answer and the source excerpts it
was grounded in.

Run from the project root:
    python app.py
Then open the local URL Gradio prints (usually http://127.0.0.1:7860).
"""

import gradio as gr
from src.task3_rag_pipeline import ComplaintRAGPipeline

print("Starting up — loading models and connecting to vector store...")
rag = ComplaintRAGPipeline()
print("Ready.")


def ask_question(question: str):
    if not question or not question.strip():
        return "Please enter a question.", ""

    result = rag.answer(question)

    sources_md = ""
    for i, src in enumerate(result["sources"][:5], start=1):
        meta = src["metadata"]
        sources_md += (
            f"**Source {i}** — {meta['product_category']} "
            f"(complaint #{meta['complaint_id']})\n\n"
            f"> {src['text']}\n\n---\n\n"
        )

    return result["answer"], sources_md


def clear_all():
    return "", "", ""


with gr.Blocks(title="CrediTrust Complaint Insights") as demo:
    gr.Markdown("# CrediTrust Complaint Insights")
    gr.Markdown(
        "Ask a plain-English question about customer complaints across "
        "Credit Cards, Personal Loans, Savings Accounts, and Money Transfers. "
        "Answers are generated from real retrieved complaint excerpts, shown below."
    )

    with gr.Row():
        question_box = gr.Textbox(
            label="Your question",
            placeholder="e.g. Why are customers unhappy with credit cards?",
            lines=2,
        )

    with gr.Row():
        submit_btn = gr.Button("Ask", variant="primary")
        clear_btn = gr.Button("Clear")

    answer_box = gr.Textbox(label="Answer", lines=4, interactive=False)
    sources_box = gr.Markdown(label="Sources")

    submit_btn.click(
        fn=ask_question,
        inputs=question_box,
        outputs=[answer_box, sources_box],
    )
    clear_btn.click(
        fn=clear_all,
        inputs=None,
        outputs=[question_box, answer_box, sources_box],
    )

if __name__ == "__main__":
    demo.launch()
