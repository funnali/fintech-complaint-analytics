"""
Task 3: RAG Core Logic (Retriever + Prompt + Generator)
CrediTrust Complaint Analysis - RAG Chatbot Project

Retrieves relevant complaint chunks from the ChromaDB vector store built in
Task 2, injects them into a grounded prompt template, and generates an
answer using a local Hugging Face LLM (flan-t5-small), with an extractive
fallback for when the small LLM produces a degenerate/too-short answer.

Run from the project root:
    python src/task3_rag_pipeline.py
"""

import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

import re
from collections import Counter

import chromadb
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "complaint_chunks"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GENERATOR_MODEL_NAME = "google/flan-t5-small"
TOP_K = 5

CONTEXT_CHUNKS_FOR_PROMPT = 3
MAX_CHARS_PER_CHUNK_IN_PROMPT = 300
MIN_ACCEPTABLE_ANSWER_LEN = 25  # shorter/degenerate answers trigger the fallback

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Answer the question in a
full sentence, using only the information in the complaint excerpts below. Do not
repeat the excerpts verbatim; summarize the common themes in your own words. If the
excerpts don't contain the answer, say you don't have enough information.

Complaint excerpts:
{context}

Question: {question}

Answer in one or two full sentences:"""

STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "do",
    "does",
    "why",
    "what",
    "how",
    "with",
    "about",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "it",
    "this",
    "that",
    "people",
    "customers",
    "unhappy",
    "report",
    "issues",
}


# ---------------------------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------------------------
class ComplaintRAGPipeline:
    def __init__(self):
        print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

        print(f"Connecting to vector store at '{VECTOR_STORE_DIR}'")
        client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
        self.collection = client.get_collection(name=COLLECTION_NAME)

        print(f"Loading generator model: {GENERATOR_MODEL_NAME}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(GENERATOR_MODEL_NAME)

    def retrieve(self, question: str, k: int = TOP_K) -> list[dict]:
        query_embedding = self.embedder.encode([question]).tolist()
        results = self.collection.query(query_embeddings=query_embedding, n_results=k)
        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        limited_chunks = chunks[:CONTEXT_CHUNKS_FOR_PROMPT]
        # Plain excerpts, numbered — no bracket "tags" for the model to latch onto.
        context = "\n\n".join(
            f"Excerpt {i + 1}: {c['text'][:MAX_CHARS_PER_CHUNK_IN_PROMPT]}"
            for i, c in enumerate(limited_chunks)
        )
        return PROMPT_TEMPLATE.format(context=context, question=question)

    def _llm_generate(self, prompt: str) -> str:
        inputs = self.gen_tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=512
        )
        output_ids = self.gen_model.generate(
            **inputs,
            max_new_tokens=200,
            min_new_tokens=15,
            num_beams=4,
            no_repeat_ngram_size=3,
        )
        return self.gen_tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

    def _extractive_fallback(self, question: str, chunks: list[dict]) -> str:
        """Used when the small LLM's answer is too short/degenerate. Scores
        sentences in the retrieved chunks by keyword overlap with the
        question and stitches the best ones into a grounded answer."""
        keywords = {
            w
            for w in re.findall(r"[a-z']+", question.lower())
            if w not in STOPWORDS and len(w) > 2
        }
        candidates = []
        for chunk in chunks:
            sentences = re.split(r"(?<=[.!?])\s+", chunk["text"])
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < 15:
                    continue
                sent_words = set(re.findall(r"[a-z']+", sent.lower()))
                score = len(keywords & sent_words)
                candidates.append((score, sent, chunk["metadata"]["product_category"]))

        candidates.sort(key=lambda x: x[0], reverse=True)
        top = [c for c in candidates if c[0] > 0][:4] or candidates[:4]
        if not top:
            return "I don't have enough information in the retrieved complaints to answer this."

        top_category = Counter(c[2] for c in top).most_common(1)[0][0]
        return (
            f"Based on retrieved complaints (mostly {top_category}), customers report: "
            + " ".join(s for _, s, _ in top)
        )

    def generate(self, question: str, chunks: list[dict]) -> str:
        prompt = self.build_prompt(question, chunks)
        llm_answer = self._llm_generate(prompt)

        # Guard against degenerate small-LLM outputs (e.g. echoing a label,
        # or a suspiciously short non-answer).
        looks_degenerate = len(
            llm_answer
        ) < MIN_ACCEPTABLE_ANSWER_LEN or llm_answer.strip().startswith("[")
        if looks_degenerate:
            return self._extractive_fallback(question, chunks)
        return llm_answer

    def answer(self, question: str, k: int = TOP_K) -> dict:
        chunks = self.retrieve(question, k=k)
        answer_text = self.generate(question, chunks)
        return {
            "question": question,
            "answer": answer_text,
            "sources": chunks,
        }


# ---------------------------------------------------------------------------
# DEMO / MANUAL TEST
# ---------------------------------------------------------------------------
def main():
    rag = ComplaintRAGPipeline()

    demo_questions = [
        "Why are customers unhappy with credit cards?",
        "What issues do people report with money transfers?",
    ]

    for q in demo_questions:
        print(f"\n{'=' * 80}\nQ: {q}\n{'=' * 80}")
        result = rag.answer(q)
        print(f"A: {result['answer']}\n")
        print("Top sources:")
        for src in result["sources"][:2]:
            print(
                f"  - [{src['metadata']['product_category']}] "
                f"complaint {src['metadata']['complaint_id']}: {src['text'][:150]}..."
            )


if __name__ == "__main__":
    main()
