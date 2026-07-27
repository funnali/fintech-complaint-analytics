"""
Task 3: RAG Core Logic (Retriever + Prompt + Generator)
CrediTrust Complaint Analysis - RAG Chatbot Project

Retrieves relevant complaint chunks from the ChromaDB vector store built in
Task 2, injects them into a grounded prompt template, and generates an
answer using a local Hugging Face LLM (flan-t5-small), with:
  - category-aware retrieval (filters toward the product category implied
    by the question, when one is clearly detected)
  - answer post-processing (strips prompt-template leakage artifacts)
  - an extractive fallback for degenerate LLM outputs

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

from utils import (
    RAGConfig,
    DEFAULT_CONFIG,
    detect_question_category,
    postprocess_answer,
    extract_keywords,
    score_sentence_relevance,
    MIN_SENTENCE_LENGTH_CHARS,
)

PROMPT_TEMPLATE = """You are a financial analyst assistant for CrediTrust. Answer the question in a
full sentence, using only the information in the complaint excerpts below. Do not
repeat the excerpts verbatim; summarize the common themes in your own words. If the
excerpts don't contain the answer, say you don't have enough information.

Complaint excerpts:
{context}

Question: {question}

Answer in one or two full sentences:"""


# ---------------------------------------------------------------------------
# RAG PIPELINE
# ---------------------------------------------------------------------------
class ComplaintRAGPipeline:
    def __init__(self, config: RAGConfig = DEFAULT_CONFIG):
        self.config = config

        print(f"Loading embedding model: {config.embedding_model_name}")
        self.embedder = SentenceTransformer(config.embedding_model_name)

        print(f"Connecting to vector store at '{config.vector_store_dir}'")
        client = chromadb.PersistentClient(path=config.vector_store_dir)
        self.collection = client.get_collection(name=config.collection_name)

        print(f"Loading generator model: {config.generator_model_name}")
        self.gen_tokenizer = AutoTokenizer.from_pretrained(config.generator_model_name)
        self.gen_model = AutoModelForSeq2SeqLM.from_pretrained(config.generator_model_name)

    def retrieve(self, question: str, k: int | None = None) -> list[dict]:
        """Embed the question and return the top-k most similar chunks.

        If the question clearly implies one of our 4 product categories,
        retrieval is filtered to that category first (falling back to an
        unfiltered search if too few results come back).
        """
        k = k or self.config.top_k
        query_embedding = self.embedder.encode([question]).tolist()

        category = detect_question_category(question)
        where_filter = {"product_category": category} if category else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=k,
            where=where_filter,
        )

        # If a category filter returned too few results, fall back to an
        # unfiltered search rather than returning a sparse answer.
        if where_filter and len(results["documents"][0]) < k:
            results = self.collection.query(query_embeddings=query_embedding, n_results=k)

        chunks = []
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            chunks.append({"text": doc, "metadata": meta, "distance": dist})
        return chunks

    def build_prompt(self, question: str, chunks: list[dict]) -> str:
        limited_chunks = chunks[: self.config.context_chunks_for_prompt]
        context = "\n\n".join(
            f"Excerpt {i+1}: {c['text'][:self.config.max_chars_per_chunk_in_prompt]}"
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
        return self.gen_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    def _extractive_fallback(self, question: str, chunks: list[dict]) -> str:
        """Used when the small LLM's answer is too short/degenerate."""
        keywords = extract_keywords(question)
        candidates = []
        for chunk in chunks:
            sentences = re.split(r'(?<=[.!?])\s+', chunk["text"])
            for sent in sentences:
                sent = sent.strip()
                if len(sent) < MIN_SENTENCE_LENGTH_CHARS:
                    continue
                score = score_sentence_relevance(sent, keywords)
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
        llm_answer = postprocess_answer(self._llm_generate(prompt), question=question)

        looks_degenerate = (
            len(llm_answer) < self.config.min_acceptable_answer_len
            or llm_answer.strip().startswith("[")
        )
        if looks_degenerate:
            return self._extractive_fallback(question, chunks)
        return llm_answer

    def answer(self, question: str, k: int | None = None) -> dict:
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
        print(f"\n{'='*80}\nQ: {q}\n{'='*80}")
        result = rag.answer(q)
        print(f"A: {result['answer']}\n")
        print("Top sources:")
        for src in result["sources"][:2]:
            print(f"  - [{src['metadata']['product_category']}] "
                  f"complaint {src['metadata']['complaint_id']}: {src['text'][:150]}...")


if __name__ == "__main__":
    main()
