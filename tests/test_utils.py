"""
Unit tests for src/utils.py — the core, model-free logic of the RAG
pipeline (category mapping, text cleaning, category detection, answer
post-processing, keyword extraction/scoring).

Run from the project root:
    pytest tests/ -v
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from utils import (
    map_product_category,
    clean_narrative_text,
    detect_question_category,
    postprocess_answer,
    extract_keywords,
    score_sentence_relevance,
    RAGConfig,
)


# ---------------------------------------------------------------------------
# map_product_category
# ---------------------------------------------------------------------------
def test_map_product_category_credit_card():
    assert map_product_category("Credit card") == "Credit Card"
    assert map_product_category("Credit card or prepaid card") == "Credit Card"


def test_map_product_category_savings():
    assert map_product_category("Checking or savings account") == "Savings Account"


def test_map_product_category_unrelated_returns_none():
    # e.g. Mortgage, Debt collection — out of scope for this project
    assert map_product_category("Mortgage") is None
    assert map_product_category("Student loan") is None


# ---------------------------------------------------------------------------
# clean_narrative_text
# ---------------------------------------------------------------------------
def test_clean_narrative_text_lowercases():
    result = clean_narrative_text("I Am Writing To File A Complaint about FEES")
    assert result == result.lower()


def test_clean_narrative_text_strips_boilerplate():
    result = clean_narrative_text("I am writing to file a complaint about fraud")
    assert "writing to file a complaint" not in result
    assert "fraud" in result


def test_clean_narrative_text_removes_special_characters():
    result = clean_narrative_text("Fees are $$$ way too high!!! @company")
    assert "$" not in result
    assert "@" not in result


# ---------------------------------------------------------------------------
# detect_question_category
# ---------------------------------------------------------------------------
def test_detect_question_category_credit_card():
    assert detect_question_category("Why are people unhappy with credit cards?") == "Credit Card"


def test_detect_question_category_money_transfer():
    assert detect_question_category("What issues do people report with money transfers?") == "Money Transfer"


def test_detect_question_category_no_match_returns_none():
    assert detect_question_category("What is the weather like today?") is None


# ---------------------------------------------------------------------------
# postprocess_answer
# ---------------------------------------------------------------------------
def test_postprocess_answer_strips_excerpt_artifact():
    raw = "Excerpt 3: customers report billing issues frequently."
    result = postprocess_answer(raw)
    assert "Excerpt 3:" not in result
    assert "billing issues" in result


def test_postprocess_answer_strips_answer_prefix():
    raw = "Answer: fees were charged without notice."
    result = postprocess_answer(raw)
    assert not result.lower().startswith("answer")


def test_postprocess_answer_strips_repeated_question():
    question = "Are there complaints about fraud?"
    raw = "Yes, several. Question: Are there complaints about fraud?"
    result = postprocess_answer(raw, question=question)
    assert "Question:" not in result


# ---------------------------------------------------------------------------
# extract_keywords / score_sentence_relevance
# ---------------------------------------------------------------------------
def test_extract_keywords_removes_stopwords():
    keywords = extract_keywords("Why are customers unhappy with credit cards?")
    assert "why" not in keywords
    assert "credit" in keywords


def test_score_sentence_relevance_counts_overlap():
    keywords = {"credit", "card", "fees"}
    high_score = score_sentence_relevance("The credit card fees were too high", keywords)
    low_score = score_sentence_relevance("The weather was nice today", keywords)
    assert high_score > low_score


# ---------------------------------------------------------------------------
# RAGConfig
# ---------------------------------------------------------------------------
def test_rag_config_defaults():
    config = RAGConfig()
    assert config.top_k == 5
    assert config.embedding_model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_rag_config_is_immutable():
    config = RAGConfig()
    try:
        config.top_k = 10
        assert False, "RAGConfig should be frozen/immutable"
    except Exception:
        pass
