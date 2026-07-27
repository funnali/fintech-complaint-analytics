"""
Shared utilities and configuration for the CrediTrust RAG pipeline.

Refactored out of the task1-4 scripts so that:
  - core logic is unit-testable without loading any ML models
  - configuration lives in one typed, documented place
  - the same category-mapping / text-cleaning / post-processing logic is
    reused consistently across preprocessing, retrieval, and generation
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RAGConfig:
    """Central, typed configuration for the RAG pipeline."""

    vector_store_dir: str = "vector_store"
    collection_name: str = "complaint_chunks"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    generator_model_name: str = "google/flan-t5-small"
    top_k: int = 5
    context_chunks_for_prompt: int = 3
    max_chars_per_chunk_in_prompt: int = 300
    min_acceptable_answer_len: int = 25


DEFAULT_CONFIG = RAGConfig()


# ---------------------------------------------------------------------------
# NAMED CONSTANTS (previously magic values scattered across scripts)
# ---------------------------------------------------------------------------
MIN_SENTENCE_LENGTH_CHARS: int = 15
MIN_KEYWORD_LENGTH_CHARS: int = 2
EVAL_SNIPPET_LENGTH_CHARS: int = 200


# ---------------------------------------------------------------------------
# PRODUCT CATEGORY MAPPING (Task 1)
# ---------------------------------------------------------------------------
PRODUCT_MAP: dict[str, str] = {
    "Credit card": "Credit Card",
    "Credit card or prepaid card": "Credit Card",
    "Payday loan, title loan, or personal loan": "Personal Loan",
    "Payday loan, title loan, personal loan, or advance loan": "Personal Loan",
    "Payday loan": "Personal Loan",
    "Consumer Loan": "Personal Loan",
    "Checking or savings account": "Savings Account",
    "Bank account or service": "Savings Account",
    "Money transfer, virtual currency, or money service": "Money Transfer",
    "Money transfers": "Money Transfer",
    "Virtual currency": "Money Transfer",
}


def map_product_category(raw_product: str | None) -> str | None:
    """Map a raw CFPB `Product` label onto one of our 4 target categories.

    Returns None if the raw label isn't one of ours (e.g. Mortgage,
    Debt collection, Student loan — out of scope for this project).
    """
    if raw_product is None:
        return None
    return PRODUCT_MAP.get(raw_product)


# ---------------------------------------------------------------------------
# TEXT CLEANING (Task 1)
# ---------------------------------------------------------------------------
BOILERPLATE_PATTERNS: tuple[str, ...] = (
    r"i am writing to file a complaint",
    r"this is in regards to",
    r"to whom it may concern",
    r"i am writing this complaint",
)


def clean_narrative_text(text: str) -> str:
    """Lowercase, strip boilerplate openers, remove special characters."""
    text = text.lower()
    for pattern in BOILERPLATE_PATTERNS:
        text = re.sub(pattern, "", text)
    text = re.sub(r"[^a-z0-9\s.,!?']", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ---------------------------------------------------------------------------
# CATEGORY-AWARE RETRIEVAL (Task 3 improvement)
# ---------------------------------------------------------------------------
CATEGORY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "Credit Card": ("credit card", "creditcard", "card", "cards"),
    "Personal Loan": ("personal loan", "loan", "loans", "payday"),
    "Savings Account": ("savings", "savings account", "checking", "bank account"),
    "Money Transfer": ("money transfer", "transfer", "wire", "remittance"),
}


def detect_question_category(question: str) -> str | None:
    """Best-effort detection of which product category a question implies,
    via simple keyword matching. Returns None if no category is clearly
    implied (in which case retrieval should search all categories)."""
    q_lower = question.lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in q_lower for kw in keywords):
            return category
    return None


# ---------------------------------------------------------------------------
# ANSWER POST-PROCESSING (Task 3 improvement)
# ---------------------------------------------------------------------------
_ARTIFACT_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"excerpt\s*\d+\s*:", re.IGNORECASE),
    re.compile(r"answer\s*\d+\s*:", re.IGNORECASE),
    re.compile(r"^answer\s*:\s*", re.IGNORECASE),
    re.compile(r"question\s*\d*\s*:", re.IGNORECASE),
)


def postprocess_answer(answer: str, question: str | None = None) -> str:
    """Strip known prompt-template leakage artifacts from a generated
    answer (e.g. 'Excerpt 3:', 'Answer:'), and drop a trailing verbatim
    repeat of the question if the model echoed it back."""
    cleaned = answer
    for pattern in _ARTIFACT_PATTERNS:
        cleaned = pattern.sub("", cleaned)

    if question:
        q_stripped = question.strip().rstrip("?").lower()
        if q_stripped and q_stripped in cleaned.lower():
            idx = cleaned.lower().find(q_stripped)
            cleaned = cleaned[:idx].strip()

    return re.sub(r"\s+", " ", cleaned).strip()


# ---------------------------------------------------------------------------
# EXTRACTIVE FALLBACK SCORING (Task 3)
# ---------------------------------------------------------------------------
STOPWORDS: frozenset[str] = frozenset(
    {
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
)


def extract_keywords(question: str, stopwords: frozenset[str] = STOPWORDS) -> set[str]:
    """Extract meaningful keywords from a question for relevance scoring."""
    return {
        w
        for w in re.findall(r"[a-z']+", question.lower())
        if w not in stopwords and len(w) > MIN_KEYWORD_LENGTH_CHARS
    }


def score_sentence_relevance(sentence: str, keywords: set[str]) -> int:
    """Score a sentence by how many question keywords it contains."""
    sentence_words = set(re.findall(r"[a-z']+", sentence.lower()))
    return len(keywords & sentence_words)
