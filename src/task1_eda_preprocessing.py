"""
Task 1: Exploratory Data Analysis and Preprocessing
CrediTrust Complaint Analysis - RAG Chatbot Project

Reads the raw CFPB CSV in CHUNKS (it's several million rows, too big to
load fully into memory on most laptops), filters down to the four target
product categories with non-empty narratives, cleans the text, and saves
the much smaller result to disk.

Run from the project root:
    python src/task1_eda_preprocessing.py
"""

import re
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from utils import PRODUCT_MAP, clean_narrative_text

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
RAW_DATA_PATH = "data/raw/complaints.csv"
OUTPUT_PATH = "data/filtered_complaints.csv"
CHUNK_SIZE = 200_000  # rows per chunk; lower this (e.g. 50_000) if you still hit memory errors

NARRATIVE_COL = "Consumer complaint narrative"
PRODUCT_COL = "Product"

# Only load the columns we actually need — this is the main memory saver.
USE_COLS = [
    "Complaint ID",
    "Date received",
    PRODUCT_COL,
    "Issue",
    "Sub-issue",
    "Company",
    "State",
    NARRATIVE_COL,
]

def main():
    print(f"Reading {RAW_DATA_PATH} in chunks of {CHUNK_SIZE:,} rows...")

    total_rows = 0
    rows_with_narrative = 0
    rows_without_narrative = 0
    raw_product_counts = Counter()
    narrative_word_counts = []
    filtered_chunks = []

    reader = pd.read_csv(
        RAW_DATA_PATH,
        usecols=USE_COLS,
        chunksize=CHUNK_SIZE,
        low_memory=False,
    )

    for i, chunk in enumerate(reader, start=1):
        total_rows += len(chunk)

        raw_product_counts.update(chunk[PRODUCT_COL].dropna().tolist())

        has_narrative = chunk[NARRATIVE_COL].notna() & (chunk[NARRATIVE_COL].str.strip() != "")
        rows_with_narrative += has_narrative.sum()
        rows_without_narrative += (~has_narrative).sum()

        # Sample word-count stats from a subset to keep memory bounded
        sample = chunk.loc[has_narrative, NARRATIVE_COL].str.split().str.len()
        narrative_word_counts.extend(sample.tolist())

        # Map to our 4 target categories and keep only matching + non-empty rows
        chunk = chunk.copy()
        chunk["product_category"] = chunk[PRODUCT_COL].map(PRODUCT_MAP)
        keep = chunk["product_category"].notna() & has_narrative
        matched = chunk.loc[keep].copy()

        if len(matched):
            matched["cleaned_narrative"] = matched[NARRATIVE_COL].apply(clean_narrative_text)
            filtered_chunks.append(matched)

        print(f"  chunk {i}: {len(chunk):,} rows read, "
              f"{len(matched):,} matched target categories so far this chunk "
              f"(running total kept: {sum(len(c) for c in filtered_chunks):,})")

    # ------------------------------------------------------------------
    # EDA summary
    # ------------------------------------------------------------------
    print(f"\n=== TOTAL rows in raw file: {total_rows:,} ===")
    print("\n=== Product distribution (top 15, raw labels) ===")
    for name, count in Counter(raw_product_counts).most_common(15):
        print(f"  {name}: {count:,}")

    print(f"\nComplaints WITH narrative:    {rows_with_narrative:,}")
    print(f"Complaints WITHOUT narrative: {rows_without_narrative:,}")

    word_counts = pd.Series(narrative_word_counts)
    print("\n=== Narrative word-count stats (rows with narrative) ===")
    print(word_counts.describe())

    plt.figure(figsize=(8, 5))
    word_counts.clip(upper=word_counts.quantile(0.99)).hist(bins=50)
    plt.title("Distribution of Consumer Narrative Word Counts (99th pct clipped)")
    plt.xlabel("Word count")
    plt.ylabel("Number of complaints")
    plt.tight_layout()
    plt.savefig("data/narrative_length_distribution.png")
    print("\nSaved histogram to data/narrative_length_distribution.png")

    # ------------------------------------------------------------------
    # Final filtered + cleaned dataset
    # ------------------------------------------------------------------
    df_final = pd.concat(filtered_chunks, ignore_index=True)
    print(f"\n=== Final filtered dataset: {len(df_final):,} rows ===")
    print(df_final["product_category"].value_counts())

    df_final.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved cleaned & filtered dataset to {OUTPUT_PATH} "
          f"({len(df_final):,} rows).")


if __name__ == "__main__":
    main()
