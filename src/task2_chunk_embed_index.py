"""
Task 2: Text Chunking, Embedding, and Vector Store Indexing
CrediTrust Complaint Analysis - RAG Chatbot Project

1. Loads the cleaned dataset from Task 1 (data/filtered_complaints.csv).
2. Draws a stratified sample of 10k-15k complaints, proportional across
   the 4 product categories.
3. Chunks each narrative with LangChain's RecursiveCharacterTextSplitter.
4. Embeds each chunk with sentence-transformers/all-MiniLM-L6-v2.
5. Stores everything in a persisted ChromaDB collection under vector_store/.

Run from the project root:
    python src/task2_chunk_embed_index.py
"""

import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import chromadb

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
INPUT_PATH = "data/filtered_complaints.csv"
SAMPLE_SIZE = 12_000  # within the 10k-15k range required
CHUNK_SIZE = 500  # characters — matches the pre-built store spec
CHUNK_OVERLAP = 50
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
VECTOR_STORE_DIR = "vector_store"
COLLECTION_NAME = "complaint_chunks"

NARRATIVE_COL = "cleaned_narrative"
CATEGORY_COL = "product_category"
ID_COL = "Complaint ID"

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. STRATIFIED SAMPLE
# ---------------------------------------------------------------------------
def stratified_sample(df: pd.DataFrame, total: int) -> pd.DataFrame:
    """Proportional sample across product_category, sized ~= `total` rows."""
    proportions = df[CATEGORY_COL].value_counts(normalize=True)
    sampled_parts = []
    for category, frac in proportions.items():
        n = max(1, round(total * frac))
        group = df[df[CATEGORY_COL] == category]
        n = min(n, len(group))
        sampled_parts.append(group.sample(n=n, random_state=RANDOM_STATE))
    sample = pd.concat(sampled_parts, ignore_index=True)
    print(f"Stratified sample: {len(sample):,} rows")
    print(sample[CATEGORY_COL].value_counts())
    return sample


# ---------------------------------------------------------------------------
# 2. CHUNKING
# ---------------------------------------------------------------------------
def chunk_narratives(df: pd.DataFrame) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    records = []
    for _, row in df.iterrows():
        narrative = str(row[NARRATIVE_COL])
        chunks = splitter.split_text(narrative)
        for i, chunk_text in enumerate(chunks):
            records.append(
                {
                    "complaint_id": str(row[ID_COL]),
                    "product_category": row[CATEGORY_COL],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "text": chunk_text,
                }
            )
    print(f"Produced {len(records):,} chunks from {len(df):,} complaints.")
    return records


# ---------------------------------------------------------------------------
# 3. EMBEDDING + INDEXING
# ---------------------------------------------------------------------------
def embed_and_index(records: list[dict]) -> None:
    print(f"Loading embedding model: {EMBEDDING_MODEL_NAME}")
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [r["text"] for r in records]
    print("Generating embeddings (this may take a few minutes)...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=64)

    client = chromadb.PersistentClient(path=VECTOR_STORE_DIR)
    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    ids = [f"{r['complaint_id']}_{r['chunk_index']}" for r in records]
    metadatas = [
        {
            "complaint_id": r["complaint_id"],
            "product_category": r["product_category"],
            "chunk_index": r["chunk_index"],
            "total_chunks": r["total_chunks"],
        }
        for r in records
    ]

    # Chroma has a max batch size for adds; insert in batches to be safe.
    BATCH = 5000
    for start in range(0, len(records), BATCH):
        end = start + BATCH
        collection.add(
            ids=ids[start:end],
            embeddings=embeddings[start:end].tolist(),
            documents=texts[start:end],
            metadatas=metadatas[start:end],
        )
        print(f"  Indexed {min(end, len(records)):,} / {len(records):,} chunks")

    print(
        f"\nVector store persisted to '{VECTOR_STORE_DIR}/' "
        f"(collection: '{COLLECTION_NAME}', {len(records):,} chunks)."
    )


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    print(f"Loading {INPUT_PATH} ...")
    df = pd.read_csv(INPUT_PATH)
    df = df.dropna(subset=[NARRATIVE_COL, CATEGORY_COL])
    print(f"Loaded {len(df):,} cleaned complaints.")

    sample = stratified_sample(df, SAMPLE_SIZE)
    records = chunk_narratives(sample)
    embed_and_index(records)


if __name__ == "__main__":
    main()
