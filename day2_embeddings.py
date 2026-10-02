"""
Study Buddy RAG - Day 2
Goal: turn every chunk into an "embedding" (a list of numbers that
captures its MEANING), save them, and search them with a question.

Install first (in your terminal, venv active):
    pip install sentence-transformers

Run (same folder as day1_pdf_chunker.py):
    python day2_embeddings.py daa.pdf
"""

import sys
import json
import numpy as np                                   # numbers/maths library
from sentence_transformers import SentenceTransformer  # makes embeddings

# We reuse the functions you built on Day 1
from day1_pdf_chunker import load_pdf, split_into_chunks

# A small, free model that runs on your own laptop (no API key needed).
# It is downloaded automatically the first time (~90 MB).
MODEL_NAME = "all-MiniLM-L6-v2"


# ---------------------------------------------------------------
# STEP 1: Build the "index" = chunks + their embeddings
# ---------------------------------------------------------------
def build_index(pdf_path):
    pages = load_pdf(pdf_path)               # Day 1: read the PDF
    chunks = split_into_chunks(pages)        # Day 1: cut into chunks

    model = SentenceTransformer(MODEL_NAME)  # load the embedding model

    # Take only the text of each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Turn every text into a vector (list of 384 numbers).
    # normalize_embeddings=True makes every vector length 1, which
    # makes comparing them simple and fast (explained below).
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    # Save both to disk so we don't need to recompute every time
    np.save("embeddings.npy", embeddings)              # the numbers
    with open("chunks.json", "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)  # the text + page

    return model, chunks, embeddings


# ---------------------------------------------------------------
# STEP 2: Search by meaning
# ---------------------------------------------------------------
def search(question, model, chunks, embeddings, top_k=3):
    # Turn the QUESTION into a vector using the same model
    q_vector = model.encode([question], normalize_embeddings=True)[0]

    # Similarity score between the question and every chunk.
    # Because vectors are normalized, this multiplication equals
    # "cosine similarity": closer to 1.0 = more similar meaning.
    scores = embeddings @ q_vector

    # argsort sorts low->high, [::-1] flips it to high->low,
    # [:top_k] keeps only the best few
    best_indexes = np.argsort(scores)[::-1][:top_k]

    return [(chunks[i], float(scores[i])) for i in best_indexes]


# ---------------------------------------------------------------
# STEP 3: Run it and try questions
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python day2_embeddings.py your_file.pdf")
        sys.exit(1)

    model, chunks, embeddings = build_index(sys.argv[1])
    print(f"\nChunks: {len(chunks)}")
    print(f"Each embedding has {embeddings.shape[1]} numbers\n")

    # Keep asking questions until you press Enter on an empty line
    while True:
        question = input("Ask a question (Enter to quit): ").strip()
        if not question:
            break

        for chunk, score in search(question, model, chunks, embeddings):
            print(f"\n[score {score:.2f}] page {chunk['page']}")
            print(chunk["text"])
        print()