"""
Study Buddy RAG - Day 1 (updated)
Goal: read a PDF and split its text into small "chunks", remembering
which page each chunk came from.

NEW in this version: clean_text() removes strange characters that some
PDFs contain. Those characters can crash the embedding step later.

Install first (in your terminal):
    pip install pypdf

Run:
    python day1_pdf_chunker.py my_slides.pdf
"""

import re                       # tools for finding/replacing text patterns
import sys                      # lets us read the PDF name from the command line
from pypdf import PdfReader     # library that reads PDF files


# ---------------------------------------------------------------
# NEW: clean the text from a PDF
# Real PDFs (slides, math symbols, special fonts) can produce
# invalid characters. The embedding model refuses them, so we
# remove them here.
# ---------------------------------------------------------------
def clean_text(text):
    # 1) Drop characters that cannot be stored as normal text
    #    (for example broken "surrogate" characters from math fonts).
    text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")

    # 2) Replace hidden control characters with a space
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)

    # 3) Collapse extra spaces and new lines into single spaces
    return " ".join(text.split())


# ---------------------------------------------------------------
# STEP 1: Read the PDF, page by page
# ---------------------------------------------------------------
def load_pdf(path):
    """Return a list of (page_number, text) for every page in the PDF."""
    reader = PdfReader(path)    # open the PDF file
    pages = []                  # we will store results here

    # enumerate gives us a counter; start=1 so pages begin at 1, not 0
    for page_number, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""   # get text; "" if the page has none
        except Exception:
            # a badly formed page should not stop the whole program
            print(f"Warning: could not read page {page_number}, skipping it.")
            continue

        text = clean_text(text)                # clean it (NEW)
        if text:                               # skip empty pages (e.g. image-only)
            pages.append((page_number, text))

    return pages


# ---------------------------------------------------------------
# STEP 2: Split text into chunks
# Why chunks? An AI can't search a whole book at once. Small pieces
# let us find only the parts that match a question.
# Why overlap? So a sentence cut at the edge still appears whole
# in the next chunk.
# ---------------------------------------------------------------
def split_into_chunks(pages, chunk_size=1000, overlap=200):
    """Turn pages into chunks of ~chunk_size characters."""
    chunks = []

    for page_number, text in pages:
        start = 0                              # where the chunk begins
        while start < len(text):
            end = start + chunk_size           # where the chunk ends
            chunk_text = text[start:end].strip()   # cut out the piece

            if chunk_text:                     # never keep an empty chunk
                chunks.append({
                    "page": page_number,       # remember the source page
                    "text": chunk_text,
                })

            # move forward, but step back by "overlap" characters
            start = end - overlap

    return chunks


# ---------------------------------------------------------------
# STEP 3: Run everything and show the result
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python day1_pdf_chunker.py your_file.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]

    pages = load_pdf(pdf_path)
    chunks = split_into_chunks(pages)

    print(f"Pages with text: {len(pages)}")
    print(f"Total chunks:    {len(chunks)}\n")

    # show the first 3 chunks so you can check they look correct
    for i, chunk in enumerate(chunks[:3], start=1):
        print(f"--- Chunk {i} (page {chunk['page']}) ---")
        print(chunk["text"])
        print()
