"""
Study Buddy RAG - Day 1
Goal: read a PDF and split its text into small "chunks", remembering
which page each chunk came from.

Install first (in your terminal):
    pip install pypdf

Run:
    python day1_pdf_chunker.py my_slides.pdf
"""

import sys                      # lets us read the PDF name from the command line
from pypdf import PdfReader     # library that reads PDF files


# ---------------------------------------------------------------
# STEP 1: Read the PDF, page by page
# ---------------------------------------------------------------
def load_pdf(path):
    """Return a list of (page_number, text) for every page in the PDF."""
    reader = PdfReader(path)    # open the PDF file
    pages = []                  # we will store results here

    # enumerate gives us a counter; start=1 so pages begin at 1, not 0
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""   # get text; "" if the page has none
        text = " ".join(text.split())      # clean extra spaces/new lines
        if text:                           # skip empty pages (e.g. image-only)
            pages.append((page_number, text))

    return pages


# ---------------------------------------------------------------
# STEP 2: Split text into chunks
# Why chunks? An AI can't search a whole book at once. Small pieces
# let us find only the parts that match a question.
# Why overlap? So a sentence cut at the edge still appears whole
# in the next chunk.
# ---------------------------------------------------------------
def split_into_chunks(pages, chunk_size=500, overlap=100):
    """Turn pages into chunks of ~chunk_size characters."""
    chunks = []

    for page_number, text in pages:
        start = 0                              # where the chunk begins
        while start < len(text):
            end = start + chunk_size           # where the chunk ends
            chunk_text = text[start:end]       # cut out the piece

            chunks.append({
                "page": page_number,           # remember the source page
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