"""
Study Buddy RAG - Image reading TRIAL (one page only)
Goal: test whether Gemini can read the picture on a single PDF page,
before we build image reading for the whole document.

Install first (terminal, venv active):
    pip install pillow

Run (the number is the PDF page, e.g. 8):
    python day4_image_trial.py daa.pdf 8
"""

import os
import sys
import time
import mimetypes                        # guesses image type from the file name
from dotenv import load_dotenv          # reads the key from .env
from google import genai
from google.genai import types          # lets us send an image to Gemini
from pypdf import PdfReader

GEMINI_MODEL = "gemini-3.7-flash"       # same model name as day3_answer.py

PROMPT = """This image comes from lecture notes about Cloud Computing.
- If it is a table, copy it as text. Put ' | ' between columns and
  put each row on a new line.
- If it is a diagram, list every label and explain in simple words
  what the diagram shows.
- If it is only a logo, photo or decoration with no study content,
  reply with exactly: DECORATION
Write only the content, nothing else."""


def ask_gemini_about_image(client, image_bytes, mime_type):
    """Send one image to Gemini and return its text. Retries if busy."""
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    PROMPT,
                ],
            )
            return response.text
        except Exception as error:
            message = str(error)
            busy = "503" in message or "429" in message or "UNAVAILABLE" in message
            if busy and attempt < 3:
                wait = 5 * attempt
                print(f"Gemini is busy. Retrying in {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
                continue
            return f"Gemini error: {error}"


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python day4_image_trial.py your_file.pdf PAGE_NUMBER")
        sys.exit(1)

    pdf_path = sys.argv[1]
    page_number = int(sys.argv[2])          # PDF page number (starts at 1)

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found. Check your .env file.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    page = PdfReader(pdf_path).pages[page_number - 1]
    images = list(page.images)              # all pictures on this page
    print(f"PDF page {page_number} has {len(images)} image(s)\n")

    for number, image in enumerate(images, start=1):
        mime_type = mimetypes.guess_type(image.name)[0] or "image/png"
        print(f"--- Image {number}: {image.name}, {len(image.data)} bytes ---")
        print(ask_gemini_about_image(client, image.data, mime_type))
        print()
