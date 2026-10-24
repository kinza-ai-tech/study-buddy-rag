"""
Study Buddy RAG - Read the images in a PDF with Gemini (in small batches)

What it does:
  - finds the pictures on each PDF page
  - asks Gemini to copy the text / describe the diagram
  - SAVES every result to <pdfname>_image_notes.json straight away
  - stops after MAX_REQUESTS_PER_RUN requests, so you do not use up the
    free daily limit. Run it again tomorrow: it continues where it stopped.

Run:
    python day4_read_images.py daa.pdf
"""

import os
import sys
import json
import time
import mimetypes
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pypdf import PdfReader

# ---------------------------------------------------------------
# SETTINGS
# ---------------------------------------------------------------
GEMINI_MODEL = "gemini-3.7-flash"   # same model name as day3_answer.py
MAX_REQUESTS_PER_RUN = 10           # images read per run (protects your quota)
MIN_IMAGE_BYTES = 15000             # smaller images are usually logos/icons
SKIP_PAGES = {1}                    # page 1 = cover page with logos

PROMPT = """This image comes from lecture notes about Cloud Computing.
- If it is a table, copy it as text. Put ' | ' between columns and
  put each row on a new line.
- If it is a diagram, list every label and explain in simple words
  what the diagram shows.
- If it is only a logo, photo or decoration with no study content,
  reply with exactly: DECORATION
Write only the content, nothing else."""


# ---------------------------------------------------------------
# Saving and loading results (so we never repeat work)
# ---------------------------------------------------------------
def notes_path_for(pdf_path):
    # daa.pdf -> daa_image_notes.json
    return os.path.splitext(pdf_path)[0] + "_image_notes.json"


def load_notes(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_notes(path, notes):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(notes, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------
# Ask Gemini about one image.
# Returns (text, error). Exactly one of them is None.
# ---------------------------------------------------------------
def read_image(client, image_bytes, mime_type):
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
                    PROMPT,
                ],
            )
            text = response.text
            if not text or not text.strip():
                # Gemini sometimes returns nothing (for example it blocked
                # the image). This is not a crash: we just skip the image.
                reason = ""
                try:
                    reason = str(response.candidates[0].finish_reason)
                except Exception:
                    pass
                return None, "EMPTY " + reason
            return text, None
        except Exception as error:
            message = str(error)
            if "PerDay" in message:           # daily quota: do not retry
                return None, "Daily free-tier limit reached for this model."
            busy = "503" in message or "429" in message or "UNAVAILABLE" in message
            if busy and attempt < 3:
                wait = 5 * attempt
                print(f"  Gemini is busy. Retrying in {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
                continue
            return None, f"Gemini error: {error}"


# ---------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------
def main(pdf_path):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("No GEMINI_API_KEY found. Check your .env file.")
        sys.exit(1)
    client = genai.Client(api_key=api_key)

    notes_path = notes_path_for(pdf_path)
    notes = load_notes(notes_path)
    print(f"Already saved: {len(notes)} image(s) in {notes_path}")

    reader = PdfReader(pdf_path)
    requests_made = 0
    skipped_small = 0
    stopped_early = False

    for page_number, page in enumerate(reader.pages, start=1):
        if page_number in SKIP_PAGES:
            continue
        try:
            images = list(page.images)
        except Exception:
            print(f"Could not read the images on page {page_number}, skipping.")
            continue

        for index, image in enumerate(images, start=1):
            key = f"{page_number}-{index}"
            if key in notes:                       # already done earlier
                continue
            if len(image.data) < MIN_IMAGE_BYTES:  # tiny = logo/icon
                skipped_small += 1
                continue
            if requests_made >= MAX_REQUESTS_PER_RUN:
                stopped_early = True
                break

            mime_type = mimetypes.guess_type(image.name)[0] or "image/png"
            print(f"Reading PDF page {page_number}, image {index} ...")
            text, error = read_image(client, image.data, mime_type)
            requests_made += 1

            if error and error.startswith("EMPTY"):
                print(f"  Gemini returned no text for this image ({error}). Skipping it.")
                continue
            if error:
                print(f"  Stopped: {error}")
                stopped_early = True
                break

            notes[key] = {"page": page_number, "text": text.strip()}
            save_notes(notes_path, notes)          # save after EVERY image
            first_line = text.strip().splitlines()[0] if text.strip() else ""
            print(f"  saved: {first_line[:70]}")

        if stopped_early:
            break

    print(f"\nRequests used this run: {requests_made}")
    print(f"Small images skipped:   {skipped_small}")
    print(f"Total saved so far:     {len(notes)}")
    if stopped_early:
        print("Not finished. Run the same command again later (e.g. tomorrow).")
    else:
        print("All images in this PDF have been processed.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python day4_read_images.py your_file.pdf")
        sys.exit(1)
    main(sys.argv[1])