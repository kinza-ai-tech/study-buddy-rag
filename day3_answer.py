"""
Study Buddy RAG - Day 3
Goal: finish the RAG loop.
    question -> find best chunks (Day 2) -> send them to Gemini -> answer

Install first (terminal, venv active):
    pip install google-genai python-dotenv

Needs a file named .env in this folder containing one line:
    GEMINI_API_KEY=your_key_here

Run:
    python day3_answer.py daa.pdf
"""

import os
import sys
import time                      # lets us wait between retries
from dotenv import load_dotenv      # reads secrets from the .env file
from google import genai            # Google's Gemini library

# Reuse what you built on Day 1 and Day 2
from day2_embeddings import build_index, search

# ---------------------------------------------------------------
# SETTINGS (easy to change)
# ---------------------------------------------------------------
GEMINI_MODEL = "gemini-3.8-flash"   # if this name gives a 404 error, list the
                                    # models available to your key and change it
TOP_K = 6                           # how many chunks we send to Gemini
MIN_SCORE = 0.20                    # below this, we say "not found"


# ---------------------------------------------------------------
# STEP 1: Load the API key from .env (never write it in the code!)
# ---------------------------------------------------------------
load_dotenv()                                   # loads .env into the program
api_key = os.getenv("GEMINI_API_KEY")           # reads the key

if not api_key:
    print("No GEMINI_API_KEY found. Check that your .env file exists")
    print("in this folder and has the line: GEMINI_API_KEY=your_key")
    sys.exit(1)

client = genai.Client(api_key=api_key)          # connection to Gemini


# ---------------------------------------------------------------
# STEP 2: Build the prompt = instructions + chunks + question
# This is the "augmented" part of RAG: we give Gemini the document
# text so it answers from YOUR file instead of guessing.
# ---------------------------------------------------------------
def make_prompt(question, results):
    context = ""
    for chunk, score in results:
        context += f"[Page {chunk['page']}]\n{chunk['text']}\n\n"

    return f"""You are a study assistant. Answer the question using ONLY
the context below, which comes from the student's document.

Rules:
- Explain in simple, clear language.
- Mention the page number(s) you used, like (page 2).
- If the answer is not in the context, say exactly:
  "I couldn't find this in the document."
- Do not use outside knowledge.

CONTEXT:
{context}
QUESTION: {question}

ANSWER:"""


# ---------------------------------------------------------------
# STEP 3: Ask Gemini
# ---------------------------------------------------------------
def answer_question(question, model, chunks, embeddings):
    # Find the best chunks (Day 2)
    results = search(question, model, chunks, embeddings, top_k=TOP_K)

    # If even the best chunk is a weak match, don't guess
    best_score = results[0][1]
    if best_score < MIN_SCORE:
        return "I couldn't find this in the document.", results

    prompt = make_prompt(question, results)

    # Gemini is sometimes busy (error 503) or rate-limited (error 429).
    # These are temporary, so we wait a little and try again (up to 3 times).
    for attempt in range(1, 4):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
            )
            return response.text, results
        except Exception as error:
            message = str(error)
            is_busy = "503" in message or "429" in message or "UNAVAILABLE" in message
            if is_busy and attempt < 3:
                wait = 5 * attempt          # wait 5 seconds, then 10
                print(f"Gemini is busy. Retrying in {wait}s (attempt {attempt}/3)...")
                time.sleep(wait)
                continue
            # Shows the real reason (wrong key, wrong model name, quota...)
            return f"Gemini error: {error}", results


# ---------------------------------------------------------------
# STEP 4: Run it
# ---------------------------------------------------------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python day3_answer.py your_file.pdf")
        sys.exit(1)

    model, chunks, embeddings = build_index(sys.argv[1])
    print(f"\nReady. {len(chunks)} chunks loaded.\n")

    while True:
        question = input("Ask a question (Enter to quit): ").strip()
        if not question:
            break

        answer, results = answer_question(question, model, chunks, embeddings)

        print("\n--- ANSWER ---")
        print(answer)

        print("\n--- SOURCES USED ---")
        for chunk, score in results:
            print(f"page {chunk['page']}  (score {score:.2f})")
        print()
