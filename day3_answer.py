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
# Models are tried in this order. If the first one is busy or out of daily
# quota, the app automatically tries the next one (each model has its own limit).
GEMINI_MODELS = ["gemini-3.7-flash", "gemini-3.6-flash"]
TOP_K = 6                           # how many chunks we send to Gemini
MIN_SCORE = 0.20                    # below this, we say "not found"
PAGE_OFFSET = 5                     # PDF page 6 = printed page 1, so 6 - 1 = 5
                                    # set to 0 if your PDF has no extra front pages


def page_label(pdf_page):
    """Show the PDF page and, when it exists, the printed page number."""
    printed = pdf_page - PAGE_OFFSET
    if PAGE_OFFSET > 0 and printed >= 1:
        return f"PDF page {pdf_page} / printed page {printed}"
    return f"PDF page {pdf_page}"


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
        context += f"[{page_label(chunk['page'])}]\n{chunk['text']}\n\n"

    return f"""You are a study assistant. Answer the question using ONLY
the context below, which comes from the student's document.

Rules:
- Explain in simple, clear language.
- Mention the page label(s) you used, exactly as written in the
  brackets, like (PDF page 14 / printed page 9).
- If the answer is not in the context, say exactly:
  "I couldn't find this in the document."
- Do not use outside knowledge.

CONTEXT:
{context}
QUESTION: {question}

ANSWER:"""


def ask_gemini(prompt):
    """Try each model in GEMINI_MODELS. Returns (text, error); one is None."""
    last_error = ""
    for model_name in GEMINI_MODELS:
        for attempt in range(1, 3):                 # 2 tries per model
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt,
                )
                text = response.text
                if text and text.strip():
                    if model_name != GEMINI_MODELS[0]:
                        text += f"\n\n(answered by backup model {model_name})"
                    return text, None
                last_error = "Gemini returned an empty reply."
                break                               # try the next model
            except Exception as error:
                message = str(error)
                last_error = message
                if "PerDay" in message:             # daily quota: next model
                    print(f"Daily limit reached for {model_name}.")
                    break
                if "404" in message or "NOT_FOUND" in message:
                    print(f"Model {model_name} is not available. Trying the next one.")
                    break
                if "503" in message or "429" in message or "UNAVAILABLE" in message:
                    if attempt < 2:
                        print(f"{model_name} is busy. Retrying in 10s...")
                        time.sleep(10)
                        continue
                    print(f"{model_name} is still busy.")
                    break
                return None, f"Gemini error: {message}"   # any other error: stop
    return None, ("All models are busy or out of daily quota. Try again later. "
                  "Last error: " + last_error[:200])


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

    text, error = ask_gemini(prompt)
    if error:
        return error, results
    return text, results


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
            print(f"{page_label(chunk['page'])}  (score {score:.2f})")
        print()