# Study Buddy RAG

Ask questions about your lecture PDFs and get answers **with page references**.

Study Buddy is a small Retrieval-Augmented Generation (RAG) app written in Python. It reads a PDF, finds the parts that match your question by *meaning* (not just exact words), and asks Google Gemini to write an answer using only those parts. It can also read **diagrams and tables that are stored as pictures** inside the PDF.

> Status: working command-line prototype. A web page (Flask) and deployment are planned, not done yet.

## What it can do

- Answers questions from your own PDF, in simple language, and cites the page (`PDF page 14 / printed page 9`)
- Searches by meaning, so a question can match text that uses different words
- Says **"I couldn't find this in the document"** instead of guessing, using a similarity threshold and a rule in the prompt
- Reads text from images in the PDF (diagrams, tables) with Gemini, in small saved batches
- Retries when Gemini is busy and switches to a backup model if the first one is unavailable

## How it works

1. **Read and clean** the PDF text page by page (`pypdf`), removing characters that break the tokenizer.
2. **Chunk** each page into pieces of 1000 characters with 200 characters of overlap, so an idea is rarely cut in half. Each chunk remembers its page.
3. **Embed** every chunk with the local model `all-MiniLM-L6-v2` (384 numbers per chunk, runs on your computer, no API key needed).
4. **Retrieve**: your question is embedded the same way, and the 6 chunks with the highest cosine similarity are selected. If the best score is below 0.20, the app stops and says it can't find the answer.
5. **Generate**: the 6 chunks and your question go to Gemini with instructions to answer only from that text and to cite the page labels.

Image text follows the same path: `day4_read_images.py` asks Gemini to transcribe each image once, saves the result to `<pdf name>_image_notes.json`, and those notes are added to the chunks automatically.

## Project files

| File | Purpose |
|---|---|
| `day1_pdf_chunker.py` | Read the PDF, clean the text, split into chunks |
| `day2_embeddings.py` | Create embeddings and search by meaning (also loads saved image notes) |
| `day3_answer.py` | The app: retrieve chunks, ask Gemini, print the answer and sources |
| `day4_read_images.py` | Read images in the PDF with Gemini, in small resumable batches |
| `day4_image_trial.py` | Test image reading on a single page |
| `requirements.txt` | Python libraries |

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/kinza-ai-tech/study-buddy-rag.git
cd study-buddy-rag
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once.

Create a free Gemini API key at <https://aistudio.google.com/apikey>, then save it in a file named `.env` in the project folder:

```
GEMINI_API_KEY=your_key_here
```

`.env`, your PDFs and the generated files are listed in `.gitignore`, so they are not uploaded to GitHub. Never share your key.

## Usage

Put your PDF in the folder (the examples use `daa.pdf`).

```powershell
# optional: read the images in the PDF (about 10 per run; run again to continue)
python day4_read_images.py daa.pdf

# start the app (the model is downloaded from Hugging Face on the first run)
python day3_answer.py daa.pdf
```

Type a question at the prompt. Press Enter on an empty line to quit.

After the first run you can tell the app not to contact Hugging Face again:

```powershell
$env:HF_HUB_OFFLINE="1"
```

## Settings

| Setting | File | Default | Meaning |
|---|---|---|---|
| `chunk_size`, `overlap` | `day1_pdf_chunker.py` | 1000, 200 | Size of chunks and repeated text between them |
| `TOP_K` | `day3_answer.py` | 6 | Chunks sent to Gemini |
| `MIN_SCORE` | `day3_answer.py` | 0.20 | Below this similarity the app refuses to answer |
| `PAGE_OFFSET` | `day3_answer.py` | 5 | Front pages before printed page 1 (set to 0 if none) |
| `GEMINI_MODELS` | `day3_answer.py` | `gemini-3.7-flash`, `gemini-3.6-flash` | Models tried in order; model names change over time |
| `MAX_REQUESTS_PER_RUN` | `day4_read_images.py` | 10 | Images read per run |

## What I tested and learned

I tested the app on one 89-page lecture-notes PDF about Cloud Computing.

- **Chunk size matters.** With 500-character chunks, a definition was cut in half and "what are the service models" listed only two of the three models. With 1000-character chunks it listed all three.
- **Content inside images is invisible to a text reader.** The comparison of IaaS, PaaS and SaaS existed only as a picture, so the app answered "I couldn't find this". After reading the images with Gemini, the same question returned a correct comparison with page citations.
- **Out-of-scope questions are refused.** Questions unrelated to the document (similarity about 0.07) are stopped before calling Gemini. A related question whose answer was not in the text was also refused by the prompt rule.
- **Real-world failures need handling.** The free Gemini tier returned busy errors (503) and a daily request limit (429), and PDFs contained characters that crashed the embedding step. The code now retries, falls back to a second model, saves progress after every image, and cleans the text.

## Known limitations

- Text-based PDFs and simple diagrams only; handwriting and complex figures are not guaranteed to be read correctly. Always check important answers against the page.
- Image reading and answering use Gemini, so they are limited by the free tier (daily quota, busy servers). Reading all images in a long PDF can take several days of runs.
- Questions that need the whole document (summaries, "list everything") may be incomplete, because only the top 6 chunks are sent.
- Comparison questions can need chunks from several places, and one search may not find them all.
- Embeddings are rebuilt on every start, which takes 10 to 20 seconds for a PDF of this size.
- Page labels assume the printed page numbers continue without gaps after the front pages.

## Roadmap

- [ ] Save and reuse embeddings so startup is fast
- [ ] Flask web page with PDF upload
- [ ] Deploy a demo
- [ ] Hybrid search (meaning + exact keywords) and multi-query search for comparison questions

## Author

Kinza Nawaz, <https://github.com/kinza-ai-tech>