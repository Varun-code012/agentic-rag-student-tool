# Ask Your Documents — Agentic RAG for Students

A question-answering tool that checks your own uploaded PDFs first, and
automatically falls back to a live web search when your document doesn't
have the answer — instead of guessing or hallucinating.

Originally built as a hackathon project (HackeRing 2.0, Agentic AI track),
now developed further as a free tool for students.

## How it works

1. Upload a PDF (or skip straight to asking a question)
2. Your document is split into chunks and indexed locally with FAISS
3. When you ask something, the system checks whether your document can
   answer it
4. If yes, you get an answer grounded in your own document
5. If not, an agent automatically searches the web and answers from that
   instead
6. Either way, you're told exactly where the answer came from

This is the core idea: not just retrieval, but a system that reasons about
whether its own knowledge is enough before answering.

## Tech stack

- **Streamlit** — the web interface
- **FAISS** — local vector search over your PDF
- **sentence-transformers** — local embeddings (no external API call for this step)
- **Groq** — fast LLM inference for the router decision and answer generation
- **CrewAI** — the web-research agent used as a fallback
- **pdfplumber** — PDF text extraction

## Running it locally

```bash
git clone https://github.com/Varun-code012/agentic-rag-student-tool.git
cd agentic-rag-student-tool
uv sync
```

Create a `.env` file in the project root:

GROQ_API_KEY=your_key_here 


Then run:
```bash
uv run streamlit run app.py
```

## Privacy

Your uploaded file is processed in memory and never saved to disk. Only
the small, relevant snippets needed to answer your specific question are
sent to our LLM provider (Groq) — never the whole document. Nothing
persists after your session ends.

## Known limitations

- **No saved history yet.** Each session is independent — refreshing the
  page clears your uploaded document. Persistent accounts and chat history
  are planned for a future version.
- **Scanned/image-only PDFs aren't supported.** Only PDFs with real,
  selectable text can be read (no OCR yet).
- **Web search responses take longer** than document-based answers, since
  they involve a real web search plus an extra reasoning step — a few
  seconds slower is expected and not a bug.

## Roadmap

- [ ] User accounts and saved chat history (PostgreSQL)
- [ ] OCR support for scanned PDFs
- [ ] Multi-document support per session

## Credits

Built by Varun ([@Varun-code012](https://github.com/Varun-code012)).
