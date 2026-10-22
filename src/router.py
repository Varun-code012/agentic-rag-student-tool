# src/router.py
import os
import json
from groq import Groq
from sentence_transformers import SentenceTransformer

_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def retrieve_top_chunks(question: str, index, chunks: list[str], k: int = 3) -> list[str]:
    question_embedding = _embedding_model.encode([question]).astype("float32")
    distances, positions = index.search(question_embedding, k)
    return [chunks[i] for i in positions[0]]


def answer_or_route(question: str, context_chunks: list[str]) -> dict:
    """
    ONE Groq call that both decides sufficiency AND writes the answer if
    the context is enough — replacing two separate calls (router + synthesis)
    with one, cutting local-path latency roughly in half.
    Returns: {"sufficient": bool, "answer": str}
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""Context:
{context}

Question: {question}

Decide if the context above is enough to answer the question confidently.
Respond with ONLY valid JSON, no other text, in this exact shape:
{{"sufficient": true, "answer": "the answer here"}}
or
{{"sufficient": false, "answer": ""}}"""

    response = _groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        response_format={"type": "json_object"},  # forces valid JSON output
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        # If the model ever returns malformed JSON, fail safe to "not sufficient"
        # rather than crash the app — this triggers the web fallback instead.
        return {"sufficient": False, "answer": ""}