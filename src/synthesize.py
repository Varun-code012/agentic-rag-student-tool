# src/synthesize.py
import os
from groq import Groq

_groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])


def synthesize_answer(question: str, context_chunks: list[str], source: str = "local") -> str:
    """
    Generates a complete answer (non-streaming) from the given context.
    """
    context = "\n\n".join(context_chunks)

    prompt = f"""Answer the question using only the information in the context below.
Be clear and concise. If the context doesn't fully answer the question, say what's missing.

Context:
{context}

Question: {question}

Answer:"""

    response = _groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )

    return response.choices[0].message.content.strip()


def synthesize_answer_stream(question: str, context_chunks: list[str]):
    """
    Same as synthesize_answer, but yields text chunks as they're generated,
    for use with st.write_stream so the answer appears progressively.
    """
    context = "\n\n".join(context_chunks)
    prompt = f"""Answer the question using only the information in the context below.
Be clear and concise.

Context:
{context}

Question: {question}

Answer:"""

    stream = _groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        stream=True,
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            yield content