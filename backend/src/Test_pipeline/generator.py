import os
from groq import Groq

from config import GEN_MODEL,GROQ_API_KEY

# GEN_PROVIDER = os.getenv("GEN_PROVIDER", "auto")  # no longer needed

client = Groq(api_key=GROQ_API_KEY)

def build_prompt(query: str, docs: list[dict]) -> str:
    context = "\n\n".join(
        f"[{i+1}] {doc['text']}" for i, doc in enumerate(docs)
    )
    return f"""Answer only from the provided context.
If the answer is not supported by the context, say you do not know.

Question:
{query}

Context:
{context}

Answer:
"""

def generate_answer(query: str, docs: list[dict]) -> str:
    prompt = build_prompt(query, docs)

    completion = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1,
    )
    return completion.choices[0].message.content.strip()