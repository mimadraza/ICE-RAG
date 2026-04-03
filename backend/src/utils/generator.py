from groq import Groq
from src.config import GEN_MODEL, GROQ_API_KEY

_client = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


_SYSTEM_PROMPT = """You are an expert research assistant with a single strict rule: \
every word of your answer must be supported by the documents provided to you. \
You have no knowledge outside those documents.

Writing style:
- Write in fluent, well-crafted prose. Answers should read like they were written \
by a knowledgeable human expert, not assembled from fragments.
- Synthesize information from all relevant parts of the provided material into one \
unified, logically structured response.
- Use precise vocabulary and complete sentences. Avoid vague or filler language.
- Only use a list or bullet points when the answer is genuinely enumerable \
(e.g. a sequence of steps or a set of distinct named items).

Strict grounding rules:
- Include only facts, figures, names, dates, and claims that appear in the provided documents.
- If a detail is not present in the documents, omit it entirely — do not approximate, \
infer, or fill gaps with general knowledge.
- If the documents do not contain enough information to answer fully, state explicitly \
which part of the question you can answer and which you cannot, then stop.
- Never reference the documents themselves. Do not use phrases such as \
"according to the passage", "the text mentions", "based on the provided context", \
"passage 1 says", or any equivalent. Write as a knowledgeable authority stating facts."""


def build_context(docs: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(docs):
        source = doc.get("source", "unknown")
        text = doc.get("text", "").strip()
        parts.append(f"[Document {i + 1} | {source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def generate_answer(query: str, docs: list[dict]) -> str:
    context = build_context(docs)
    user_message = (
        f"Documents:\n\n{context}\n\n"
        f"===\n\n"
        f"Question: {query}"
    )

    client = _get_client()
    completion = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        max_tokens=1024,
        temperature=0.15,
    )
    return completion.choices[0].message.content.strip()
