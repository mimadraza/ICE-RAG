import json
import numpy as np
from groq import Groq
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import EMBED_MODEL, GROQ_API_KEY, GEN_MODEL

_client = None
_embed_model = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        _client = Groq(api_key=GROQ_API_KEY)
    return _client


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _llm(prompt: str, max_tokens: int = 300) -> str:
    client = _get_client()
    resp = client.chat.completions.create(
        model=GEN_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def _parse_json_list(raw: str) -> list:
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _parse_json_dict(raw: str) -> dict:
    raw = raw.strip().strip("`")
    if raw.startswith("json"):
        raw = raw[4:].strip()
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def extract_claims(answer: str) -> list[str]:
    prompt = f"""Extract atomic factual claims from the answer below.
Return ONLY a valid JSON array of strings. No markdown fences.

Answer:
{answer}""".strip()
    return _parse_json_list(_llm(prompt, 200))


def verify_claims(claims: list[str], context: str) -> tuple[float, list[dict]]:
    results = []
    supported = 0
    for claim in claims:
        prompt = f"""Verify whether this claim is supported by the context.
Return ONLY valid JSON: {{"label": "supported" or "unsupported", "reason": "..."}}

Claim: {claim}
Context: {context}""".strip()
        item = _parse_json_dict(_llm(prompt, 120))
        if not item:
            item = {"label": "unsupported", "reason": "parse_error"}
        item["claim"] = claim
        results.append(item)
        if item.get("label") == "supported":
            supported += 1
    score = supported / len(claims) if claims else 0.0
    return score, results


def evaluate_run(query: str, answer: str, docs: list[dict]) -> dict:
    context = "\n\n".join(d["text"] for d in docs)
    claims = extract_claims(answer)
    faithfulness, verification = verify_claims(claims, context)

    # Relevancy via alternative questions
    alt_prompt = f"""Generate 3 alternative questions that could be answered by:
{answer}
Return ONLY a JSON array of strings."""
    alt_qs = _parse_json_list(_llm(alt_prompt, 120))[:3]

    relevancy, rel_scores = 0.0, []
    if alt_qs:
        model = _get_embed_model()
        q_emb = model.encode([query])
        alt_emb = model.encode(alt_qs)
        sims = cosine_similarity(q_emb, alt_emb)[0]
        relevancy = float(np.mean(sims))
        rel_scores = [float(x) for x in sims]

    return {
        "faithfulness": faithfulness,
        "relevancy": relevancy,
        "claims": claims,
        "claim_verification": verification,
        "alt_questions": alt_qs,
        "relevancy_scores": rel_scores,
    }
