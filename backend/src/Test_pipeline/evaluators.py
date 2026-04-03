from groq import Groq
import json
import os
import requests
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from config import EMBED_MODEL, GROQ_API_KEY, GEN_MODEL

client_groq = Groq(api_key=GROQ_API_KEY)


def hf_complete(prompt: str, max_new_tokens: int = 300) -> str:
    """
    Call Groq inference via the OpenAI-compatible chat completions format.
    """
    payload = {
        "model": GEN_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    resp = client_groq.chat.completions.create(**payload)
    try:
        return resp.choices[0].message.content.strip()
    except Exception as e:
        raise RuntimeError(f"Unexpected Groq response format: {resp}") from e


def extract_json_list(raw: str) -> list[str]:
    """
    Try to parse a JSON list even if the model wraps it in markdown fences.
    """
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:].strip()

    try:
        data = json.loads(raw)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    return []


def extract_json_dict(raw: str) -> dict:
    """
    Try to parse a JSON dict even if the model wraps it in markdown fences.
    """
    raw = raw.strip()

    if raw.startswith("```"):
        raw = raw.strip("`")
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
    prompt = f"""
Extract atomic factual claims from the answer below.
Return ONLY a valid JSON array of strings.
Do not include markdown fences or explanation.

Answer:
{answer}
""".strip()

    raw = hf_complete(prompt, max_new_tokens=200)
    return extract_json_list(raw)


def verify_claims(claims: list[str], context: str) -> tuple[float, list[dict]]:
    results = []
    supported = 0

    for claim in claims:
        prompt = f"""
You are verifying whether a claim is supported by the context.

Return ONLY valid JSON in this exact shape:
{{"label": "supported" or "unsupported", "reason": "..."}}

Claim:
{claim}

Context:
{context}
""".strip()

        raw = hf_complete(prompt, max_new_tokens=120)
        item = extract_json_dict(raw)

        if not item:
            item = {"label": "unsupported", "reason": "parse_error"}

        item["claim"] = claim
        results.append(item)

        if item.get("label") == "supported":
            supported += 1

    score = supported / len(claims) if claims else 0.0
    return score, results


class RelevancyScorer:
    def __init__(self):
        self.model = SentenceTransformer(EMBED_MODEL)

    def generate_alt_questions(self, answer: str) -> list[str]:
        prompt = f"""
Generate 3 alternative user questions that could be answered by the answer below.
Return ONLY a valid JSON array of strings.
Do not include markdown fences or explanation.

Answer:
{answer}
""".strip()

        raw = hf_complete(prompt, max_new_tokens=120)
        data = extract_json_list(raw)

        if isinstance(data, list):
            return data[:3]
        return []

    def score(self, query: str, alt_questions: list[str]) -> tuple[float, list[float]]:
        if not alt_questions:
            return 0.0, []

        q_emb = self.model.encode([query])
        alt_emb = self.model.encode(alt_questions)
        sims = cosine_similarity(q_emb, alt_emb)[0]
        return float(np.mean(sims)), [float(x) for x in sims]


def evaluate_run(query: str, answer: str, docs: list[dict]) -> dict:
    context = "\n\n".join(d["text"] for d in docs)

    claims = extract_claims(answer)
    faithfulness, verification = verify_claims(claims, context)

    rel = RelevancyScorer()
    alt_qs = rel.generate_alt_questions(answer)
    relevancy, rel_scores = rel.score(query, alt_qs)

    return {
        "faithfulness": faithfulness,
        "claims": claims,
        "claim_verification": verification,
        "alt_questions": alt_qs,
        "relevancy": relevancy,
        "relevancy_scores": rel_scores,
    }