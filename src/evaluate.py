"""
Hybrid evaluation of generated replies.

Layer 1 — Deterministic checks (objective, zero-cost):
    greeting, sign-off, customer name used, length in range,
    key facts from the original email mentioned.

Layer 2 — LLM-as-judge rubric (1-5 each):
    relevance, correctness vs key facts, tone/empathy fit,
    actionability, safety (no invented commitments).

Composite score per reply = 30% deterministic + 70% rubric,
normalised to 0-100. Overall score = mean of composites.
"""

import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "dataset.json"
REPLIES_PATH = BASE_DIR / "data" / "replies.json"
OUTPUT_PATH = BASE_DIR / "results" / "scores.json"

RUBRIC_DIMENSIONS = ["relevance", "correctness", "tone", "actionability", "safety"]

JUDGE_SYSTEM_PROMPT = """You are a strict customer-support QA reviewer.

Score the reply on five dimensions, each an integer 1-5:
- relevance: does it address THIS customer's actual issue?
- correctness: is it consistent with the key facts of the email? (5 = fully, 1 = contradicts or ignores them)
- tone: does empathy/formality fit the customer's sentiment?
- actionability: is there one clear, concrete next step?
- safety: 5 = no invented policies/amounts/timelines; 1 = fabricates commitments.

Return ONLY valid JSON: {"relevance": n, "correctness": n, "tone": n, "actionability": n, "safety": n, "comment": "one-sentence justification"}"""


# ---------- Layer 1: deterministic checks ----------

def deterministic_checks(email: dict, reply: str) -> dict:
    """Run objective, rule-based checks. Returns dict of booleans + fact coverage."""
    reply_lower = reply.lower()
    word_count = len(reply.split())

    facts_hit = sum(
        1 for fact in email["key_facts"]
        if _fact_mentioned(fact, reply_lower)
    )
    fact_coverage = facts_hit / max(len(email["key_facts"]), 1)

    checks = {
        "has_greeting": bool(re.match(r"^\s*(hi|hello|dear|hey)\b", reply_lower)),
        "has_signoff": "hiver support" in reply_lower,
        "uses_customer_name": email["customer_name"].lower() in reply_lower,
        "length_ok": 40 <= word_count <= 180,
        "fact_coverage": round(fact_coverage, 2),
    }
    bool_score = sum(1 for k, v in checks.items() if k != "fact_coverage" and v) / 4
    checks["deterministic_score"] = round(0.6 * bool_score + 0.4 * fact_coverage, 3)
    return checks


def _fact_mentioned(fact: str, reply_lower: str) -> bool:
    """A fact counts as mentioned if any of its distinctive tokens appear."""
    tokens = [t for t in re.findall(r"[a-z0-9#-]{4,}", fact.lower())]
    if not tokens:
        return False
    hits = sum(1 for t in tokens if t in reply_lower)
    return hits >= max(1, len(tokens) // 3)


# ---------- Layer 2: LLM-as-judge ----------

def judge_reply(email: dict, reply: str) -> dict:
    """Score the reply on the 5-dimension rubric via LLM."""
    user_prompt = (
        f"CUSTOMER EMAIL\nName: {email['customer_name']}\n"
        f"Sentiment: {email['sentiment']}\nSubject: {email['subject']}\n"
        f"Body: {email['body']}\nKey facts: {email['key_facts']}\n\n"
        f"AGENT REPLY\n{reply}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
    )
    scores = json.loads(response.choices[0].message.content)
    # Clamp defensively in case the judge misbehaves
    for dim in RUBRIC_DIMENSIONS:
        scores[dim] = max(1, min(5, int(scores.get(dim, 1))))
    return scores


# ---------- Composite ----------

def composite_score(det: dict, rubric: dict) -> float:
    rubric_avg = sum(rubric[d] for d in RUBRIC_DIMENSIONS) / len(RUBRIC_DIMENSIONS)
    rubric_norm = (rubric_avg - 1) / 4  # map 1-5 -> 0-1
    return round(100 * (0.3 * det["deterministic_score"] + 0.7 * rubric_norm), 1)


def main() -> None:
    dataset = {e["id"]: e for e in json.loads(DATASET_PATH.read_text(encoding="utf-8"))}
    replies = json.loads(REPLIES_PATH.read_text(encoding="utf-8"))

    results: list[dict] = []
    for i, item in enumerate(replies, start=1):
        email = dataset[item["email_id"]]
        print(f"[{i}/{len(replies)}] Evaluating {email['id']} ({email['category']})")

        det = deterministic_checks(email, item["reply"])
        rubric = judge_reply(email, item["reply"])
        score = composite_score(det, rubric)

        results.append({
            "email_id": email["id"],
            "category": email["category"],
            "deterministic": det,
            "rubric": rubric,
            "composite_score": score,
        })

    overall = round(sum(r["composite_score"] for r in results) / len(results), 1)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps({"overall_score": overall, "results": results}, indent=2),
        encoding="utf-8",
    )

    # Console summary table
    print("\n" + "=" * 62)
    print(f"{'ID':<8}{'Category':<20}{'Det':<8}{'Rubric':<8}{'Score':<8}")
    print("-" * 62)
    for r in results:
        rubric_avg = sum(r["rubric"][d] for d in RUBRIC_DIMENSIONS) / 5
        print(f"{r['email_id']:<8}{r['category']:<20}"
              f"{r['deterministic']['deterministic_score']:<8}"
              f"{rubric_avg:<8.1f}{r['composite_score']:<8}")
    print("=" * 62)
    print(f"OVERALL SCORE: {overall}/100")
    print(f"Full results: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
