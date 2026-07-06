"""
Generates a synthetic dataset of customer support emails.

Each record contains the incoming email plus ground-truth metadata
(category, sentiment, key facts) that the evaluator later uses to
verify that generated replies actually address the customer's issue.
"""

import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    api_key=os.getenv("GROQ_API_KEY"),
    base_url="https://api.groq.com/openai/v1",
)

MODEL = "llama-3.3-70b-versatile"
NUM_EMAILS = 24  # 4 per category
OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "dataset.json"

CATEGORIES = [
    "billing_issue",
    "refund_request",
    "bug_report",
    "angry_escalation",
    "feature_question",
    "account_access",
]

SYSTEM_PROMPT = """You generate realistic customer support emails for a SaaS product called Hiver (shared inbox tool for Gmail).

Return ONLY valid JSON (no markdown fences, no commentary) with this exact schema:
{
  "emails": [
    {
      "id": "string, e.g. em_001",
      "category": "one of the given categories",
      "customer_name": "realistic first name",
      "sentiment": "positive | neutral | frustrated | angry",
      "subject": "email subject line",
      "body": "the full email body, 40-120 words, realistic tone",
      "key_facts": ["2-4 concrete facts a good reply MUST address, e.g. order ID, error message, plan name"]
    }
  ]
}"""


def generate_batch(category: str, count: int) -> list[dict]:
    """Generate `count` emails for a single category."""
    user_prompt = (
        f"Generate {count} distinct customer support emails in category '{category}'. "
        "Vary sentiment, writing style, and specifics (invent ticket IDs, plan names, "
        "error messages). Include concrete details in key_facts."
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.9,
        response_format={"type": "json_object"},
    )
    payload = json.loads(response.choices[0].message.content)
    return payload["emails"]


def main() -> None:
    per_category = NUM_EMAILS // len(CATEGORIES)
    dataset: list[dict] = []

    for category in CATEGORIES:
        print(f"Generating {per_category} emails for: {category}")
        emails = generate_batch(category, per_category)
        dataset.extend(emails)

    # Re-assign sequential IDs so they are guaranteed unique across batches
    for i, email in enumerate(dataset, start=1):
        email["id"] = f"em_{i:03d}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(dataset, indent=2), encoding="utf-8")
    print(f"\nWrote {len(dataset)} emails to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
