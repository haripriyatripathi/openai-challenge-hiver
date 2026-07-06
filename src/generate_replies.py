"""
Generates AI support replies for every email in data/dataset.json.

The system prompt encodes Hiver-style support guidelines: acknowledge
the issue, reference the customer's specifics, give a clear next step,
match tone to sentiment, and never invent policies or commitments.
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
BASE_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = BASE_DIR / "data" / "dataset.json"
OUTPUT_PATH = BASE_DIR / "data" / "replies.json"

SYSTEM_PROMPT = """You are a senior customer support agent for Hiver, a shared-inbox tool for Gmail.

Write a reply email following these rules:
1. Greet the customer by name.
2. Acknowledge their specific issue in the first two sentences.
3. Reference the concrete details they mentioned (IDs, plan names, error messages).
4. Give ONE clear, actionable next step (what you will do, or what they should do).
5. Match tone to their sentiment: extra empathy for frustrated/angry customers.
6. NEVER invent refund amounts, timelines, policies, or promises not implied by the email.
7. 60-140 words. Professional but warm. Sign off as "Priya, Hiver Support".

Return ONLY the reply email body as plain text. No subject line, no JSON."""


def generate_reply(email: dict) -> str:
    """Generate a single support reply for one customer email."""
    user_prompt = (
        f"Customer name: {email['customer_name']}\n"
        f"Sentiment: {email['sentiment']}\n"
        f"Subject: {email['subject']}\n\n"
        f"Email:\n{email['body']}"
    )
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def main() -> None:
    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    replies: list[dict] = []

    for i, email in enumerate(dataset, start=1):
        print(f"[{i}/{len(dataset)}] Replying to {email['id']} ({email['category']})")
        reply_text = generate_reply(email)
        replies.append({"email_id": email["id"], "reply": reply_text})

    OUTPUT_PATH.write_text(json.dumps(replies, indent=2), encoding="utf-8")
    print(f"\nWrote {len(replies)} replies to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
