# openai-challenge-hiver

AI-powered customer support email reply generator with automated response quality scoring.

---

## Overview

This project was built for the Hiver OpenAI Challenge.

It generates customer support replies using an LLM and evaluates how good each generated reply is using multiple quality metrics.

---

## Features

- Generate a synthetic customer support dataset
- Generate AI-powered email replies
- Evaluate generated replies
- Save evaluation scores automatically

---

## Project Structure

```text
email-ai/
│
├── assets/
│   ├── dataset-generation.png
│   ├── dataset_replies.png
│   └── dataset_evaluation.png
│
├── data/
│   ├── dataset.json
│   └── replies.json
│
├── results/
│   └── scores.json
│
├── src/
│   ├── generate_dataset.py
│   ├── generate_replies.py
│   └── evaluate.py
│
├── requirements.txt
└── README.md
```

---

# Installation

Clone the repository

```bash
git clone https://github.com/haripriyatripathi/openai-challenge-hiver.git
```

Go inside the project

```bash
cd openai-challenge-hiver
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file

```env
OPENAI_API_KEY=your_openai_api_key
```

---

# Step 1 - Generate Dataset

Run

```bash
python .\src\generate_dataset.py
```

Output

![Dataset Generation](assets/dataset-generation.png)

Creates a synthetic dataset containing customer support emails for:

- Billing Issues
- Refund Requests
- Bug Reports
- Feature Questions
- Account Access

Output file

```text
data/dataset.json
```

---

# Step 2 - Generate AI Replies

Run

```bash
python .\src\generate_replies.py
```

Output

![Reply Generation](assets/dataset_replies.png)

Generates AI responses for every email in the dataset.

Output file

```text
data/replies.json
```

---

# Step 3 - Evaluate Responses

Run

```bash
python .\src\evaluate.py
```

Output

![Evaluation](assets/dataset_evaluation.png)

Evaluates every generated response and calculates an overall quality score.

Output file

```text
results/scores.json
```

---

## Evaluation Metrics

The response quality is measured using:

- Semantic Similarity
- BERTScore
- ROUGE-L
- Retrieval Confidence
- Response Length

Each response receives an individual score along with an overall system score.

---

## Tech Stack

- Python
- OpenAI API
- LangChain
- FAISS
- Sentence Transformers
- BERTScore

---

## Run Everything

```bash
python .\src\generate_dataset.py

python .\src\generate_replies.py

python .\src\evaluate.py
```

---

## AI Usage

ChatGPT was used for brainstorming, debugging, implementation guidance, and documentation.

The project structure, implementation, testing, and final integration were completed manually.

---

## Repository

https://github.com/haripriyatripathi/openai-challenge-hiver