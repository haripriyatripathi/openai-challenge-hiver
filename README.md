# openai-challenge-hiver

an ai-powered customer support email reply system that generates context-aware responses using an llm and evaluates their quality through multiple response metrics.

---

## overview

this project was built for the hiver openai challenge.

the system:

- creates a synthetic customer support email dataset
- generates suggested replies using an llm
- evaluates each generated reply
- produces per-response and overall quality scores

the project runs end-to-end and is fully reproducible.

---

## features

- synthetic email dataset generation
- llm-powered response generation
- retrieval-based grounding using previous examples
- automated response evaluation
- per-response scoring
- overall system quality score

---

## project structure

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

## installation

clone the repository

```bash
git clone https://github.com/haripriyatripathi/openai-challenge-hiver.git
```

move into the project

```bash
cd openai-challenge-hiver
```

install dependencies

```bash
pip install -r requirements.txt
```

create a `.env` file

```env
groq_api_key=your_api_key
```

---

## step 1 - generate dataset

```bash
python src/generate_dataset.py
```

creates a synthetic dataset containing customer support emails across multiple categories including:

- billing issues
- refund requests
- bug reports
- feature questions
- account access
- angry escalation

output:

```text
data/dataset.json
```

---

## step 2 - generate replies

```bash
python src/generate_replies.py
```

generates suggested replies for every email using an llm grounded on previous examples.

output:

```text
data/replies.json
```

---

## step 3 - evaluate replies

```bash
python src/evaluate.py
```

evaluates every generated reply and produces individual scores together with an overall system score.

output:

```text
results/scores.json
```

---

## evaluation methodology

exact text matching is not suitable for email generation because multiple responses can correctly answer the same customer request.

instead, each reply is evaluated using several complementary metrics:

- semantic similarity
- bertscore
- rouge-l
- retrieval confidence
- response length

these metrics together measure meaning preservation, response completeness, grounding quality, and readability.

the final system score is calculated from the combined evaluation of all generated responses.

---

## technology stack

- python
- groq api
- llama 3.3 70b
- langchain
- faiss
- sentence transformers
- bertscore

---

## run the complete pipeline

```bash
python src/generate_dataset.py
python src/generate_replies.py
python src/evaluate.py
```

---

## ai usage

claude was used to assist with code boilerplate, implementation ideas, and debugging during development.

all project integration, customization, testing, evaluation pipeline, and final implementation decisions were completed manually.
