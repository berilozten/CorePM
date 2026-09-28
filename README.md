# CorePM — A Local RAG Assistant for Product Teams

## Overview

CorePM is a local RAG assistant that helps Product Managers analyze customer feedback and prioritize product issues. It processes customer feedback records, performs semantic search to retrieve the most relevant evidence, and synthesizes structured, evidence-grounded product insights using a deterministic PM priority rubric and a local LLM.

## How it works

Customer Feedback
→ Feedback Parser
→ Local Embeddings
→ Cosine Similarity
→ Top-K Retrieval
→ Local LLM
→ PM Insight

1. **Customer Feedback**: Ingests multi-record customer feedback entries with metadata (`User ID`, `Feature`, `Feedback`).
2. **Feedback Parser**: Robustly parses independent customer feedback records and prepares unit-level chunks.
3. **Local Embeddings**: Generates 1024-dimensional semantic embeddings locally using Microsoft Foundry Local.
4. **Cosine Similarity**: Computes mathematical cosine similarity between the product manager's query and feedback records.
5. **Top-K Retrieval**: Selects the top-3 most relevant customer feedback records along with user IDs and feature tags.
6. **Local LLM**: Generates concise, evidence-grounded product problem summaries, evidence quotes, justifications, and actionable recommendations.
7. **PM Insight**: Applies the deterministic CorePM PM impact rubric to assign authoritative priority (High, Medium, Low) and formats a clean, standardized insight.

## Features

* Local RAG pipeline
* Semantic search
* Top-3 retrieval
* Interactive product questions
* Evidence-grounded PM insights
* Deterministic PM priority rubric
* Runs locally with Microsoft Foundry Local

## Tech Stack

* Python
* Microsoft Foundry Local
* foundry-local-sdk
* Local embedding model (`qwen3-embedding-0.6b-generic-cpu:1`)
* Local generation model (`qwen3-0.6b-generic-gpu:2`)
* Cosine similarity

## Demo Data

The included customer feedback dataset (`docs/feedback.txt`) is synthetic demo data representing diverse customer feedback across onboarding, payments, checkout, search, performance, notifications, and customer support.

## Run locally

### Prerequisites

1. **Python Environment**:
   - Python 3.10+ (recommended: Python 3.12).
   - Create and activate a virtual environment:
     ```bash
     python -m venv corepm_env
     source corepm_env/bin/activate
     ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Microsoft Foundry Local**:
   - Microsoft Foundry Local must be available and running on your machine.
   - The required local models must be available in your local Foundry catalog:
     - Embedding model: `qwen3-embedding-0.6b-generic-cpu:1`
     - Generation model: `qwen3-0.6b-generic-gpu:2`

### Running the Application

Run the application with:

```bash
python app.py
```

Upon launching:
- CorePM will prompt you for an interactive product question.
- Press **Enter** to run the default question (*"What problems are users experiencing during onboarding and what should the product team prioritize?"*), or type your own custom question (e.g., *"What payment and checkout problems are users experiencing?"*).
- CorePM will retrieve the top-3 most relevant feedback records, compute the priority using the PM impact rubric, and output a concise product insight.
