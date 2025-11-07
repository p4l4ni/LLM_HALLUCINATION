# LLM_HALLUCINATION
#  Hallucination Behavior Signatures: Profiling and Mining LLM Response Volatility via NoSQL Context Memory

**Author:** Navitha E  
**Institution:** Vellore Institute of Technology, Chennai, India  
**Contact:** navitha.e2022@vitstudent.ac.in  

---

##  Overview

This project investigates **hallucination patterns in Large Language Models (LLMs)** by analyzing how their responses change when internal representations are slightly perturbed.  
The goal is to **profile and quantify “response volatility”** across different LLMs using both **open-source** and **closed-source** systems.

The system uses a **NoSQL-based memory (MongoDB)** to log response metrics and performs **entity-level risk mining** to identify which topics trigger the most hallucinations.

---

##  Core Idea

LLMs sometimes generate text that sounds plausible but is factually incorrect — this is known as **hallucination**.  
By slightly modifying (perturbing) model embeddings and comparing how much the output shifts, we can **measure the sensitivity** of the model to factual instability.

The idea is to compute:
- **How much the response distribution changes (KL Divergence)**  
- **How much the predicted word order drifts (ΔRank)**  

From these, a **risk score** is derived that measures how hallucination-prone each entity or model is.

---

##  Models Evaluated

| Category | Model | Params | Source | Purpose |
|-----------|--------|--------|---------|----------|
| Open | GPT-2 | 117M | OpenAI | Legacy baseline |
| Open | Facebook OPT-125M | 125M | Meta AI | Lightweight GPT-style model |
| Open | EleutherAI GPT-Neo-125M | 125M | EleutherAI | Open GPT-3-like architecture |
| Closed | Gemini-Pro | Billions | Google DeepMind | Modern reference model |

These models were selected to ensure diversity in:
- **Architecture (Transformer variants)**  
- **Scale (100M → multi-billion)**  
- **Openness (Open vs. API-locked)**  

---

##  System Architecture

1. **Dataset** — TruthfulQA (200 factual question-answer pairs)
2. **Probing Engine** — Generates original and perturbed responses
3. **Metrics Calculator** — Computes KL Divergence & Rank Delta
4. **MongoDB Context Memory** — Stores all model responses & metrics
5. **Pattern Miner** — Aggregates risk per entity & model
6. **Visualization** — Plots hallucination risk and cross-model comparisons

---

## ⚙️ Setup Instructions

### 1️. Prerequisites
- Python 3.10+
- MongoDB installed locally (e.g., `D:\MongoDB\server\bin`)
- Internet access for API models
- Required libraries:

### 2️. Folder Structure
D:\SEM7\FDA\project
│
├── TruthfulQA.csv                # Dataset
├── open_probe.py                 # For GPT-2, OPT-125M, GPT-Neo
├── closed_probe.py               # For Gemini, Grok, Perplexity
├── closed_providers.json         # Closed model API configs
├── pattern_mine.py               # Risk aggregation and charts
├── storage.py                    # MongoDB logger
└── entity_risk_scores.csv        # Final results (auto-generated)

### 3️.MongoDB Setup

- Start MongoDB before running any probe:

net start MongoDB
mongosh

- Stop it after experiments:

net stop MongoDB

### 4.Running the Project
-  Run Open LLM Experiments
python open_probe.py

-  Run Closed LLM Experiments
python closed_probe.py

-  Aggregate and Analyze Risk
python pattern_mine.py
```bash
pip install torch transformers pandas numpy matplotlib pymongo tqdm requests
