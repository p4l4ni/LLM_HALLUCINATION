"""
open_probe.py
-------------
Probes open-weight LLMs (GPT-2, OPT-125M, GPT-Neo-125M)
using the full TruthfulQA dataset to detect hallucination sensitivity.

Stores results in MongoDB.
"""

import os
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymongo import MongoClient
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import spacy


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_PATH = r"D:\SEM7\FDA\project\TruthfulQA.csv"
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "hbs_db"
COLL_NAME = "open_logs"

OPEN_MODELS = ["gpt2", "facebook/opt-125m", "EleutherAI/gpt-neo-125M"]


client = MongoClient(MONGO_URI)
collection = client[DB_NAME][COLL_NAME]


def softmax_np(logits):
    e = np.exp(logits - np.max(logits))
    return e / e.sum()

def kl_div(p, q, eps=1e-12):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    return float(np.sum(p * np.log(p / q)))


nlp = None
def get_entity(text):
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess, sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    if doc.ents:
        return doc.ents[0].text
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            return token.text
    return text.split()[0]


def probe_open_model(model_name, prompt, entity):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :].cpu().numpy()
        orig_probs = softmax_np(logits)

    emb_layer = model.get_input_embeddings()
    token_id = tokenizer.encode(entity, add_special_tokens=False)[0]
    vec = emb_layer.weight[token_id].detach().clone()

    KLs, deltas = [], []
    for _ in range(3):  
        noise = torch.tensor(np.random.normal(scale=0.05, size=vec.shape), dtype=torch.float32).to(DEVICE)
        noisy_vec = vec + noise
        with torch.no_grad():
            backup = emb_layer.weight[token_id].clone()
            emb_layer.weight[token_id] = noisy_vec
            pert_logits = model(**inputs).logits[0, -1, :].cpu().numpy()
            emb_layer.weight[token_id] = backup

        pert_probs = softmax_np(pert_logits)
        KLs.append(kl_div(orig_probs, pert_probs))
        orig_top = np.argsort(-orig_probs)[:20]
        pert_top = np.argsort(-pert_probs)
        rank_diffs = np.mean([abs(np.where(pert_top == t)[0][0] - i) for i, t in enumerate(orig_top)])
        deltas.append(rank_diffs)

    mean_KL = float(np.mean(KLs))
    mean_delta = float(np.mean(deltas))

    if mean_KL >= 1.0:
        classification = "Contradictory"
    elif mean_KL >= 0.5:
        classification = "Slight Drift"
    else:
        classification = "Aligned"

    rec = {
        "type": "open",
        "model": model_name,
        "prompt": prompt,
        "entity": entity,
        "mean_KL": mean_KL,
        "mean_delta_rank": mean_delta,
        "classification": classification,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    }
    return rec


def load_truthfulqa(path):
    df = pd.read_csv(path)
    df = df.dropna(subset=["Question"], how="any")
    return df["Question"].tolist()

if __name__ == "__main__":
    prompts = load_truthfulqa(DATASET_PATH)
    prompts = prompts[:200]
    print(f"Loaded {len(prompts)} TruthfulQA prompts")

    for model_name in OPEN_MODELS:
        print(f"\n🚀 Probing model: {model_name}")
        for p in tqdm(prompts, desc=f"Probing {model_name}"):
            try:
                entity = get_entity(p)
                res = probe_open_model(model_name, p, entity)
                collection.insert_one(res)
            except Exception as e:
                print(f"⚠️ Skipped: {p[:50]} | Error: {e}")

    print("\n Done. All open model results saved in MongoDB.")
