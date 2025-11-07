import os
import json
import time
import re
import requests
from tqdm import tqdm
from pymongo import MongoClient
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np
from datetime import datetime
import spacy


TRUTHFULQA_PATH = r"D:\SEM7\FDA\project\TruthfulQA.csv"
PROVIDER_CONFIG_PATH = r"D:\SEM7\FDA\project\closed_providers.json"
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "hallucination_db"
COLLECTION_NAME = "closed_logs"


client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]

def load_provider_configs(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

providers = load_provider_configs(PROVIDER_CONFIG_PATH)


def load_truthfulqa(path):
    df = pd.read_csv(path)
    prompts = df["Question"].dropna().tolist()[:200]
    print(f"Loaded {len(prompts)} TruthfulQA prompts")
    return prompts

prompts = load_truthfulqa(TRUTHFULQA_PATH)


embedder = SentenceTransformer('all-MiniLM-L6-v2')
from sklearn.preprocessing import normalize

def kl_divergence(p, q):
    """Compute symmetric KL divergence between normalized distributions."""
    p = np.asarray(p)
    q = np.asarray(q)
    
    
    p = np.abs(p)
    q = np.abs(q)
    p = p / np.sum(p)
    q = q / np.sum(q)
    
    
    eps = 1e-10
    p = p + eps
    q = q + eps
    
    return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
"""
def kl_divergence(p, q):
    
    p = np.asarray(p) + 1e-10
    q = np.asarray(q) + 1e-10
    return 0.5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))
"""
def classify_hallucination(kl_value):
    if kl_value < 0.2:
        return "Aligned"
    elif kl_value < 0.8:
        return "Slight Drift"
    else:
        return "Contradictory"


nlp = None

def get_entity(text):
    """Extract the most relevant entity from text using spaCy NER."""
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except:
            import subprocess
            import sys
            subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
            nlp = spacy.load("en_core_web_sm")

    doc = nlp(text)
    
    
    if doc.ents:
        return doc.ents[0].text
    
    
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN"):
            return token.text
    
    
    return text.split()[0] if text.split() else "unknown"


def call_provider_http(provider_cfg, prompt, timeout=20):
    url = provider_cfg["url"]
    headers = provider_cfg.get("headers", {}).copy()
    api_key = provider_cfg.get("key")
    if api_key:
        if "Authorization" not in headers:
            headers["Authorization"] = f"Bearer {api_key}"

    payload_template = provider_cfg["payload_template"]
    payload_str = json.dumps(payload_template).replace("{prompt}", prompt)
    payload = json.loads(payload_str)

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except json.JSONDecodeError:
        
        return resp.text.strip()
    except Exception as e:
        raise RuntimeError(f"Request failed: {e}")

    
    key_path = provider_cfg.get("response_key", "")
    val = data
    for part in re.split(r"\.|\[|\]", key_path):
        if not part:
            continue
        if part.isdigit():
            val = val[int(part)]
        else:
            val = val.get(part, {})
    return val if isinstance(val, str) else str(val)


def probe_closed_provider(provider_name, provider_cfg, prompt):
    try:
        text_orig = call_provider_http(provider_cfg, prompt)
        noisy_prompt = prompt + " (in a parallel universe)"
        text_noisy = call_provider_http(provider_cfg, noisy_prompt)

        emb_orig = embedder.encode(text_orig, normalize_embeddings=True)
        emb_noisy = embedder.encode(text_noisy, normalize_embeddings=True)

        kl_val = kl_divergence(emb_orig, emb_noisy)
        classification = classify_hallucination(kl_val)

        record = {
            "provider": provider_name,
            "prompt": prompt,
            "orig_response": text_orig,
            "noisy_response": text_noisy,
            "mean_KL": float(kl_val),
            "classification": classification,
            "timestamp": datetime.now().isoformat(),
        }

        
        record["entity"] = get_entity(prompt)

        collection.insert_one(record)
        print(f"{provider_name}: {record['entity']} | {classification} | KL={kl_val:.3f}")
        return record

    except Exception as e:
        raise RuntimeError(f"{provider_name} failed for prompt '{prompt[:60]}': {str(e)}")


for provider_name, provider_cfg in providers.items():
    print(f"\n🚀 Probing provider: {provider_name}")
    for prompt in tqdm(prompts, desc=provider_name):
        try:
            probe_closed_provider(provider_name, provider_cfg, prompt)
            time.sleep(1.5)  # rate-limit safeguard
        except Exception as e:
            print(f"⚠️ Skipped: {prompt[:60]} | Error: {str(e)}")

print("\n Closed LLM probing completed and stored in MongoDB.")