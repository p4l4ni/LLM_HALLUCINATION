from pymongo import MongoClient
import pandas as pd
from collections import defaultdict


MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "hbs_db"
COLLECTION_NAME = "hbs_logs"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]
collection = db[COLLECTION_NAME]


records = list(collection.find({}))
if not records:
    print("⚠️ No records found in MongoDB. Run multiprobe.py first.")
    exit()

df = pd.DataFrame(records)
df = df[df['mean_KL'].notna() & df['mean_delta_rank'].notna()]


df['norm_KL'] = df.groupby('model')['mean_KL'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
df['norm_delta'] = df.groupby('model')['mean_delta_rank'].transform(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-9))
df['risk_score'] = 0.7 * df['norm_KL'] + 0.3 * df['norm_delta']


agg = df.groupby(['model', 'entity']).agg({
    'mean_KL': 'mean',
    'mean_delta_rank': 'mean',
    'risk_score': 'mean',
    'classification': lambda x: x.value_counts().index[0]
}).reset_index()


for model in agg['model'].unique():
    print(f"\n🚀 Top Hallucination-Prone Entities for {model}:")
    subset = agg[agg['model'] == model].sort_values(by='risk_score', ascending=False).head(10)
    for _, row in subset.iterrows():
        print(f"  - {row['entity']:20} | Risk={row['risk_score']:.3f} | KL={row['mean_KL']:.3f} | Type={row['classification']}")


agg.to_csv("entity_risk_scores.csv", index=False)
print("\n✅ Saved aggregated results to entity_risk_scores.csv")
