import json
from pipeline import load_pipeline
from search.search_engine import search_eurovision

def precision_at_k(retrieved, relevant, k=5):

    retrieved_k = retrieved[:k]

    relevant_found = len(
        set(retrieved_k).intersection(set(relevant))
    )

    return relevant_found / k

def recall_at_k(retrieved, relevant, k=5):

    if len(relevant) == 0:
        return 0 
     
    retrieved_k = retrieved[:k]

    if len(retrieved_k) == 0:
        return 0

    relevant_found = len(
        set(retrieved_k).intersection(set(relevant))
    )

    return relevant_found / len(relevant)

def hit_rate_at_k(retrieved, relevant, k=5):

    retrieved_k = retrieved[:k]

    return int(
        len(set(retrieved_k).intersection(set(relevant))) > 0
    )

# Load system
df, bm25, preprocess, model, index, embeddings = load_pipeline()

# Load evaluation dataset
with open("evaluation/evaluation_dataset.json", "r") as f:
    queries = json.load(f)

precisions = []
recalls = []
hits = []

for item in queries:

    query = item["query"]
    relevant = item["relevant_docs"]

    results = search_eurovision(
        query,
        df,
        bm25,
        preprocess,
        top_n=5
    )

    retrieved = results["doc_id"].astype(int).tolist()

    precisions.append(
        precision_at_k(retrieved, relevant)
    )

    recalls.append(
        recall_at_k(retrieved, relevant)
    )

    hits.append(
        hit_rate_at_k(retrieved, relevant)
    )

print("\n===== EVALUATION RESULTS =====")
print(f"Precision@5: {sum(precisions)/len(precisions):.3f}")
print(f"Recall@5: {sum(recalls)/len(recalls):.3f}")
print(f"HitRate@5: {sum(hits)/len(hits):.3f}")