from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import re

def build_faiss_index(df):
    """
    Builds FAISS index from Eurovision dataset.
    Returns: model, index, embeddings
    """

    model = SentenceTransformer('all-MiniLM-L6-v2')

    sentences = df["combined_text"].tolist()

    embeddings = model.encode(
        sentences,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")

    # normalize for cosine similarity
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index, embeddings

def semantic_search(query, df, model, index, embeddings, top_n=5):
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    scores, idx = index.search(q, top_n)

    results = df.iloc[idx[0]][["doc_id","Artist", "Song", "Year", "Country"]].copy()
    results["score"] = scores[0]

    return results

# Final Smart Search

def final_smart_search(query, df, model, index, embeddings):

    # Extract year 
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', query)

    # Extract country 
    all_countries = df["Country"].dropna().unique()
    found_country = next(
        (c for c in all_countries if c.lower() in query.lower()),
        None
    )

    # Start with full dataset 
    temp_df = df.copy()

    # Apply filters 
    if year_match:
        temp_df = temp_df[temp_df["Year"] == int(year_match.group(1))]

    if found_country:
        temp_df = temp_df[
            temp_df["Country"].str.contains(found_country, case=False, na=False)
        ]

    
    # CASE 1: FILTERED RESULTS
    
    if len(temp_df) > 0:

        if len(temp_df) > 1:
            subset_idx = temp_df.index.values
            subset_embeddings = embeddings[subset_idx]

            q = model.encode([query], convert_to_numpy=True).astype("float32")
            faiss.normalize_L2(q)

            scores = np.dot(subset_embeddings, q[0])

            best_idx = subset_idx[np.argmax(scores)]

            return df.iloc[[best_idx]], "Filtered + Semantic"

        return temp_df.head(1), "Exact Filter Match"

    
    # CASE 2: FALLBACK
    
    q = model.encode([query], convert_to_numpy=True).astype("float32")
    faiss.normalize_L2(q)

    scores, idx = index.search(q, 1)

    return df.iloc[[idx[0][0]]], "FAISS Fallback"