# Basic BM25 search 
from rank_bm25 import BM25Okapi 
import pandas as pd

import pandas as pd
from rank_bm25 import BM25Okapi


def search_eurovision(query, df, bm25, preprocess, top_n=10):
    processed = preprocess(query)
    scores = bm25.get_scores(processed)

    top_idx = pd.Series(scores).sort_values(ascending=False).index[:top_n]

    results = df.iloc[top_idx]["doc_id","Artist", "Song", "Year", "Country"].copy()
    results["score"] = [scores[i] for i in top_idx]

    return results


def search_eurovision_with_lyrics(query, df, bm25, preprocess, top_n=5):
    processed = preprocess(query)
    scores = bm25.get_scores(processed)

    top_idx = pd.Series(scores).sort_values(ascending=False).index[:top_n]

    results = df.iloc[top_idx]["doc_id", "Artist", "Song", "Year", "Lyrics translation"].copy()

    results["snippet"] = results["Lyrics translation"].fillna("").str[:150]
    results["score"] = [scores[i] for i in top_idx]

    return results.drop(columns=["Lyrics translation"])


def advanced_eurovision_search(query, df, bm25, preprocess,
                    year=None, country=None, language=None, top_n=5):

    filtered = df.copy()

    if year is not None:
        filtered = filtered[filtered["Year"] == year]

    if country:
        filtered = filtered[filtered["Country"].str.contains(country, case=False, na=False)]

    if language:
        filtered = filtered[filtered["Language"].str.contains(language, case=False, na=False)]

    if filtered.empty:
        return pd.DataFrame({"message": ["No results"]})

    subset_bm25 = BM25Okapi(filtered["tokenized_lyrics"].tolist())

    processed = preprocess(query)
    scores = subset_bm25.get_scores(processed)

    top_idx = pd.Series(scores).sort_values(ascending=False).index[:top_n]

    results = filtered.iloc[top_idx]["doc_id", "Artist", "Song", "Year", "Country", "Language"].copy()

    results["score"] = [scores[i] for i in top_idx]

    return results
