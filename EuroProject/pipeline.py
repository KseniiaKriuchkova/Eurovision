import pandas as pd
from data.loader import load_eurovision_dataset 
from preprocessing.text_preprocessing import preprocess
from indexing.bm25 import build_bm25
from indexing.faiss_index import build_faiss_index
from data.sqlite_database import save_to_sqlite

def load_pipeline():
    #Load data
    df = load_eurovision_dataset()
    df["doc_id"] = range(len(df))

    print("Shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df.head())

    # Clean data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')

    # Save to SQLite
    save_to_sqlite(df)

    # Prepare text, combine it and create unified text field: Use English translation if available, otherwise fall back to original lyrics
    df["combined_text"] = df["Lyrics translation"].fillna(df["Lyrics"]).astype(str)

    print("Preprocessing lyrics... this will take a moment.")

    # Tokenization (artist + song + lyrics)
    df["tokenized_lyrics"] = df["combined_text"].apply(preprocess)

    print("Done! Lyrics are now preprocessed.")

    # Display sample metadata safely
    meta_cols = ["Artist", "Song", "Year", "Country"]
    existing_meta_cols = [c for c in meta_cols if c in df.columns]

    print("\nSample songs:")
    print(df[existing_meta_cols].head(10))

    # Identify rows where the 'Lyrics translation' is missing (NaN or empty string)
    # Missing translations analysis
    missing_translations = df[
        df["Lyrics translation"].isna() |
        (df["Lyrics translation"].astype(str).str.strip() == "")
    ]

    print(f"\nNumber of songs missing translations: {len(missing_translations)}")

    print("\nSample missing translations:")
    print(
        missing_translations[['Artist', 'Song', 'Country', 'Year']]
        .head(10)
        .to_string(index=False)
    )

    # Build BM25 -> list of token lists
    bm25 = build_bm25(df['tokenized_lyrics'].tolist())

    model, index, embeddings = build_faiss_index(df)

    return df, bm25, preprocess, model, index, embeddings