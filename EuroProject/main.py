import pandas as pd
import json
from preprocessing.nltk_setup import setup_nltk
from pipeline import load_pipeline
from preprocessing.text_preprocessing import preprocess
from search.search_engine import search_eurovision, search_eurovision_with_lyrics, advanced_eurovision_search
from indexing.inverted_index import build_inverted_index, save_index
from indexing.faiss_index import build_faiss_index, final_smart_search
import os

os.makedirs("evaluation", exist_ok=True)

# Better console display
pd.set_option("display.width", 200)

setup_nltk()

# Load everything
df, bm25, preprocess, model, index, embeddings = load_pipeline()
print("\nBuilding FAISS index (semantic search)...")

model, index, embeddings = build_faiss_index(df)

print("FAISS ready!")

print("Dataset loaded:", df.shape)

#Test
print("\nTesting basic search...")
print(
    search_eurovision(
        "love song from Greece",
        df,
        bm25,
        preprocess
    ).to_string(index=False)
)

# Test with lyrics 
results = search_eurovision_with_lyrics(
    "peace and love in the world",
    df,
    bm25,
    preprocess
)

print(results.to_string(index=False))

# Test queries
test_queries = [
    "songs about peace and war",
    "emotional ballads about heartbreak",
    "dancing and disco party",
    "lyrics mentioning fire and flames",
    "mental health and sadness",
    "moon and stars in the night sky",
    "traditional folk instruments and culture",
    "flying high like a bird",
    "romantic love songs",
    "songs about water and the sea"
]

# Create evaluation report
for i, q in enumerate(test_queries, 1):
    print(f"\nQUERY {i}: '{q}'")

    results = search_eurovision(
        q,
        df,
        bm25,
        preprocess,
        top_n=5
    )

    print(results.to_string(index=False))
    print("-" * 60)

# Advanced examples
print("\nAdvanced search examples:")
print("\n Example 1: Search for songs about 'love', specifically for Serbia (any year)")
print(
    advanced_eurovision_search(
        "love",
        df,
        bm25,
        preprocess,
        country="Serbia",
        top_n=5
    ).to_string(index=False)
)

print("\n Example 2: Search for songs about 'fire', specifically from the year 2020")
print(
    advanced_eurovision_search(
        "fire",
        df,
        bm25,
        preprocess,
        year="2020",
        top_n=5
    ).to_string(index=False)
)

print("\n Example 3: Search for songs about 'dance', specifically in the French language")
print(
    advanced_eurovision_search(
        "dance",
        df,
        bm25,
        preprocess,
        language="French",
        top_n=5
    ).to_string(index=False)
)

#Inverted index
inverted_index = build_inverted_index(df)
save_index(inverted_index)

print("\nInverted index saved!")

# Load evaluation dataset
results = df[df["combined_text"].str.contains("peace|war", case=False, na=False)][["doc_id", "Song", "Artist"]].head(1700)
print("\n===== EVALUATION DATASET =====\n")
print(results[["doc_id", "Artist", "Song"]].head(1700))

#Chatbot
print("\n--- 🤖 FINAL EUROVISION SMART ASSISTANT ---")
print("Type 'exit' to quit.")
print("-" * 60)

while True:
    user_input = input("\n👤 YOU: ")

    if user_input.lower() in ['exit', 'quit', 'stop']:
        print("🤖 SYSTEM: Goodbye!")
        break

    res_row, logic_used = final_smart_search(
        user_input,
        df,
        model,
        index,
        embeddings
    )

    if not res_row.empty:
        res = res_row.iloc[0]

        print(f"\n🤖 SYSTEM (Logic: {logic_used})")
        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        print(f"🏆 {res['Artist']} - '{res['Song']}'")
        print(f"📍 {res['Country']} ({res['Year']})")

        if "Language" in res:
            print(f"🌍 {res['Language']}")

        print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

        snippet = str(res.get("Lyrics translation", ""))[:250]
        print(f"📝 \"{snippet}...\"")

    else:
        print("\n🤖 SYSTEM: No match found.")

    print("-" * 60)