import pandas as pd
from deep_translator import GoogleTranslator
from data.loader import load_eurovision_dataset

def translate_missing(df):
    # Ensure column exists
    if "Lyrics translation" not in df.columns:
        df["Lyrics translation"] = None

    missing = df[df["Lyrics translation"].isna() | (df["Lyrics translation"].astype(str).str.strip() == "")]
    ids_to_translate = missing.index.tolist()

    translator = GoogleTranslator(source="auto", target="en")

    for idx in ids_to_translate:
        original_text = df.loc[idx, "Lyrics"]

        # Skip empty lyrics
        if pd.isna(original_text):
            continue

        print(f"Translating song ID: {idx}...")

        try:
            df.at[idx, "Lyrics translation"] = translator.translate(str(original_text)[:2000])
        except Exception as e:
            print(f"Error at {idx}: {e}")

    print("Translation complete!")
    # return ids so we can use them outside
    return df, ids_to_translate

if __name__ == "__main__":
    df = load_eurovision_dataset()
    df, ids = translate_missing(df)

    #replace display() with print
    print("\nTranslated samples:")
    print(df.loc[ids, ["Artist", "Song", "Lyrics translation"]].head(10))

