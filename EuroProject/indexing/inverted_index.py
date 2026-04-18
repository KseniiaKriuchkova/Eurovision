from collections import defaultdict
import pickle
import os

def build_inverted_index(df):
    index = defaultdict(set)

    for i, tokens in zip (df["doc_id"], df["tokenized_lyrics"]):
        if not isinstance(tokens, list):
            continue
        for token in tokens:
            index[token].add(i)

    return dict(index)


def save_index(index, path="storage/inverted_index.pkl"):
    folder = os.path.dirname(path)
    os.makedirs(folder, exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(index, f)


def load_index(path="storage/inverted_index.pkl"):
    with open(path, "rb") as f:
        return pickle.load(f)