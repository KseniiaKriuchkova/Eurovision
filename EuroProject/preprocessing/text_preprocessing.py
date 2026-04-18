import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Create stopwords
STOP_WORDS = set(stopwords.words("english")) # remove standard English stopwords

# Preprocessing
def preprocess(text):
    if not isinstance(text, str):
        return []

    text = text.lower()
    text = re.sub(r"\[.*?\]", "", text)          # remove metadata like [chorus]
    text = re.sub(r"[^a-z\s]", " ", text)       # keep only letters and remove non-alphabetical characters
    text = re.sub(r"\s+", " ", text).strip()    # normalize spaces

    tokens = word_tokenize(text) # tokenize text into individual words

    return [token for token in tokens if token not in STOP_WORDS and len(token) > 1]

