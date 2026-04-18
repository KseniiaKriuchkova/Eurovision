# Download NLTK resources for text processing 
import nltk

def setup_nltk():
    resources = {
        "punkt_tab": "tokenizers/punkt_tab",
        "stopwords": "corpora/stopwords"
    }

    for resource_name, resource_path in resources.items():
        try:
            nltk.data.find(resource_path)
        except LookupError:
            nltk.download(resource_name)

