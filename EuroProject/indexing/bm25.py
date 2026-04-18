from rank_bm25 import BM25Okapi 

def build_bm25(tokenized_corpus): 
    return BM25Okapi(tokenized_corpus) 