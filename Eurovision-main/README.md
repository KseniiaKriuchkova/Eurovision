# Eurovision
This project is a full Information Retrieval (IR) and Information Extraction (IE) system built as part of a university course.

It explores how to build a modern search engine pipeline using:

 - Traditional lexical retrieval (BM25 + inverted index)
 - Semantic search (SentenceTransformers + FAISS)
 - Hybrid ranking and filtering
 - Dataset preprocessing, indexing, and evaluation

The system is built on a Eurovision Song Lyrics dataset, allowing users to search songs based on meaning, keywords, emotions, and metadata (artist, country, year).

-------------
Project Goals
-------------

 - Build a complete IR pipeline from scratch
 - Compare lexical vs semantic retrieval methods
 - Implement scalable indexing structures
 - Experiment with hybrid search strategies
 - Create an evaluation set for retrieval quality
 - Gain hands-on experience with real-world IR systems

-----------
Dataset
-----------

Source: Kaggle Eurovision Song Lyrics dataset
Content: Song lyrics, artist names, countries, years, metadata
Type: Structured text dataset
Language: Mostly English + multilingual lyrics (translated when needed)
Preprocessing steps:
 - Lowercasing text
 - Removing punctuation and noise
 - Tokenization using NLTK
 - Stopword removal
 - English translation for missing lyrics (Google Translate API)
 - Combined text field creation:
 - Lyrics translation + Lyrics + metadata

--------------------
System Architecture
--------------------

The project includes multiple IR components:

1. Lexical Search (BM25)
Rank documents using term frequency + inverse document frequency
Implemented using rank_bm25
2. Inverted Index
Custom dictionary-based inverted index
Maps tokens → document IDs
Stored as serialized .pkl file
3. Semantic Search (FAISS)
Sentence embeddings using SentenceTransformers
Vector similarity search using FAISS
Cosine similarity ranking
4. Hybrid Search
Combines filtering (year, country, language)
Applies semantic fallback when filters fail
5. Storage Layer
SQLite database for structured metadata storage

-------------------
Key features
-------------------

 - BM25 keyword-based retrieval
 - Semantic search using embeddings
 - Hybrid search engine (filters + vector similarity)
 - Inverted index for fast token lookup
 - SQLite storage for metadata
 - Evaluation dataset with test queries
 - Interactive CLI chatbot search system
 - Translation of missing lyrics for better coverage

-----------------------
How to Run the Project
-----------------------

1. Clone the repository
git clone https://github.com/KseniiaKriuchkova/Eurovision.git
cd Eurovision
2. Install dependencies
pip install -r requirements.txt
3. Download dataset (automatic via KaggleHub)
The dataset is automatically downloaded using: kagglehub.dataset_download("minitree/eurovision-song-lyrics")
4. Run the system
python main.py

-------------------
Evaluation Approach
-------------------

An evaluation set was created with ~10–20 test queries.

Each query includes:
 - Expected relevant document IDs
 - Manual inspection of top-5 retrieved results
 - Measurement of retrieval success rate

Example format:

Query	            Expected Docs
songs about love	3, 8, 15, 22, 31
songs about war	    2, 9, 14, 18, 40