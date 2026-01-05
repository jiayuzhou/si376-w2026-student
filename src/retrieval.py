"""Information retrieval utilities for finding relevant evidence.

This module implements three different retrieval paradigms for finding evidence
sentences relevant to a claim:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
   - Classic keyword-based search
   - Fast and interpretable
   - Works well for exact word matches
   - Used in Week 4 baseline

2. **BM25 (Best Matching 25)**:
   - Improved keyword-based search (refinement of TF-IDF)
   - Better handles document length and term saturation
   - Industry standard for search engines
   - Used in Week 12 RAG comparison

3. **Embeddings (Dense Semantic Search)**:
   - Neural network-based similarity
   - Captures semantic meaning, not just keywords
   - Can match "car" to "vehicle" even without shared words
   - Slower but more powerful for complex queries
   - Used in Week 12 RAG comparison

For beginners: Retrieval is like using a search engine - given a query (claim),
find the most relevant documents (evidence sentences). Different methods have
different strengths and trade-offs.

Common Pattern:
All retrievers follow the same two-step pattern:
1. Build an index: Process all documents once (slow, done once)
2. Retrieve: Given a query, find top-k most relevant documents (fast, done many times)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Tuple, Optional

import numpy as np


# ============================================================
# TF-IDF Retrieval (Keyword-Based)
# ============================================================
@dataclass
class TfidfIndex:
    """Index for TF-IDF retrieval.

    TF-IDF (Term Frequency-Inverse Document Frequency) measures how important
    a word is to a document in a collection. It gives high scores to words that
    appear often in a specific document but rarely in others.

    For beginners: Think of it like this:
    - TF (Term Frequency): How often does the word appear in this document?
    - IDF (Inverse Document Frequency): Is this word rare or common across all documents?
    - TF-IDF = TF × IDF: Words that appear often in THIS document but rarely
      in OTHERS are most important.

    Example:
    - "the" appears in almost every document → low IDF → low TF-IDF
    - "Einstein" appears often in physics docs but rarely elsewhere → high TF-IDF

    Attributes
    ----------
    vectorizer : TfidfVectorizer
        Sklearn object that converts text to TF-IDF vectors
    matrix : scipy.sparse matrix
        TF-IDF vectors for all documents (rows=docs, cols=words)
        For beginners: Sparse means most values are 0 (saves memory)
    keys : List[str]
        Unique identifiers for each document (e.g., "PageName::sent_id")
    texts : List[str]
        The actual text of each document
    """
    vectorizer: Any  # sklearn.feature_extraction.text.TfidfVectorizer
    matrix: Any      # scipy.sparse matrix [n_docs, n_features]
    keys: List[str]  # Document IDs
    texts: List[str] # Document texts


def build_tfidf_index(
    keys: List[str],
    texts: List[str],
    max_features: int = 50000,
    ngram_range: Tuple[int, int] = (1, 2)
) -> TfidfIndex:
    """Build a TF-IDF index from a collection of text documents.

    This function processes all documents and creates a TF-IDF matrix that can
    be used for fast retrieval. Only needs to be run once per corpus.

    For beginners: This is the "indexing" step - like creating an index at the
    back of a textbook. Slow to build, but makes searching fast later.

    Parameters
    ----------
    keys : List[str]
        Unique identifier for each document (same length as texts)
    texts : List[str]
        Text content of each document
    max_features : int, default=50000
        Maximum number of unique words/n-grams to keep. Higher = more accurate
        but slower and uses more memory.
        For beginners: This limits the "vocabulary" - we keep only the 50000
        most common words/phrases to save memory.
    ngram_range : Tuple[int, int], default=(1, 2)
        Range of n-grams to extract. (1, 2) means unigrams and bigrams.
        For beginners: n-grams are word sequences:
        - 1-gram (unigram): single words ("Albert", "Einstein")
        - 2-gram (bigram): two-word phrases ("Albert Einstein")
        - (1, 2) captures both individual words and two-word phrases

    Returns
    -------
    TfidfIndex
        Index object ready for retrieval with tfidf_retrieve()
    """
    # Import sklearn's TF-IDF implementation
    # For beginners: sklearn (scikit-learn) is a popular machine learning library
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create and configure the TF-IDF vectorizer
    # For beginners: A vectorizer converts text into numbers (vectors) that computers can process
    vectorizer = TfidfVectorizer(
        max_features=max_features,      # Limit vocabulary size (memory efficiency)
        ngram_range=ngram_range,        # Capture both single words and phrases
        stop_words="english"            # Remove common words like "the", "is", "and"
    )

    # Fit the vectorizer on all texts and transform them to TF-IDF vectors
    # For beginners: fit_transform() does two things:
    # 1. Learn the vocabulary from texts (which words exist, their IDF scores)
    # 2. Convert each text to a TF-IDF vector
    # X is a matrix where each row is one document's TF-IDF vector
    X = vectorizer.fit_transform(texts)

    # Return the index containing vectorizer, matrix, and metadata
    return TfidfIndex(vectorizer=vectorizer, matrix=X, keys=keys, texts=texts)


def tfidf_retrieve(
    index: TfidfIndex,
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """Retrieve top-k most relevant documents for a query using TF-IDF.

    For beginners: Given a question/claim, find the most relevant evidence
    sentences by comparing TF-IDF vectors. Higher score = more similar.

    Parameters
    ----------
    index : TfidfIndex
        Pre-built TF-IDF index from build_tfidf_index()
    query : str
        Query text (e.g., a claim to verify)
    top_k : int, default=5
        Number of top results to return

    Returns
    -------
    List[Tuple[str, float, str]]
        List of (key, score, text) tuples, sorted by score (highest first)
        For example: [("Page1::0", 0.85, "Albert Einstein was a physicist."), ...]
    """
    # Convert query to TF-IDF vector using the same vectorizer
    # For beginners: We need to represent the query in the same "vocabulary space"
    # as the documents. [query] wraps it in a list because transform expects a list.
    q = index.vectorizer.transform([query])

    # Compute similarity scores between query and all documents
    # For beginners: @ is matrix multiplication
    # - index.matrix is [n_docs, n_features] (all documents)
    # - q.T is [n_features, 1] (query, transposed)
    # - Result is [n_docs, 1] (similarity score for each document)
    # .toarray() converts sparse matrix to regular numpy array
    # .reshape(-1) flattens to 1D array [n_docs]
    scores = (index.matrix @ q.T).toarray().reshape(-1)

    # Find indices of top-k highest scores
    # For beginners: np.argsort(scores) gives indices that would sort the array
    # - Negative sign (-scores) sorts in descending order (highest first)
    # - [:top_k] takes first k elements (top k scores)
    top_idx = np.argsort(-scores)[:top_k]

    # Return list of (key, score, text) tuples for top results
    # For beginners: This is a list comprehension that builds the result list
    return [(index.keys[i], float(scores[i]), index.texts[i]) for i in top_idx]


# ============================================================
# BM25 Retrieval (Improved Keyword-Based)
# ============================================================
@dataclass
class BM25Index:
    """Index for BM25 retrieval.

    BM25 (Best Matching 25) is an improved version of TF-IDF that addresses
    two key limitations:
    1. **Term saturation**: TF-IDF scores keep growing with term frequency, but
       seeing a word 100 times vs 10 times doesn't make it 10x more relevant.
       BM25 uses a saturation function to limit this growth.
    2. **Document length normalization**: Longer documents naturally have higher
       TF scores. BM25 normalizes for document length more effectively.

    For beginners: Think of BM25 as "TF-IDF 2.0" - it fixes known issues and
    generally gives better search results. It's the algorithm used by many
    search engines (Elasticsearch, Lucene).

    When to use BM25 vs TF-IDF:
    - BM25: Better accuracy, standard for production search systems
    - TF-IDF: Simpler, easier to understand and debug

    Attributes
    ----------
    bm25 : BM25Okapi
        BM25 model from rank-bm25 library
    keys : List[str]
        Unique identifiers for each document
    texts : List[str]
        The actual text of each document
    """
    bm25: Any        # rank_bm25.BM25Okapi
    keys: List[str]  # Document IDs
    texts: List[str] # Document texts


def build_bm25_index(keys: List[str], texts: List[str]) -> BM25Index:
    """Build a BM25 index from a collection of text documents.

    For beginners: Similar to build_tfidf_index(), but uses BM25 algorithm
    instead. BM25 generally gives better results than TF-IDF for search.

    Parameters
    ----------
    keys : List[str]
        Unique identifier for each document
    texts : List[str]
        Text content of each document

    Returns
    -------
    BM25Index
        Index object ready for retrieval with bm25_retrieve()
    """
    # Try to import BM25 library
    # For beginners: rank-bm25 is a third-party library implementing BM25
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:
        # If library not installed, give helpful error message
        raise ImportError("Missing rank-bm25. Install with: pip install rank-bm25") from e

    # Tokenize all texts (convert to lowercase and split into words)
    # For beginners: BM25 works on individual words (tokens), not full text
    # .lower() makes everything lowercase ("The" → "the")
    # .split() splits on whitespace ("hello world" → ["hello", "world"])
    # List comprehension does this for all texts: [result for t in texts]
    tokenized = [t.lower().split() for t in texts]

    # Build BM25 index from tokenized texts
    # For beginners: BM25Okapi learns statistics about word frequencies and
    # document lengths to enable fast retrieval later
    bm25 = BM25Okapi(tokenized)

    return BM25Index(bm25=bm25, keys=keys, texts=texts)


def bm25_retrieve(
    index: BM25Index,
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """Retrieve top-k most relevant documents for a query using BM25.

    For beginners: Same idea as tfidf_retrieve(), but uses BM25 scoring which
    generally gives better results for keyword-based search.

    Parameters
    ----------
    index : BM25Index
        Pre-built BM25 index from build_bm25_index()
    query : str
        Query text (e.g., a claim to verify)
    top_k : int, default=5
        Number of top results to return

    Returns
    -------
    List[Tuple[str, float, str]]
        List of (key, score, text) tuples, sorted by score (highest first)
    """
    # Tokenize query (same way we tokenized documents during indexing)
    # For beginners: We need to process the query the same way we processed documents
    q = query.lower().split()

    # Compute BM25 scores for all documents
    # For beginners: .get_scores() compares the query against all documents
    # and returns a score for each (higher = more relevant)
    scores = index.bm25.get_scores(q)

    # Find indices of top-k highest scores (same as TF-IDF)
    top_idx = np.argsort(-scores)[:top_k]

    # Return list of (key, score, text) tuples
    return [(index.keys[i], float(scores[i]), index.texts[i]) for i in top_idx]


# ============================================================
# Embedding Retrieval (Semantic/Neural Search)
# ============================================================
@dataclass
class EmbeddingIndex:
    """Index for embedding-based semantic retrieval.

    Embeddings are dense vector representations of text created by neural networks.
    Unlike TF-IDF/BM25 which only match keywords, embeddings capture semantic
    meaning - they know that "car" and "vehicle" are related even without
    shared words.

    For beginners: Think of embeddings as "meaning vectors". Similar meanings
    → similar vectors. We measure similarity using cosine similarity (the angle
    between vectors).

    Example:
    - "Albert Einstein was a physicist" → [0.12, -0.45, 0.78, ...] (768 numbers)
    - "Einstein studied physics" → [0.15, -0.42, 0.80, ...] (very similar vector!)
    - "I like pizza" → [-0.50, 0.20, -0.10, ...] (very different vector)

    When to use embeddings vs keyword search:
    - Embeddings: Better for semantic/conceptual matching, handles synonyms
    - TF-IDF/BM25: Faster, better for exact keyword matching, more interpretable

    Attributes
    ----------
    model : SentenceTransformer
        Neural network model that converts text to embeddings
    embeddings : np.ndarray
        Pre-computed embedding vectors for all documents [n_docs, embedding_dim]
        For beginners: Each row is one document's embedding (e.g., 768 numbers)
    keys : List[str]
        Unique identifiers for each document
    texts : List[str]
        The actual text of each document
    """
    model: Any              # sentence_transformers.SentenceTransformer
    embeddings: np.ndarray  # [n_docs, embedding_dim], usually float32
    keys: List[str]         # Document IDs
    texts: List[str]        # Document texts


def build_embedding_index(
    keys: List[str],
    texts: List[str],
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64
) -> EmbeddingIndex:
    """Build an embedding index from a collection of text documents.

    This function uses a neural network to convert all documents into embedding
    vectors. This is the slowest indexing method but enables powerful semantic search.

    For beginners: The model is a neural network trained on millions of sentences
    to learn what "similar meaning" looks like. We run each document through the
    model to get its embedding vector.

    Parameters
    ----------
    keys : List[str]
        Unique identifier for each document
    texts : List[str]
        Text content of each document
    model_name : str, default="sentence-transformers/all-MiniLM-L6-v2"
        Name of the sentence embedding model to use.
        For beginners: all-MiniLM-L6-v2 is a small, fast model (80MB) that gives
        good results. Larger models like all-mpnet-base-v2 are more accurate but
        slower. See: https://www.sbert.net/docs/pretrained_models.html
    batch_size : int, default=64
        Number of texts to process at once. Higher = faster but more memory.
        For beginners: Processing in batches is more efficient than one-by-one.

    Returns
    -------
    EmbeddingIndex
        Index object ready for retrieval with embedding_retrieve()
    """
    # Try to import sentence-transformers library
    # For beginners: sentence-transformers is a library for embedding models
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise ImportError(
            "Missing sentence-transformers. Install with: pip install sentence-transformers"
        ) from e

    # Load the pre-trained embedding model
    # For beginners: This downloads the model the first time (~80MB), then caches it
    model = SentenceTransformer(model_name)

    # Encode all texts to embeddings
    # For beginners: .encode() runs the neural network on each text
    # - batch_size: Process this many texts at once (efficiency)
    # - show_progress_bar: Display progress (useful for large corpuses)
    # - normalize_embeddings: Scale vectors to length 1 (enables cosine similarity via dot product)
    emb = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True  # Important: enables fast cosine similarity
    )

    # Convert to numpy array with float32 type (saves memory vs float64)
    # For beginners: float32 uses half the memory of float64, with negligible accuracy loss
    emb = np.asarray(emb, dtype=np.float32)

    return EmbeddingIndex(model=model, embeddings=emb, keys=keys, texts=texts)


def embedding_retrieve(
    index: EmbeddingIndex,
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """Retrieve top-k most relevant documents for a query using embeddings.

    For beginners: Convert the query to an embedding, then find documents with
    the most similar embeddings (using cosine similarity = dot product for
    normalized vectors).

    Parameters
    ----------
    index : EmbeddingIndex
        Pre-built embedding index from build_embedding_index()
    query : str
        Query text (e.g., a claim to verify)
    top_k : int, default=5
        Number of top results to return

    Returns
    -------
    List[Tuple[str, float, str]]
        List of (key, score, text) tuples, sorted by score (highest first)
        For embeddings, scores are cosine similarities between -1 and 1
        (in practice usually between 0 and 1 for normalized embeddings)
    """
    # Encode query to embedding using the same model
    # For beginners: [query] wraps in list because encode expects a list
    # normalize_embeddings=True ensures we can use dot product for similarity
    q = index.model.encode([query], normalize_embeddings=True)

    # Convert to numpy array and extract the single embedding
    # For beginners: q is shape [1, embedding_dim], we want [embedding_dim]
    # np.asarray converts to numpy array, dtype=float32 matches index embeddings
    # [0] extracts the first (and only) embedding
    q = np.asarray(q, dtype=np.float32)[0]

    # Compute cosine similarity scores via dot product
    # For beginners: @ is matrix multiplication (dot product)
    # - index.embeddings is [n_docs, embedding_dim]
    # - q is [embedding_dim]
    # - Result is [n_docs] - one similarity score per document
    # Why does dot product give cosine similarity? Because vectors are normalized!
    # For normalized vectors: cos(θ) = dot(a, b) / (||a|| × ||b||) = dot(a, b) / (1 × 1) = dot(a, b)
    scores = index.embeddings @ q

    # Find indices of top-k highest scores (same as TF-IDF and BM25)
    top_idx = np.argsort(-scores)[:top_k]

    # Return list of (key, score, text) tuples
    return [(index.keys[i], float(scores[i]), index.texts[i]) for i in top_idx]
