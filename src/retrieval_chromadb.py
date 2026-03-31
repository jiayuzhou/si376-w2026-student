"""ChromaDB-based retrieval for fact-checking.

This module provides an alternative retrieval implementation using ChromaDB,
an industry-standard vector database. It complements the custom implementation
in src/retrieval.py by showing how real-world systems handle vector search.

Why ChromaDB?
- **Industry standard**: Used in production RAG systems
- **Simple API**: `collection.add()`, `collection.query()`
- **Persistent storage**: No rebuilding indexes every time
- **Built-in embeddings**: Can use sentence-transformers directly

Why compare with custom implementation?
- **Understanding vs convenience**: Custom code shows how retrieval works
- **Trade-offs**: ChromaDB adds a dependency but simplifies code
- **Career relevance**: Students should know both approaches

For beginners: ChromaDB is like a specialized database for storing and
searching vectors. Instead of writing our own similarity search (like we
did with TF-IDF and embeddings in the main lab), ChromaDB handles it for us.

Used in Week 12 Part 10 for comparison with custom retrieval.
"""

from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

# Check if chromadb is available
try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False


# ============================================================
# ChromaDB Collection Wrapper
# ============================================================
class ChromaDBRetriever:
    """Vector retrieval using ChromaDB.

    For beginners: This class wraps ChromaDB to provide the same interface
    as our custom retrieval functions. You can swap between:
    - Custom: tfidf_retrieve(index, query, top_k)
    - ChromaDB: retriever.retrieve(query, top_k)

    The results are the same format: [(key, score, text), ...]

    Attributes
    ----------
    collection : chromadb.Collection
        The ChromaDB collection storing our documents
    client : chromadb.Client
        The ChromaDB client (in-memory or persistent)
    """

    def __init__(
        self,
        collection_name: str = "fever_evidence",
        persist_directory: Optional[str] = None,
        embedding_function: Optional[str] = "all-MiniLM-L6-v2"
    ):
        """Initialize ChromaDB retriever.

        For beginners: ChromaDB can run in two modes:
        - In-memory: Fast, but data lost when program ends
        - Persistent: Saves to disk, survives restarts

        Parameters
        ----------
        collection_name : str
            Name for the ChromaDB collection
        persist_directory : str, optional
            If provided, data is saved to this directory.
            If None, uses in-memory storage.
        embedding_function : str, optional
            Sentence-transformer model for embeddings.
            Default is 'all-MiniLM-L6-v2' (same as Week 12).
        """
        if not CHROMADB_AVAILABLE:
            raise ImportError(
                "ChromaDB is not installed. Install with:\n"
                "  pip install chromadb"
            )

        # Create client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        # Set up embedding function
        self._embedding_fn = None
        if embedding_function:
            try:
                from chromadb.utils import embedding_functions
                self._embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=embedding_function
                )
            except Exception:
                # Fall back to default
                pass

        # Get or create collection
        if self._embedding_fn:
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                embedding_function=self._embedding_fn
            )
        else:
            self.collection = self.client.get_or_create_collection(
                name=collection_name
            )

        self.collection_name = collection_name

    def add_documents(
        self,
        keys: List[str],
        texts: List[str],
        batch_size: int = 500,
        show_progress: bool = True
    ):
        """Add documents to the ChromaDB collection.

        For beginners: This is like building an index - we're storing
        all our Wikipedia sentences so we can search them later.

        The key difference from our custom approach:
        - Custom: We compute TF-IDF vectors or embeddings ourselves
        - ChromaDB: It computes and stores embeddings for us!

        Parameters
        ----------
        keys : List[str]
            Unique identifiers for each document (e.g., "Einstein::5")
        texts : List[str]
            The document texts (Wikipedia sentences)
        batch_size : int
            How many documents to add at once (for memory efficiency)
        show_progress : bool
            Whether to print progress updates
        """
        if len(keys) != len(texts):
            raise ValueError("keys and texts must have same length")

        total = len(keys)

        if show_progress:
            print(f"Adding {total} documents to ChromaDB collection...")

        # Add in batches
        for i in range(0, total, batch_size):
            batch_keys = keys[i:i + batch_size]
            batch_texts = texts[i:i + batch_size]

            # ChromaDB's add() method
            self.collection.add(
                ids=batch_keys,
                documents=batch_texts
            )

            if show_progress and (i + batch_size) % 5000 == 0:
                print(f"  Added {min(i + batch_size, total)}/{total} documents...")

        if show_progress:
            print(f"  Done! Collection has {self.collection.count()} documents.")

    def retrieve(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Tuple[str, float, str]]:
        """Retrieve most similar documents to query.

        For beginners: This is the same output format as our custom
        retrieval functions:
        - tfidf_retrieve(index, query, top_k) → [(key, score, text), ...]
        - retriever.retrieve(query, top_k) → [(key, score, text), ...]

        So you can swap between them easily!

        Parameters
        ----------
        query : str
            The search query (e.g., a claim to fact-check)
        top_k : int
            Number of results to return

        Returns
        -------
        List[Tuple[str, float, str]]
            List of (document_id, similarity_score, document_text) tuples,
            sorted by similarity (highest first).
        """
        # Query ChromaDB
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            include=["documents", "distances"]
        )

        # Format results to match our custom interface
        # ChromaDB returns: {'ids': [[...]], 'documents': [[...]], 'distances': [[...]]}
        output = []
        if results and results['ids'] and results['ids'][0]:
            ids = results['ids'][0]
            docs = results['documents'][0] if results['documents'] else [''] * len(ids)
            # ChromaDB returns L2 distance; convert to similarity score
            # For cosine similarity: score = 1 - (distance / 2)
            distances = results['distances'][0] if results['distances'] else [0] * len(ids)

            for key, doc, dist in zip(ids, docs, distances):
                # Convert distance to similarity (higher = more similar)
                similarity = 1.0 / (1.0 + dist)  # Simple conversion
                output.append((key, similarity, doc))

        return output

    def count(self) -> int:
        """Return number of documents in collection."""
        return self.collection.count()

    def clear(self):
        """Clear all documents from collection."""
        self.client.delete_collection(self.collection_name)
        if self._embedding_fn:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                embedding_function=self._embedding_fn
            )
        else:
            self.collection = self.client.create_collection(
                name=self.collection_name
            )


# ============================================================
# Convenience Functions (match custom interface)
# ============================================================
def build_chromadb_index(
    keys: List[str],
    texts: List[str],
    collection_name: str = "fever_evidence",
    embedding_model: str = "all-MiniLM-L6-v2"
) -> ChromaDBRetriever:
    """Build a ChromaDB retrieval index.

    For beginners: This function matches the interface of our custom
    build_tfidf_index() and build_embedding_index() functions.

    Custom approach:
        index = build_tfidf_index(keys, texts)
        results = tfidf_retrieve(index, query, top_k)

    ChromaDB approach:
        index = build_chromadb_index(keys, texts)
        results = chromadb_retrieve(index, query, top_k)

    Parameters
    ----------
    keys : List[str]
        Document identifiers
    texts : List[str]
        Document texts
    collection_name : str
        Name for the collection
    embedding_model : str
        Sentence-transformer model to use

    Returns
    -------
    ChromaDBRetriever
        Retriever object that can be queried
    """
    if not CHROMADB_AVAILABLE:
        raise ImportError(
            "ChromaDB is not installed. Install with:\n"
            "  pip install chromadb"
        )

    retriever = ChromaDBRetriever(
        collection_name=collection_name,
        embedding_function=embedding_model
    )
    retriever.add_documents(keys, texts)
    return retriever


def chromadb_retrieve(
    retriever: ChromaDBRetriever,
    query: str,
    top_k: int = 5
) -> List[Tuple[str, float, str]]:
    """Retrieve documents using ChromaDB.

    For beginners: This matches the interface of tfidf_retrieve()
    and embedding_retrieve() from src/retrieval.py.

    Parameters
    ----------
    retriever : ChromaDBRetriever
        The retriever object from build_chromadb_index()
    query : str
        Search query
    top_k : int
        Number of results

    Returns
    -------
    List[Tuple[str, float, str]]
        List of (key, score, text) tuples
    """
    return retriever.retrieve(query, top_k)


# ============================================================
# Comparison Helper
# ============================================================
def compare_retrieval_methods(
    query: str,
    custom_results: List[Tuple[str, float, str]],
    chromadb_results: List[Tuple[str, float, str]]
) -> str:
    """Compare results from custom and ChromaDB retrieval.

    For beginners: This helps you see how the two approaches
    give similar (but not always identical) results.

    Parameters
    ----------
    query : str
        The search query
    custom_results : List[Tuple[str, float, str]]
        Results from custom retrieval (TF-IDF or embeddings)
    chromadb_results : List[Tuple[str, float, str]]
        Results from ChromaDB retrieval

    Returns
    -------
    str
        Formatted comparison string
    """
    lines = [
        "=" * 60,
        "RETRIEVAL COMPARISON",
        "=" * 60,
        f"Query: {query}",
        "",
        "Custom Results:",
    ]

    for i, (key, score, text) in enumerate(custom_results[:3], 1):
        text_short = text[:60] + "..." if len(text) > 60 else text
        lines.append(f"  {i}. [{score:.3f}] {text_short}")

    lines.extend(["", "ChromaDB Results:"])

    for i, (key, score, text) in enumerate(chromadb_results[:3], 1):
        text_short = text[:60] + "..." if len(text) > 60 else text
        lines.append(f"  {i}. [{score:.3f}] {text_short}")

    # Check overlap
    custom_keys = {k for k, _, _ in custom_results[:5]}
    chroma_keys = {k for k, _, _ in chromadb_results[:5]}
    overlap = len(custom_keys & chroma_keys)

    lines.extend([
        "",
        f"Overlap in top-5: {overlap}/5 documents",
        "",
        "KEY DIFFERENCES:",
        "  - Custom: You write the similarity search code",
        "  - ChromaDB: The database handles it for you",
        "  - Both use the same embedding model!",
        "=" * 60
    ])

    return "\n".join(lines)
