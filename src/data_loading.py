"""Data loading helpers for the FEVER-style course project.

This module provides functions to load and process the FEVER dataset, which contains
claims (statements to verify) and gold evidence sentences (Wikipedia sentences that
prove or disprove each claim).

Primary dataset used in labs:
- copenlu/fever_gold_evidence (claims + gold evidence sentences)

Key features:
- Smart caching: First load downloads from HuggingFace (~30 seconds), subsequent
  loads use cached data (~1 second). Cache keys are based on parameters, so
  changing split/sample_size/seed will load fresh data.
- Evidence corpus building: Extract all unique evidence sentences for retrieval.
- Normalized columns: Consistent column names across all weekly labs.

For beginners: This module handles all the "data wrangling" - downloading,
parsing, and organizing the FEVER dataset into a format that's easy to work with
in weekly labs. You'll use load_fever_gold() in almost every week.
"""

from __future__ import annotations

import hashlib  # For generating cache keys (unique IDs for cached data)
import json     # For converting parameters to strings
from dataclasses import dataclass  # For creating simple data classes
from pathlib import Path  # For handling file paths
from typing import List, Optional  # For type hints

import numpy as np  # For numerical operations and handling numpy arrays
import pandas as pd  # For working with tabular data (DataFrames)

from .config import LABELS, DEFAULT_DATASET, DEFAULT_DATASET_CONFIG, PATHS
from .utils import set_seed


# ============================================================
# Data Classes
# ============================================================
@dataclass
class EvidenceSentence:
    """A single evidence sentence from Wikipedia.

    Each evidence sentence has three components:
    - page: The Wikipedia page title (e.g., "Albert_Einstein")
    - sent_id: The sentence number within that page (e.g., 0 for first sentence)
    - text: The actual text of the sentence

    For beginners: A dataclass is like a simple container for data. It's more
    organized than using a dictionary and has better code completion in editors.

    Example
    -------
    >>> ev = EvidenceSentence(
    ...     page="Barack_Obama",
    ...     sent_id=0,
    ...     text="Barack Obama was the 44th president of the United States."
    ... )
    >>> print(ev.page)
    'Barack_Obama'
    """
    page: str       # Wikipedia page title
    sent_id: int    # Sentence ID within the page
    text: str       # The actual sentence text


# ============================================================
# Cache Helper Functions
# ============================================================
def _get_cache_path(cache_key: str) -> Path:
    """Generate cache file path from a cache key.

    For beginners: The cache directory stores processed data files so we don't
    have to re-download and re-process data every time. This function just
    creates the file path where cached data will be saved/loaded.

    Parameters
    ----------
    cache_key : str
        A unique identifier for the cached data (includes parameters)

    Returns
    -------
    Path
        Full path to the cache file (e.g., .cache/fever_gold_a1b2c3.pkl)
    """
    # .pkl is a "pickle" file - Python's way of saving objects to disk
    return PATHS.cache_dir / f"{cache_key}.pkl"


def _make_cache_key(prefix: str, **kwargs) -> str:
    """Create a deterministic cache key from parameters.

    The cache key is a unique ID based on the function parameters. If you call
    load_fever_gold with the same parameters twice, you get the same cache key
    and thus the same cached data.

    For beginners: This uses a "hash function" (MD5) to convert parameters into
    a short unique string. Think of it like a fingerprint - different parameters
    = different fingerprint. Same parameters = same fingerprint.

    Parameters
    ----------
    prefix : str
        A name describing what's cached (e.g., "fever_gold")
    **kwargs : dict
        Arbitrary keyword arguments (e.g., split="train", sample_size=5000)

    Returns
    -------
    str
        A unique cache key like "fever_gold_a1b2c3d4e5f6"
    """
    # Convert all parameters to a JSON string (deterministic order with sort_keys=True)
    # For beginners: JSON is a text format. We convert parameters to text so we can hash them.
    params_str = json.dumps(kwargs, sort_keys=True)

    # Hash the parameters using MD5 to get a unique identifier
    # .encode() converts string to bytes (required by hashlib)
    # .hexdigest() converts hash to a hex string (letters and numbers)
    # [:12] takes first 12 characters (enough to be unique, shorter than full hash)
    hash_str = hashlib.md5(params_str.encode()).hexdigest()[:12]

    # Combine prefix with hash: e.g., "fever_gold_a1b2c3d4e5f6"
    return f"{prefix}_{hash_str}"


# ============================================================
# Evidence Parsing
# ============================================================
def _parse_evidence_list(evidence_field) -> List[EvidenceSentence]:
    """Parse a single row's `evidence` field from the FEVER dataset.

    The FEVER dataset stores evidence as nested lists. This function converts
    that raw format into a clean list of EvidenceSentence objects.

    Expected format (per HuggingFace dataset viewer):
        [ [page_title, sentence_id, sentence_text], ... ]

    For beginners: Raw data often comes in messy formats. This function cleans
    it up and handles edge cases (missing data, wrong formats, etc.).

    Parameters
    ----------
    evidence_field : list or None
        Raw evidence data from the dataset (nested list)

    Returns
    -------
    List[EvidenceSentence]
        Clean list of evidence sentences
    """
    # Start with an empty list
    # For beginners: We'll add evidence sentences to this list as we parse them
    out: List[EvidenceSentence] = []

    # Handle None case (some claims have no evidence)
    if evidence_field is None:
        return out

    # Loop through each evidence item
    # For beginners: evidence_field looks like [[page1, id1, text1], [page2, id2, text2], ...]
    for item in evidence_field:
        # Safety check: each item should be a list/tuple/array with at least 3 elements
        # For beginners: Data can be messy, so we check before using it
        # Note: HuggingFace datasets sometimes return numpy arrays instead of lists
        if not isinstance(item, (list, tuple, np.ndarray)) or len(item) < 3:
            continue  # Skip invalid items

        # Extract the three components
        # item[0] = page title, item[1] = sentence ID, item[2] = sentence text
        page, sent_id, sent_text = item[0], item[1], item[2]

        # Try to convert sent_id to integer (sometimes it's a string)
        try:
            sent_id_int = int(sent_id)
        except Exception:
            # If conversion fails, use -1 as default
            sent_id_int = -1

        # Create an EvidenceSentence object and add to our list
        # For beginners: str() ensures everything is a string (handles weird types)
        out.append(EvidenceSentence(page=str(page), sent_id=sent_id_int, text=str(sent_text)))

    return out


# ============================================================
# Main Data Loading Function
# ============================================================
def load_fever_gold(
    split: str = "train",
    sample_size: Optional[int] = 5000,
    seed: int = 42,
    dataset_name: str = DEFAULT_DATASET,
    dataset_config: str = DEFAULT_DATASET_CONFIG,
    verbose: bool = True,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Load FEVER-style claims + evidence text as a pandas DataFrame.

    This is the main function you'll use to load data in weekly labs. It handles
    downloading from HuggingFace, caching for speed, and normalizing columns.

    For beginners: A DataFrame is like a spreadsheet in Python - rows and columns
    of data. This function returns a DataFrame where each row is one claim with
    its evidence and label.

    Parameters
    ----------
    split : str, default="train"
        Which split to load. Options:
        - "train": Training data (~145k examples)
        - "validation": Validation data (~19k examples)
        - "test": Test data (~19k examples, no labels in original FEVER)
    sample_size : int or None, default=5000
        How many examples to load. None = load all examples.
        For beginners: We use small samples (5000) for classroom use to keep
        loading fast and not overwhelm laptops.
    seed : int, default=42
        Random seed for reproducibility (ensures same sample each time)
    dataset_name : str
        HuggingFace dataset identifier (default from config.py)
    dataset_config : str
        Dataset configuration/subset name (default from config.py)
    verbose : bool, default=True
        Whether to print loading progress
    use_cache : bool, default=True
        Whether to use cached data. First load caches data, subsequent loads
        are much faster. Set to False to force re-download.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - claim: The text claim to verify
        - label: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
        - evidence_objs: List of EvidenceSentence objects
        - evidence_sentences: List of evidence text strings
        - evidence_text: All evidence concatenated into one string
        - gold_sentence_keys: List of "{page}::{sent_id}" identifiers
        - gold_pages: List of Wikipedia page titles (deduplicated)
        Plus other original columns from the dataset.
    """
    # ====== Step 1: Check cache first ======
    # For beginners: Caching means saving processed data so we don't have to
    # download and process it again. First time = slow, later times = fast!
    if use_cache:
        # Create a unique cache key based on all parameters
        cache_key = _make_cache_key(
            "fever_gold",
            split=split,
            sample_size=sample_size,
            seed=seed,
            dataset_name=dataset_name,
            dataset_config=dataset_config,
        )
        cache_path = _get_cache_path(cache_key)

        # Check if cache file exists
        # For beginners: .exists() checks if a file is already on disk
        if cache_path.exists():
            try:
                if verbose:
                    # .name gives just the filename, not the full path
                    print(f"Loading from cache: {cache_path.name}")

                # Load the cached DataFrame from disk
                # For beginners: pd.read_pickle() loads a saved DataFrame
                df = pd.read_pickle(cache_path)

                if verbose:
                    # len(df) gives the number of rows
                    print(f"✓ Loaded {len(df)} examples from cache")

                # Return cached data immediately (fast!)
                return df
            except Exception as e:
                # If cache loading fails (corrupted file, etc.), print error and continue
                if verbose:
                    print(f"Cache load failed: {e}, re-loading from source...")

    # ====== Step 2: Load from HuggingFace ======
    # If we get here, either cache is disabled or doesn't exist
    # Set random seed for reproducibility
    set_seed(seed)

    if verbose:
        print(f"Loading FEVER dataset (split={split}, sample_size={sample_size})...")

    # Try to import the datasets library (from HuggingFace)
    # For beginners: try/except lets us handle errors gracefully
    try:
        from datasets import load_dataset
    except ImportError as e:
        # If library isn't installed, give a helpful error message
        raise ImportError(
            "Missing dependency `datasets`. Install with: pip install datasets"
        ) from e

    # Download and load the dataset from HuggingFace
    # For beginners: This is downloading data from the internet, may take 30 seconds first time
    ds = load_dataset(dataset_name, dataset_config, split=split)

    # If sample_size is specified, randomly sample that many examples
    if sample_size is not None:
        # .shuffle() randomizes the order (using seed for reproducibility)
        # .select() picks the first sample_size examples
        # min() ensures we don't try to select more than exist
        ds = ds.shuffle(seed=seed).select(range(min(sample_size, len(ds))))

    # Convert HuggingFace Dataset to pandas DataFrame
    # For beginners: DataFrames are easier to work with than HF Datasets
    df = ds.to_pandas()

    # ====== Step 3: Validate required columns ======
    # Make sure the dataset has the columns we expect
    if "label" not in df.columns or "claim" not in df.columns:
        raise ValueError(f"Unexpected columns: {df.columns.tolist()}")

    # ====== Step 4: Parse evidence into structured format ======
    # For beginners: Raw evidence is messy nested lists. We convert it to clean objects.
    if "evidence" in df.columns:
        # Apply our parsing function to every row
        # For beginners: .apply() runs a function on each element of a column
        # It's like a for loop but more efficient
        evidence_objs = df["evidence"].apply(_parse_evidence_list)
    else:
        # If no evidence column, create empty lists for each row
        # For beginners: This is a list comprehension - creates a list of empty lists
        evidence_objs = [[] for _ in range(len(df))]

    # Add new columns derived from evidence
    df["evidence_objs"] = evidence_objs

    # Extract just the text from each EvidenceSentence object
    # For beginners: lambda is an "anonymous function" - a function without a name
    # lambda xs: [x.text for x in xs] means: for a list xs, extract .text from each item x
    df["evidence_sentences"] = df["evidence_objs"].apply(lambda xs: [x.text for x in xs])

    # Join all evidence sentences into one big string
    # For beginners: " ".join(xs) combines a list of strings with spaces between them
    df["evidence_text"] = df["evidence_sentences"].apply(lambda xs: " ".join(xs))

    # Create unique keys for each evidence sentence (format: "PageName::sentence_id")
    # For beginners: These keys let us identify specific sentences for retrieval
    df["gold_sentence_keys"] = df["evidence_objs"].apply(lambda xs: [f"{x.page}::{x.sent_id}" for x in xs])

    # Extract unique Wikipedia pages (remove duplicates with set, then sort)
    # For beginners: {x.page for x in xs} is a "set comprehension" - like list comprehension but unique values
    df["gold_pages"] = df["evidence_objs"].apply(lambda xs: sorted(list({x.page for x in xs})))

    # ====== Step 5: Clean label column ======
    # Ensure label column is string type (sometimes it's stored as integer code)
    # For beginners: .astype(str) converts all values to strings
    df["label"] = df["label"].astype(str)

    # Normalize label format: "NOT_ENOUGH_INFO" -> "NOT ENOUGH INFO"
    # For beginners: .replace() swaps values (like find-and-replace in Word)
    df["label"] = df["label"].replace({"NOT_ENOUGH_INFO": "NOT ENOUGH INFO"})

    # Filter to only valid labels and reset row numbers
    # For beginners:
    # - .isin(LABELS) checks if each label is in the valid list
    # - df[condition] keeps only rows where condition is True
    # - .reset_index(drop=True) renumbers rows from 0, 1, 2, ... (drop=True discards old index)
    df = df[df["label"].isin(LABELS)].reset_index(drop=True)

    if verbose:
        # Print summary: how many rows and columns
        print(f"✓ Loaded {len(df)} examples with {len(df.columns)} features")

    # ====== Step 6: Save to cache for next time ======
    if use_cache:
        try:
            # Create cache directory if it doesn't exist
            # parents=True creates parent directories too if needed
            PATHS.cache_dir.mkdir(exist_ok=True, parents=True)

            # Save DataFrame to pickle file
            # For beginners: Pickle is Python's way of saving objects to files
            df.to_pickle(cache_path)

            if verbose:
                print(f"✓ Saved to cache: {cache_path.name}")
        except Exception as e:
            # If caching fails, print warning but continue (not critical)
            if verbose:
                print(f"Warning: Failed to save cache: {e}")

    return df


# ============================================================
# Evidence Corpus Building
# ============================================================
def build_evidence_corpus(
    df: pd.DataFrame,
    max_sentences: int = 50000,
    seed: int = 42,
    show_progress: bool = True,
    use_cache: bool = True,
    cache_key_prefix: str = "evidence_corpus",
) -> pd.DataFrame:
    """Build a deduplicated evidence sentence corpus from a DF with `evidence_objs`.

    This function extracts all unique evidence sentences from the dataset and returns
    them as a DataFrame. This is useful for building retrieval indexes (Week 4, 12).

    For beginners: Think of this as creating a "database" of all evidence sentences.
    Instead of having evidence scattered across claims, we collect all unique sentences
    in one place. This is what a retrieval system searches through.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with `evidence_objs` column (from load_fever_gold)
    max_sentences : int, default=50000
        Maximum number of sentences to include (samples if more exist)
        For beginners: Keeping this small speeds up retrieval and fits in memory
    seed : int, default=42
        Random seed for reproducibility when sampling
    show_progress : bool, default=True
        Whether to show progress bar during extraction
    use_cache : bool, default=True
        Whether to use cached corpus (much faster for repeated calls)
    cache_key_prefix : str, default="evidence_corpus"
        Prefix for cache key (allows multiple cached corpuses)

    Returns
    -------
    pd.DataFrame
        Corpus DataFrame with columns:
        - key: Unique identifier "{page}::{sent_id}"
        - page: Wikipedia page title
        - sent_id: Sentence ID within the page
        - text: The evidence sentence text
    """
    # ====== Step 1: Check cache first ======
    if use_cache:
        # Create cache key based on parameters
        # For beginners: Different max_sentences or seed = different cache file
        cache_key = _make_cache_key(
            cache_key_prefix,
            max_sentences=max_sentences,
            seed=seed,
            n_rows=len(df),  # Number of input rows affects output
        )
        cache_path = _get_cache_path(cache_key)

        if cache_path.exists():
            try:
                if show_progress:
                    print(f"Loading evidence corpus from cache: {cache_path.name}")
                corpus = pd.read_pickle(cache_path)
                if show_progress:
                    print(f"✓ Loaded {len(corpus)} sentences from cache")
                return corpus
            except Exception as e:
                if show_progress:
                    print(f"Cache load failed: {e}, rebuilding...")

    # ====== Step 2: Extract evidence sentences ======
    set_seed(seed)
    rows = []  # Will collect dictionaries for each sentence

    # Create an iterator over evidence objects
    iterator = df["evidence_objs"]

    # Try to show progress bar if tqdm is available
    # For beginners: tqdm is a library that shows progress bars. Optional but nice!
    if show_progress:
        try:
            from tqdm import tqdm
            # Wrap iterator with tqdm to show progress
            iterator = tqdm(iterator, desc="Building evidence corpus", unit=" claims")
        except ImportError:
            # If tqdm not installed, just use regular iterator (no progress bar)
            pass

    # Loop through all claims and extract evidence sentences
    # For beginners: This is a nested loop - for each claim, for each evidence sentence
    for ev_list in iterator:
        for ev in ev_list:
            # Create a dictionary for this sentence
            # For beginners: We'll convert this list of dictionaries to a DataFrame
            rows.append({
                "key": f"{ev.page}::{ev.sent_id}",  # Unique identifier
                "page": ev.page,
                "sent_id": ev.sent_id,
                "text": ev.text
            })

    # ====== Step 3: Create DataFrame and remove duplicates ======
    # Convert list of dictionaries to DataFrame
    # For beginners: pd.DataFrame(rows) creates a table from a list of dictionaries
    corpus = pd.DataFrame(rows)

    # Remove duplicate sentences (same key) and reset row numbers
    # For beginners: .drop_duplicates("key") keeps only first occurrence of each key
    corpus = corpus.drop_duplicates("key").reset_index(drop=True)

    # ====== Step 4: Sample if too many sentences ======
    # If corpus is larger than max_sentences, randomly sample
    if len(corpus) > max_sentences:
        if show_progress:
            print(f"Sampling {max_sentences} from {len(corpus)} unique sentences...")

        # Randomly sample max_sentences rows
        # For beginners:
        # - .sample(n=...) picks n random rows
        # - random_state=seed ensures reproducibility
        # - .reset_index(drop=True) renumbers rows
        corpus = corpus.sample(n=max_sentences, random_state=seed).reset_index(drop=True)

    # ====== Step 5: Save to cache ======
    if use_cache:
        try:
            PATHS.cache_dir.mkdir(exist_ok=True, parents=True)
            corpus.to_pickle(cache_path)
            if show_progress:
                print(f"✓ Saved to cache: {cache_path.name}")
        except Exception as e:
            if show_progress:
                print(f"Warning: Failed to save cache: {e}")

    return corpus
