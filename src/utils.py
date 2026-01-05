"""Basic utility functions used throughout the FEVER course project.

This module provides commonly-used helper functions like setting random seeds
for reproducibility, computing softmax for probability distributions, and
checking if code is running in Google Colab.

For beginners: These are "helper functions" - small utilities that make your
code cleaner and avoid repeating the same operations everywhere.
"""

from __future__ import annotations

import os
import random
from typing import Any

import numpy as np


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch.

    Reproducibility means getting the same results every time you run your code.
    This is important for debugging and comparing experiments. By setting a "seed"
    (a starting number for randomness), you ensure that random operations like
    shuffling data or initializing model weights produce identical results.

    For beginners: Think of a seed like a recipe that tells the computer exactly
    how to be "random". The same seed = the same "random" numbers every time.

    Parameters
    ----------
    seed : int
        The random seed value (commonly 42 in examples, but any integer works)

    Example
    -------
    >>> set_seed(42)  # Now all random operations will be reproducible
    >>> x = random.randint(1, 100)
    >>> set_seed(42)  # Reset to same seed
    >>> y = random.randint(1, 100)
    >>> x == y  # True - same seed produces same "random" number
    True
    """
    # Set seed for Python's built-in random module (used by random.choice, random.shuffle, etc.)
    random.seed(seed)

    # Set seed for NumPy's random operations (used by np.random.rand, np.random.shuffle, etc.)
    np.random.seed(seed)

    # Try to set seed for PyTorch (used for neural network weight initialization)
    # We use try/except because PyTorch might not be installed in early weeks
    try:
        import torch
        # Set seed for CPU operations
        torch.manual_seed(seed)
        # Set seed for GPU operations (if GPU is available)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        # If PyTorch isn't installed, just skip this part (no problem)
        pass


def ensure_dir(path: str) -> None:
    """Create a directory if it doesn't already exist.

    For beginners: This is like creating a folder on your computer. If the folder
    already exists, nothing happens (no error). This is useful before saving files
    to make sure the destination folder exists.

    Parameters
    ----------
    path : str
        The directory path to create (can be relative like "outputs" or
        absolute like "/home/user/outputs")

    Example
    -------
    >>> ensure_dir("outputs/week01")  # Creates outputs/week01/ if needed
    >>> # Now you can safely save files to outputs/week01/
    """
    # os.makedirs creates all necessary parent directories
    # exist_ok=True means "don't error if directory already exists"
    os.makedirs(path, exist_ok=True)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Convert raw scores (logits) to probabilities that sum to 1.

    Softmax is a mathematical function commonly used to convert model outputs
    into probabilities. For example, if a model outputs [2.0, 1.0, 0.1], softmax
    converts these to [0.66, 0.24, 0.10] - probabilities that sum to 1.0.

    For beginners: Think of softmax as converting "confidence scores" into
    "probabilities". Higher input values become higher probabilities, and all
    probabilities add up to 100%.

    Key concept: The function uses a numerical stability trick (subtracting the max)
    to avoid overflow errors from very large exponentials.

    Parameters
    ----------
    x : np.ndarray
        Input array of raw scores (can be 1D like [2.0, 1.0, 0.1] or
        multi-dimensional like prediction scores for many examples)
    axis : int, default=-1
        Which axis (dimension) to compute softmax over. Default -1 means
        the last axis (most common). For a 2D array of shape [N, C] where
        N=number of examples and C=number of classes, axis=-1 computes
        softmax over classes for each example separately.

    Returns
    -------
    np.ndarray
        Probability distribution with same shape as input. Values are between
        0 and 1, and sum to 1.0 along the specified axis.

    Example
    -------
    >>> scores = np.array([2.0, 1.0, 0.1])  # Raw model outputs
    >>> probs = softmax(scores)
    >>> print(probs)
    [0.659  0.242  0.099]
    >>> print(probs.sum())  # Probabilities sum to 1
    1.0

    >>> # Multi-class example: 3 examples, 3 classes each
    >>> scores_batch = np.array([[2.0, 1.0, 0.1],
    ...                           [0.5, 0.5, 0.5],
    ...                           [1.0, 2.0, 3.0]])
    >>> probs_batch = softmax(scores_batch, axis=1)  # Softmax over classes (axis=1)
    >>> print(probs_batch[0])  # Probabilities for first example
    [0.659  0.242  0.099]
    """
    # Step 1: Subtract max for numerical stability
    # Why? Without this, np.exp() could produce numbers too large to represent (overflow)
    # For example: np.exp(1000) would overflow, but np.exp(1000 - 1000) = np.exp(0) = 1
    # Subtracting the max keeps numbers in a safe range while preserving the result
    # keepdims=True maintains the original array shape for proper broadcasting
    x = x - np.max(x, axis=axis, keepdims=True)

    # Step 2: Compute exponentials (e^x for each element)
    # This is the core of softmax: exp() makes larger values much larger
    e = np.exp(x)

    # Step 3: Divide by sum to get probabilities
    # Each value divided by the sum of all values = probability distribution
    # Adding 1e-12 (tiny number) prevents division by zero in edge cases
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-12)


def is_colab() -> bool:
    """Check if code is running in Google Colab environment.

    Google Colab is a free cloud platform for running Python notebooks. This
    function detects if your code is running in Colab by checking environment
    variables that Colab sets.

    For beginners: Sometimes you want code to behave differently in Colab vs.
    locally (e.g., different file paths, different visualization settings).
    This function helps you detect which environment you're in.

    Returns
    -------
    bool
        True if running in Google Colab, False otherwise

    Example
    -------
    >>> if is_colab():
    ...     print("Running in Colab - using /content/ paths")
    ... else:
    ...     print("Running locally - using relative paths")
    """
    # Check for Colab-specific environment variables
    # COLAB_GPU: Set when Colab is running (even without GPU)
    # PYTHONPATH: Contains 'google.colab' when in Colab environment
    return 'COLAB_GPU' in os.environ or 'google.colab' in str(os.environ.get('PYTHONPATH', ''))
