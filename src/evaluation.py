"""Evaluation metrics for fact-checking models.

This module provides functions to measure how well models perform on fact-checking
tasks. Different metrics capture different aspects of model quality:

1. **Accuracy**: What percentage of predictions are correct?
   - Simple and intuitive
   - Can be misleading if classes are imbalanced

2. **F1 Score**: Harmonic mean of precision and recall
   - Better for imbalanced datasets
   - Macro F1 treats all classes equally (good for FEVER's 3 classes)

3. **Confusion Matrix**: Table showing which classes get confused with each other
   - Helps diagnose specific error patterns (e.g., "SUPPORTS" confused with "REFUTES")

4. **Expected Calibration Error (ECE)**: Are probability predictions trustworthy?
   - If model says 80% confident, is it right 80% of the time?
   - Important for decision-making and abstention (Week 5)

5. **Recall@k**: For retrieval, do we find relevant evidence in top-k results?
   - Used to evaluate retrieval systems (Weeks 4, 12)

For beginners: Evaluation metrics help you answer "is my model good?" Different
metrics reveal different strengths and weaknesses.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# ============================================================
# Classification Metrics
# ============================================================
@dataclass
class ClassificationReport:
    """Container for classification evaluation metrics.

    For beginners: This is a simple object that holds three related metrics
    computed from the same predictions. Using a dataclass keeps them organized.

    Attributes
    ----------
    accuracy : float
        Fraction of predictions that are correct (0.0 to 1.0)
        For example: 0.85 = 85% accuracy
    macro_f1 : float
        Macro-averaged F1 score across all classes (0.0 to 1.0)
        For beginners: F1 balances precision (correctness) and recall (coverage)
        "Macro" means we compute F1 for each class, then average them (treats
        all classes equally, even if some have fewer examples)
    cm : np.ndarray
        Confusion matrix showing prediction patterns
        For beginners: A table where rows = true labels, columns = predictions
        Diagonal = correct predictions, off-diagonal = errors
        Example:
                  Pred:SUPP  Pred:REF  Pred:NEI
        True:SUPP    90         5         5       (90% correct, 5% confused with REFUTES)
        True:REF      3        92         5       (92% correct)
        True:NEI     10        10        80       (80% correct, confused with both)
    """
    accuracy: float      # Overall accuracy (0-1)
    macro_f1: float      # Macro F1 score (0-1)
    cm: np.ndarray       # Confusion matrix [n_classes, n_classes]


def classification_report(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str]
) -> ClassificationReport:
    """Compute classification metrics (accuracy, F1, confusion matrix).

    For beginners: This function compares true labels with predicted labels
    and computes several metrics that tell you how well your model performed.

    Parameters
    ----------
    y_true : List[str]
        True labels (ground truth) for each example
        For example: ["SUPPORTS", "REFUTES", "SUPPORTS", ...]
    y_pred : List[str]
        Predicted labels from your model (same length as y_true)
        For example: ["SUPPORTS", "SUPPORTS", "SUPPORTS", ...]
    labels : List[str]
        List of all possible labels (defines order for confusion matrix)
        For FEVER: ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]

    Returns
    -------
    ClassificationReport
        Object containing accuracy, macro_f1, and confusion matrix
    """
    # Compute accuracy: fraction of correct predictions
    # For beginners: sklearn's accuracy_score counts matching labels and divides by total
    acc = float(accuracy_score(y_true, y_pred))

    # Compute macro F1 score
    # For beginners: F1 = 2 * (precision * recall) / (precision + recall)
    # - Precision: Of predictions labeled X, how many are truly X? (quality)
    # - Recall: Of true X examples, how many did we predict as X? (coverage)
    # - average="macro": Compute F1 for each class, then average (treats classes equally)
    macro = float(f1_score(y_true, y_pred, labels=labels, average="macro"))

    # Compute confusion matrix
    # For beginners: Shows which classes get confused with each other
    # cm[i, j] = number of examples with true label i predicted as label j
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return ClassificationReport(accuracy=acc, macro_f1=macro, cm=cm)


# ============================================================
# Calibration Metrics
# ============================================================
def expected_calibration_error(
    probs: np.ndarray,
    y_true: np.ndarray,
    n_bins: int = 10
) -> float:
    """Compute Expected Calibration Error (ECE) for multiclass classification.

    Calibration measures whether a model's confidence scores are trustworthy.
    A well-calibrated model that says "80% confident" should be correct 80% of
    the time. ECE quantifies the mismatch between confidence and actual accuracy.

    For beginners: Imagine your model predicts with these confidences:
    - Example 1: 90% confident → correct! (well-calibrated)
    - Example 2: 90% confident → wrong! (overconfident)
    - Example 3: 90% confident → correct! (well-calibrated)
    If 2 out of 3 are correct, actual accuracy is 67%, not 90% → poorly calibrated!

    ECE Algorithm:
    1. Group predictions into bins by confidence (0-10%, 10-20%, ..., 90-100%)
    2. For each bin, compare average confidence to actual accuracy in that bin
    3. ECE = weighted average of |confidence - accuracy| across bins

    Lower ECE = better calibration (0.0 = perfectly calibrated)

    For beginners: ECE tells you if you can trust model probabilities for
    decision-making (e.g., "only act when model is >90% confident").

    Parameters
    ----------
    probs : np.ndarray
        Predicted probability distributions, shape [N, C]
        where N = number of examples, C = number of classes
        For example: [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], ...]
        (each row sums to 1.0)
    y_true : np.ndarray
        True class labels as integers, shape [N]
        For example: [0, 1, 2, 0, ...] where 0=SUPPORTS, 1=REFUTES, 2=NEI
    n_bins : int, default=10
        Number of confidence bins to use (10 = deciles: 0-10%, 10-20%, ...)
        For beginners: More bins = finer-grained analysis but need more data

    Returns
    -------
    float
        Expected Calibration Error (0.0 to 1.0, lower is better)
        For example: 0.05 means average 5% mismatch between confidence and accuracy
    """
    # ====== Step 1: Extract confidence scores and predictions ======
    # For beginners: We use the max probability as the model's confidence
    # For example: probs = [0.7, 0.2, 0.1] → confidence = 0.7 (most confident class)

    # Get maximum probability for each example (model's confidence)
    # For beginners: .max(axis=1) finds the max along axis 1 (across columns)
    # - probs.shape is [N, C] (rows=examples, cols=classes)
    # - axis=1 means "for each row, find the max across columns"
    # - Result shape: [N] (one confidence score per example)
    confidences = probs.max(axis=1)

    # Get predicted class for each example (class with highest probability)
    # For beginners: .argmax(axis=1) finds the INDEX of the max along axis 1
    # For example: [0.7, 0.2, 0.1] → argmax = 0 (first class has highest prob)
    preds = probs.argmax(axis=1)

    # Check which predictions are correct (1.0 = correct, 0.0 = wrong)
    # For beginners: preds == y_true creates boolean array [True, False, True, ...]
    # .astype(np.float32) converts to numbers [1.0, 0.0, 1.0, ...]
    correct = (preds == y_true).astype(np.float32)

    # ====== Step 2: Create confidence bins ======
    # For beginners: We divide the 0-1 confidence range into n_bins equal parts
    # np.linspace(0.0, 1.0, n_bins+1) creates bin edges
    # For n_bins=10: [0.0, 0.1, 0.2, 0.3, ..., 0.9, 1.0]
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    # Initialize ECE accumulator
    ece = 0.0

    # ====== Step 3: Process each bin ======
    # For beginners: For each confidence range (e.g., 60-70%), we compare
    # the average confidence (e.g., 65%) to actual accuracy in that range
    for i in range(n_bins):
        # Get lower and upper bounds for this bin
        # For example: bin 6 has lo=0.6, hi=0.7 (60-70% confidence)
        lo, hi = bins[i], bins[i + 1]

        # Create mask: which examples fall in this confidence bin?
        # For beginners: & is "and", creates boolean array showing which
        # examples have confidence in range [lo, hi)
        mask = (confidences >= lo) & (confidences < hi)

        # Skip empty bins (no examples in this confidence range)
        # For beginners: .sum() counts True values in the mask
        if mask.sum() == 0:
            continue

        # Compute average accuracy for examples in this bin
        # For beginners: correct[mask] selects only the examples in this bin
        # .mean() computes their average (fraction that are correct)
        acc_bin = correct[mask].mean()

        # Compute average confidence for examples in this bin
        conf_bin = confidences[mask].mean()

        # Add weighted contribution to ECE
        # For beginners: This computes |accuracy - confidence| for this bin,
        # weighted by the fraction of examples in this bin
        # - mask.mean() = fraction of all examples in this bin (weight)
        # - abs(acc_bin - conf_bin) = calibration error for this bin
        ece += (mask.mean()) * abs(acc_bin - conf_bin)

    # Return ECE
    # For beginners: Lower ECE = better calibration
    # - ECE < 0.05: Well-calibrated (confidence matches accuracy within 5%)
    # - ECE > 0.15: Poorly calibrated (confidence is misleading)
    return float(ece)


# ============================================================
# Retrieval Metrics
# ============================================================
def recall_at_k(
    retrieved_keys: List[List[str]],
    gold_keys: List[List[str]],
    k: int
) -> float:
    """Compute Recall@k for retrieval: fraction of queries with ANY gold result in top-k.

    Recall@k measures whether a retrieval system finds relevant results within
    the top-k retrieved items. It's a pass/fail metric: either we find something
    relevant in top-k (success) or we don't (failure).

    For beginners: Think of it like this:
    - You search for "Albert Einstein physicist" on Google
    - Google shows 10 results (k=10)
    - If ANY of those 10 results are actually about Einstein's physics work → success!
    - Recall@10 = fraction of queries where at least one relevant result appears in top-10

    This is different from "precision" which cares about how MANY of the top-k are relevant.
    Recall@k just asks: did we find AT LEAST ONE relevant result?

    Example:
    - Query 1: top-5 retrieved = [A, B, C, D, E], gold = [B, X, Y] → success! (B found)
    - Query 2: top-5 retrieved = [F, G, H, I, J], gold = [K, L] → failure (none found)
    - Query 3: top-5 retrieved = [M, N, O, P, Q], gold = [P, Q, R] → success! (P and Q found)
    - Recall@5 = 2/3 = 0.67 (2 successes out of 3 queries)

    Use cases:
    - Evaluating retrieval for RAG (Week 12): Do we retrieve relevant evidence?
    - Optimizing retrieval systems: Higher recall@k = more likely to find what you need

    Parameters
    ----------
    retrieved_keys : List[List[str]]
        Retrieved results for each query, outer list = queries, inner list = results
        For example: [["Page1::0", "Page2::5", ...], ["Page3::1", ...], ...]
    gold_keys : List[List[str]]
        Gold (relevant) keys for each query (same length as retrieved_keys)
        For example: [["Page1::0", "Page1::2"], ["Page3::1"], ...]
    k : int
        How many top results to consider (e.g., k=5 means top-5)

    Returns
    -------
    float
        Recall@k score between 0.0 and 1.0
        For example: 0.75 means 75% of queries found at least one relevant result in top-k
    """
    # Initialize hit counter
    hits = 0

    # Get number of queries
    # For beginners: We'll compute recall as hits / total_queries
    n = len(gold_keys)

    # Process each query
    # For beginners: zip(retrieved_keys, gold_keys) pairs up corresponding elements
    # For example: zip([['A','B'], ['C','D']], [['X'], ['D']]) gives: (['A','B'], ['X']), (['C','D'], ['D'])
    for r, g in zip(retrieved_keys, gold_keys):
        # Take top-k retrieved results
        # For beginners: r[:k] is list slicing - takes first k elements
        # If r has fewer than k elements, it just takes all of them (no error)
        r_k = r[:k]

        # Check if ANY retrieved result matches ANY gold result
        # For beginners: set() converts list to set (unordered, unique items)
        # & is set intersection (items in both sets)
        # len(...) > 0 checks if intersection is non-empty (at least one match)
        # For example: set(['A','B']) & set(['B','C']) = {'B'} → len > 0 → True
        if len(set(r_k) & set(g)) > 0:
            hits += 1  # Count this as a successful query

    # Compute recall: hits / total queries
    # For beginners: max(n, 1) prevents division by zero if n=0
    return hits / max(n, 1)
