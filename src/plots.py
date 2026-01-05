"""Visualization utilities for exploratory data analysis and model evaluation.

This module provides plotting functions for common visualizations in the FEVER
course project. Visualizations help you understand your data and model performance
at a glance.

Common visualizations:
1. **Label distribution**: How balanced are the classes (SUPPORTS/REFUTES/NEI)?
2. **Histograms**: Distribution of numerical values (text length, confidence scores)
3. **Confusion matrix**: Which classes does the model confuse with each other?

For beginners: "A picture is worth a thousand numbers" - plots make patterns
visible that are hard to see in raw data. Use these functions to explore your
data and diagnose model behavior.
"""

from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt  # For creating plots
import numpy as np
import pandas as pd


def plot_label_counts(df: pd.DataFrame, title: str = "Label distribution") -> None:
    """Plot bar chart showing count of each label in the dataset.

    For beginners: This creates a bar chart showing how many examples have each
    label (SUPPORTS, REFUTES, NOT ENOUGH INFO). Useful for checking if your
    dataset is balanced or imbalanced.

    Example output:
    ```
    SUPPORTS          1500 ████████████████
    REFUTES           1200 █████████████
    NOT ENOUGH INFO    800 ████████
    ```

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a "label" column
    title : str, default="Label distribution"
        Title for the plot

    Example
    -------
    >>> from src.data_loading import load_fever_gold
    >>> df = load_fever_gold(split="train", sample_size=1000)
    >>> plot_label_counts(df)
    # Displays bar chart showing label distribution
    """
    # Count how many times each label appears
    # For beginners: .value_counts() counts occurrences of each unique value
    # .sort_index() sorts by label name alphabetically (consistent ordering)
    counts = df["label"].value_counts().sort_index()

    # Create a new figure (plot window)
    # For beginners: matplotlib uses "figures" to hold plots
    plt.figure()

    # Create bar plot
    # For beginners: .plot(kind="bar") creates a bar chart from the counts
    counts.plot(kind="bar")

    # Add labels and title
    # For beginners: These make the plot readable and self-explanatory
    plt.title(title)
    plt.xlabel("Label")  # X-axis label
    plt.ylabel("Count")  # Y-axis label

    # Adjust layout to prevent labels from being cut off
    # For beginners: tight_layout() automatically adjusts spacing
    plt.tight_layout()

    # Display the plot
    # For beginners: .show() opens a window displaying the plot
    # In Jupyter notebooks, this displays inline; in scripts, opens a window
    plt.show()


def plot_hist(series: pd.Series, title: str, xlabel: str) -> None:
    """Plot histogram showing distribution of numerical values.

    Histograms show how values are distributed. They divide the range into bins
    and count how many values fall in each bin.

    For beginners: Think of a histogram like sorting values into buckets. For
    example, if plotting sentence lengths, one bucket might be "0-10 words"
    (containing 50 sentences), another "10-20 words" (containing 120 sentences), etc.

    Use cases:
    - Text length distribution: Are claims short or long?
    - Confidence scores: Is the model often confident or uncertain?
    - Evidence count: How many evidence sentences per claim?

    Parameters
    ----------
    series : pd.Series
        Pandas Series containing numerical values to plot
        For example: df["claim"].apply(lambda x: len(x.split()))  # word counts
    title : str
        Title for the plot
    xlabel : str
        Label for x-axis (what are we plotting?)
        For example: "Number of words in claim"

    Example
    -------
    >>> df = load_fever_gold(split="train", sample_size=1000)
    >>> claim_lengths = df["claim"].apply(lambda x: len(x.split()))
    >>> plot_hist(claim_lengths, "Claim Length Distribution", "Number of words")
    # Displays histogram of claim lengths
    """
    # Create a new figure
    plt.figure()

    # Create histogram
    # For beginners: plt.hist() creates a histogram
    # - series.values extracts the numerical values from the Series
    # - bins=30 means divide the range into 30 buckets
    #   (more bins = finer detail, fewer bins = simpler overview)
    plt.hist(series.values, bins=30)

    # Add labels and title
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")  # Y-axis shows count in each bin

    # Adjust layout
    plt.tight_layout()

    # Display
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    title: str = "Confusion matrix"
) -> None:
    """Plot confusion matrix as a heatmap with annotations.

    A confusion matrix shows which classes your model confuses with each other.
    Rows = true labels, columns = predicted labels. The diagonal shows correct
    predictions, off-diagonal shows errors.

    For beginners: This is the best way to diagnose classification errors. For example:
    - If row "REFUTES" has high values in column "SUPPORTS", the model often
      mistakes REFUTES for SUPPORTS
    - If diagonal values are much larger than off-diagonal, model is accurate

    Example confusion matrix:
    ```
                   Predicted:    Predicted:    Predicted:
                    SUPPORTS      REFUTES        NEI
    True:SUPPORTS      90            5            5      ← 90% correct
    True:REFUTES        3           92            5      ← 92% correct
    True:NEI           10           10           80      ← 80% correct, confused with both
    ```

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix from sklearn.metrics.confusion_matrix()
        Shape: [n_classes, n_classes]
        For example: 3x3 matrix for SUPPORTS/REFUTES/NEI
    labels : List[str]
        Class names in same order as confusion matrix
        For FEVER: ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    title : str, default="Confusion matrix"
        Title for the plot

    Example
    -------
    >>> from src.evaluation import classification_report
    >>> from src.config import LABELS
    >>> report = classification_report(y_true, y_pred, labels=LABELS)
    >>> plot_confusion_matrix(report.cm, labels=LABELS)
    # Displays confusion matrix heatmap
    """
    # Create a new figure
    plt.figure()

    # Display confusion matrix as an image (heatmap)
    # For beginners: plt.imshow() displays a 2D array as an image
    # - cm is the confusion matrix (higher values = brighter colors)
    # - interpolation="nearest" means no smoothing (keep values exact)
    plt.imshow(cm, interpolation="nearest")

    # Add title
    plt.title(title)

    # Add color bar showing what colors mean
    # For beginners: colorbar shows the scale (darker = fewer, brighter = more)
    plt.colorbar()

    # Set up tick marks for axes
    # For beginners: We want to show class names instead of numbers 0, 1, 2
    # np.arange(len(labels)) creates [0, 1, 2, ...] for positioning
    tick_marks = np.arange(len(labels))

    # Set x-axis tick labels (predicted labels)
    # For beginners: rotation=45 tilts labels to prevent overlap
    # ha="right" means "horizontal alignment = right" (aligns rotated text nicely)
    plt.xticks(tick_marks, labels, rotation=45, ha="right")

    # Set y-axis tick labels (true labels)
    plt.yticks(tick_marks, labels)

    # ====== Annotate cells with numbers ======
    # For beginners: We'll add the actual count to each cell for clarity

    # Calculate threshold for text color (black or white)
    # For beginners: On dark cells, use white text; on light cells, use black text
    # cm.max() / 2.0 is the midpoint - values above this are "dark", below are "light"
    # if cm.size is a safety check for empty matrices
    thresh = cm.max() / 2.0 if cm.size else 0.0

    # Loop through all cells and add text annotations
    # For beginners: Nested loops to process each cell in the matrix
    # - i iterates over rows (true labels)
    # - j iterates over columns (predicted labels)
    for i in range(cm.shape[0]):  # cm.shape[0] = number of rows
        for j in range(cm.shape[1]):  # cm.shape[1] = number of columns
            # Add text annotation to cell (i, j)
            # For beginners: plt.text(x, y, text, ...) adds text at position (x, y)
            # - j, i is the position (note: x=j, y=i for column, row)
            # - format(int(cm[i, j]), "d") converts the count to a string (e.g., "90")
            # - ha="center" means horizontal alignment = center
            # - va="center" means vertical alignment = center
            # - color: white for dark cells, black for light cells (for readability)
            plt.text(
                j, i,  # Position: column j, row i
                format(int(cm[i, j]), "d"),  # The count value as a string
                ha="center",  # Center horizontally
                va="center",  # Center vertically
                color="white" if cm[i, j] > thresh else "black"  # Text color
            )

    # Add axis labels
    # For beginners: These clarify what rows and columns represent
    plt.ylabel("True label")
    plt.xlabel("Predicted label")

    # Adjust layout
    plt.tight_layout()

    # Display
    plt.show()
