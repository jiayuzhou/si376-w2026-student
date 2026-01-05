"""Natural Language Inference (NLI) for fact verification.

NLI (Natural Language Inference), also called Textual Entailment, is a task where a
model determines the relationship between two sentences:
- **Premise**: A sentence providing context/evidence
- **Hypothesis**: A sentence to verify
- **Labels**: ENTAILMENT / NEUTRAL / CONTRADICTION

For fact-checking, we use NLI to verify claims against evidence:
- Premise = Evidence sentence
- Hypothesis = Claim to verify
- ENTAILMENT → Evidence supports the claim → SUPPORTS
- CONTRADICTION → Evidence contradicts the claim → REFUTES
- NEUTRAL → Evidence is unrelated/insufficient → NOT ENOUGH INFO

Example:
```
Premise: "Albert Einstein was a theoretical physicist."
Hypothesis: "Einstein was a scientist."
→ ENTAILMENT (premise supports hypothesis)

Premise: "Albert Einstein was a theoretical physicist."
Hypothesis: "Einstein was a chemist."
→ CONTRADICTION (premise contradicts hypothesis)

Premise: "Albert Einstein was a theoretical physicist."
Hypothesis: "Einstein liked classical music."
→ NEUTRAL (premise doesn't address this)
```

Why NLI for fact-checking?
- Leverages pre-trained models (no large fact-checking dataset needed)
- Works sentence-by-sentence (interpretable)
- Can aggregate multiple evidence sentences
- Used in Week 8 and agent pipelines (Weeks 13-14)

For beginners: Think of NLI as a "does A prove/disprove B?" detector. It's like
a logical reasoning module that understands when one statement follows from another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np

from .utils import softmax


# ============================================================
# Data Classes
# ============================================================
@dataclass
class NLIPrediction:
    """Result from an NLI model prediction.

    For beginners: This simple object holds the NLI model's prediction - both
    the best label and the probability distribution across all labels.

    Attributes
    ----------
    label : str
        Predicted label (highest probability)
        One of: "ENTAILMENT", "NEUTRAL", "CONTRADICTION"
    probs : Dict[str, float]
        Probability for each label
        For example: {"ENTAILMENT": 0.85, "NEUTRAL": 0.10, "CONTRADICTION": 0.05}
        Probabilities sum to 1.0
    """
    label: str               # Predicted label (ENTAILMENT/NEUTRAL/CONTRADICTION)
    probs: Dict[str, float]  # Probability distribution over labels


# ============================================================
# NLI Model Loading
# ============================================================
def load_nli_model(
    model_name: str = "huggingface/distilbert-base-uncased-finetuned-mnli"
) -> Tuple[Any, Any]:
    """Load a pre-trained NLI model and tokenizer from HuggingFace.

    For beginners: This downloads a neural network that's been trained to recognize
    entailment/neutral/contradiction relationships. The default model (distilbert-mnli)
    is trained on MNLI (Multi-Genre Natural Language Inference), a large NLI dataset.

    The model has two parts:
    1. **Tokenizer**: Converts text to numbers (tokens) the model can process
    2. **Model**: The actual neural network that makes predictions

    Parameters
    ----------
    model_name : str, default="huggingface/distilbert-base-uncased-finetuned-mnli"
        HuggingFace model identifier. Options:
        - "huggingface/distilbert-base-uncased-finetuned-mnli" (default, ~250MB, fast)
        - "facebook/bart-large-mnli" (~1.5GB, more accurate, slower)
        - "microsoft/deberta-large-mnli" (~1.5GB, state-of-the-art)
        For beginners: Larger models are more accurate but slower. Use default for
        classroom/laptop use.

    Returns
    -------
    Tuple[Any, Any]
        (tokenizer, model) - both from transformers library
        For beginners: You'll pass both to nli_predict() to make predictions

    Example
    -------
    >>> tokenizer, model = load_nli_model()
    Downloading model... (first time only, ~30 seconds)
    >>> # Now ready to use with nli_predict()
    """
    # Import HuggingFace transformers library
    # For beginners: transformers is the standard library for pre-trained NLP models
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # Load tokenizer (converts text to token IDs)
    # For beginners: from_pretrained() downloads the model the first time, then caches it
    # Auto classes automatically detect the right architecture from the model name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model (the neural network)
    # For beginners: SequenceClassification means it classifies text pairs (premise+hypothesis)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    return tokenizer, model


# ============================================================
# NLI Prediction
# ============================================================
def nli_predict(
    premise: str,
    hypothesis: str,
    tokenizer: Any,
    model: Any
) -> NLIPrediction:
    """Run NLI model on premise-hypothesis pair.

    For beginners: Given two sentences (premise = evidence, hypothesis = claim),
    determine if the premise entails, contradicts, or is neutral to the hypothesis.

    How it works:
    1. Tokenize: Convert text to numbers
    2. Forward pass: Run neural network
    3. Softmax: Convert outputs to probabilities
    4. Return: Best label + all probabilities

    Parameters
    ----------
    premise : str
        The premise sentence (context/evidence)
        For fact-checking: This is an evidence sentence
    hypothesis : str
        The hypothesis sentence (claim to verify)
        For fact-checking: This is the claim
    tokenizer : AutoTokenizer
        Tokenizer from load_nli_model()
    model : AutoModelForSequenceClassification
        Model from load_nli_model()

    Returns
    -------
    NLIPrediction
        Prediction with label and probability distribution

    Example
    -------
    >>> tokenizer, model = load_nli_model()
    >>> pred = nli_predict(
    ...     premise="Einstein was a physicist.",
    ...     hypothesis="Einstein was a scientist.",
    ...     tokenizer=tokenizer,
    ...     model=model
    ... )
    >>> print(pred.label)
    'ENTAILMENT'
    >>> print(pred.probs)
    {'ENTAILMENT': 0.92, 'NEUTRAL': 0.05, 'CONTRADICTION': 0.03}
    """
    # Import PyTorch (needed for tensor operations)
    # For beginners: PyTorch is the deep learning framework transformers uses
    import torch

    # Set model to evaluation mode (disables dropout, batch normalization updates)
    # For beginners: Neural networks have different behavior during training vs inference
    # .eval() tells the model "we're just making predictions, not training"
    model.eval()

    # Run inference without computing gradients (saves memory and time)
    # For beginners: Gradients are used for training. During inference, we don't need them.
    # torch.no_grad() is like saying "I just want predictions, not training"
    with torch.no_grad():
        # Tokenize the premise and hypothesis pair
        # For beginners: Tokenization converts text to numbers (token IDs)
        # - premise, hypothesis: The two sentences to compare
        # - return_tensors="pt": Return PyTorch tensors (needed by model)
        # - truncation=True: Cut off text longer than max_length
        # - max_length=256: Maximum sequence length (longer = more memory/time)
        inputs = tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",  # PyTorch tensors
            truncation=True,      # Truncate if too long
            max_length=256        # Max tokens (words ≈ tokens × 0.75)
        )

        # Run the model (forward pass through neural network)
        # For beginners: **inputs unpacks the dictionary into keyword arguments
        # model(**inputs) passes input_ids, attention_mask, etc. to the model
        # .logits are the raw model outputs (not yet probabilities)
        # .detach() disconnects from computation graph (no gradients)
        # .cpu() moves from GPU to CPU (if using GPU)
        # .numpy() converts PyTorch tensor to NumPy array
        # [0] gets first (and only) example from the batch
        logits = model(**inputs).logits.detach().cpu().numpy()[0]

        # Convert logits to probabilities using softmax
        # For beginners: logits are unbounded scores, softmax converts to probabilities (sum to 1.0)
        probs = softmax(logits)

        # Get label names from model configuration
        # For beginners: id2label maps 0→"entailment", 1→"neutral", 2→"contradiction"
        # Different models may use different orderings, so we read from config
        id2label = model.config.id2label

        # Create probability dictionary with label names
        # For beginners: This is a dictionary comprehension
        # For each label index i, map id2label[i].upper() (label name) → probs[i] (probability)
        # .upper() converts "entailment" → "ENTAILMENT" for consistency
        prob_map = {id2label[i].upper(): float(probs[i]) for i in range(len(probs))}

        # Find label with highest probability
        # For beginners: max(..., key=...) finds the key with maximum value
        # key=prob_map.get means "use the probability value for comparison"
        best_label = max(prob_map, key=prob_map.get)

        # Return prediction
        return NLIPrediction(label=best_label, probs=prob_map)


# ============================================================
# Aggregation for Fact-Checking
# ============================================================
def aggregate_verdict_from_nli(
    nli_preds: List[NLIPrediction],
    entail_thresh: float = 0.5,
    contra_thresh: float = 0.5
) -> str:
    """Aggregate sentence-level NLI predictions into a FEVER-style verdict.

    For fact-checking, we often have MULTIPLE evidence sentences. This function
    combines their NLI predictions into a single SUPPORTS/REFUTES/NOT ENOUGH INFO
    verdict using a max-aggregation strategy.

    Strategy:
    1. Find the strongest ENTAILMENT score across all evidence
    2. Find the strongest CONTRADICTION score across all evidence
    3. If contradiction is strongest and above threshold → REFUTES
    4. Else if entailment is strongest and above threshold → SUPPORTS
    5. Else → NOT ENOUGH INFO

    For beginners: Think of it like this: "If ANY evidence strongly contradicts
    the claim, it's REFUTES. Else if ANY evidence strongly supports it, it's
    SUPPORTS. Otherwise, there's not enough clear evidence."

    Example:
    ```
    Evidence 1: "Einstein was a physicist" vs Claim: "Einstein was a scientist"
      → ENTAILMENT (0.9)
    Evidence 2: "Einstein won Nobel Prize" vs Claim: "Einstein was a scientist"
      → ENTAILMENT (0.7)
    Evidence 3: "Einstein lived in Germany" vs Claim: "Einstein was a scientist"
      → NEUTRAL (0.8)

    Max ENTAILMENT = 0.9, Max CONTRADICTION = 0.0
    → 0.9 > 0.5 threshold → SUPPORTS
    ```

    Parameters
    ----------
    nli_preds : List[NLIPrediction]
        NLI predictions for each evidence sentence
        For example: [pred1, pred2, pred3] where each pred is from nli_predict()
    entail_thresh : float, default=0.5
        Minimum ENTAILMENT probability to return SUPPORTS
        For beginners: Higher threshold = more conservative (need stronger evidence)
    contra_thresh : float, default=0.5
        Minimum CONTRADICTION probability to return REFUTES
        For beginners: Higher threshold = more conservative

    Returns
    -------
    str
        FEVER-style verdict: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"

    Example
    -------
    >>> preds = [
    ...     nli_predict(ev1, claim, tokenizer, model),
    ...     nli_predict(ev2, claim, tokenizer, model),
    ... ]
    >>> verdict = aggregate_verdict_from_nli(preds)
    >>> print(verdict)
    'SUPPORTS'
    """
    # Initialize trackers for best scores
    # For beginners: We'll update these as we process each prediction
    best_entail = 0.0  # Highest ENTAILMENT probability seen
    best_contra = 0.0  # Highest CONTRADICTION probability seen

    # Process each NLI prediction
    # For beginners: Loop through all evidence-claim comparisons
    for p in nli_preds:
        # Update best ENTAILMENT score
        # For beginners: .get("ENTAILMENT", 0.0) safely gets the value, defaulting to 0.0 if missing
        # max() keeps the higher value between current best and this prediction
        best_entail = max(best_entail, p.probs.get("ENTAILMENT", 0.0))

        # Update best CONTRADICTION score
        best_contra = max(best_contra, p.probs.get("CONTRADICTION", 0.0))

    # Decide verdict based on thresholds and relative strengths
    # For beginners: We check conditions in priority order (contradiction first, then entailment)

    # If strong contradiction found AND it's stronger than entailment → REFUTES
    # For beginners: >= contra_thresh ensures contradiction is confident enough
    # >= best_entail ensures contradiction is at least as strong as entailment
    if best_contra >= contra_thresh and best_contra >= best_entail:
        return "REFUTES"

    # Else if strong entailment found AND it's stronger than contradiction → SUPPORTS
    # For beginners: >= entail_thresh ensures entailment is confident enough
    # > best_contra ensures entailment is strictly stronger than contradiction
    if best_entail >= entail_thresh and best_entail > best_contra:
        return "SUPPORTS"

    # Otherwise, no strong evidence either way → NOT ENOUGH INFO
    # For beginners: This happens when:
    # - All predictions are NEUTRAL
    # - ENTAILMENT/CONTRADICTION scores are below thresholds
    # - Scores are similar (inconclusive)
    return "NOT ENOUGH INFO"
