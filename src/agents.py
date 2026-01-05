"""Fact-checking agent with tool use and interpretable tracing.

An **agent** is a system that autonomously performs a multi-step task using "tools"
(functions it can call). This module implements a simple but effective fact-checking
agent that combines retrieval, NLI, and explanation generation into an end-to-end
pipeline.

Agent Architecture (Three-Stage Pipeline):

1. **Retrieve**: Use retrieval tool to find candidate evidence sentences
   - Input: Claim
   - Tool: Retriever (TF-IDF, BM25, or embeddings)
   - Output: Top-k evidence sentences

2. **Verify**: Use NLI model to verify claim against each evidence sentence
   - Input: Claim + retrieved evidence
   - Tool: NLI model (entailment classifier)
   - Output: Verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO)

3. **Explain**: Generate human-readable explanation for the verdict
   - Input: Claim + verdict + top evidence
   - Tool: LLM or template
   - Output: Natural language explanation

Why Agents?
- **Modularity**: Each tool (retrieval, NLI, LLM) can be swapped independently
- **Interpretability**: Trace shows exactly what the agent did at each step
- **Flexibility**: Can add new tools or change the pipeline easily
- **Transparency**: Users see the reasoning process, not just a black box

For beginners: Think of an agent like a detective following a process:
1. Gather evidence (retrieval)
2. Check if evidence supports or contradicts the claim (NLI)
3. Write a report explaining the verdict (LLM/template)

Used in Weeks 13-14 for single-agent and multi-agent fact-checking systems.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from .llm import explain
from .nli import aggregate_verdict_from_nli, nli_predict


# ============================================================
# Data Classes
# ============================================================
@dataclass
class ToolTraceStep:
    """Record of a single tool use by the agent.

    For interpretability and debugging, we record each step the agent takes.
    This is called "tracing" - like leaving a trail of breadcrumbs showing
    what the agent did.

    For beginners: When an agent makes a mistake, the trace helps you see where
    things went wrong (bad retrieval? Wrong NLI prediction? Poor explanation?).

    Attributes
    ----------
    tool : str
        Name of the tool used
        For example: "retrieve", "verify_nli", "explain"
    input : str
        Summary of what was passed to the tool
        For example: "claim=Einstein was a scientist, top_k=5"
    output_summary : str
        Summary of what the tool returned
        For example: "retrieved 5 sentences" or "verdict=SUPPORTS"
    """
    tool: str             # Tool name
    input: str            # Input summary
    output_summary: str   # Output summary


@dataclass
class FactCheckResult:
    """Complete result from fact-checking agent.

    For beginners: This is the final output of the agent - everything it found,
    decided, and explained, plus the trace showing how it got there.

    Attributes
    ----------
    claim : str
        The original claim that was verified
    verdict : str
        Final verdict: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
    evidence : List[Tuple[str, float, str]]
        Retrieved evidence sentences, each as (key, score, text)
        For example: [("Page1::0", 0.85, "Einstein was a physicist"), ...]
    explanation : str
        Human-readable explanation of the verdict
        For example: "The evidence supports the claim. According to [E1]..."
    trace : List[ToolTraceStep]
        Step-by-step record of what the agent did
        For debugging and interpretability
    """
    claim: str                                 # Original claim
    verdict: str                               # Final verdict
    evidence: List[Tuple[str, float, str]]     # Retrieved evidence (key, score, text)
    explanation: str                           # Generated explanation
    trace: List[ToolTraceStep]                 # Execution trace


# ============================================================
# Fact-Checking Agent
# ============================================================
class FactCheckAgent:
    """A tool-using agent for end-to-end FEVER-style fact-checking.

    This agent implements a simple but effective three-stage pipeline:
    Retrieve → Verify → Explain. It's "deterministic" (no randomness in the
    plan - always follows the same three steps) and "tool-using" (calls external
    functions to do the work).

    For beginners: Unlike a single model that tries to do everything, this agent
    is like a workflow that delegates each part to a specialized tool. Retrieval
    is good at finding evidence, NLI is good at logical reasoning, LLMs are good
    at explanation - combine them for a better system.

    Why this architecture?
    - **Better than end-to-end**: Each component is interpretable and replaceable
    - **Leverages pre-trained models**: Don't need large fact-checking datasets
    - **Modular**: Can improve one component without retraining everything
    - **Transparent**: Trace shows exactly what happened

    Attributes
    ----------
    retriever : Callable
        Retrieval function that takes (claim: str, top_k: int) and returns
        List[(key, score, text)]
        For example: lambda c, k: tfidf_retrieve(index, c, k)
    tokenizer : Any
        Transformers tokenizer for NLI model
    nli_model : Any
        Transformers NLI model for verification
    """

    def __init__(
        self,
        retriever: Callable[[str, int], List[Tuple[str, float, str]]],
        tokenizer: Any,
        nli_model: Any
    ):
        """Initialize fact-checking agent with tools.

        For beginners: The agent needs three "tools" (functions it can use):
        1. A retriever to find evidence
        2. An NLI model (tokenizer + model) to verify claims
        3. An LLM/template to explain (called in .run(), not stored here)

        Parameters
        ----------
        retriever : Callable[[str, int], List[Tuple[str, float, str]]]
            Function that takes (claim, top_k) and returns retrieved evidence
            For example: lambda c, k: tfidf_retrieve(my_index, c, k)
        tokenizer : Any
            HuggingFace tokenizer for NLI model (from load_nli_model())
        nli_model : Any
            HuggingFace NLI model (from load_nli_model())

        Example
        -------
        >>> from src.retrieval import build_tfidf_index, tfidf_retrieve
        >>> from src.nli import load_nli_model
        >>> # Build retrieval index
        >>> index = build_tfidf_index(keys, texts)
        >>> # Load NLI model
        >>> tokenizer, nli_model = load_nli_model()
        >>> # Create retriever function
        >>> retriever = lambda claim, top_k: tfidf_retrieve(index, claim, top_k)
        >>> # Create agent
        >>> agent = FactCheckAgent(retriever, tokenizer, nli_model)
        """
        # Store the tools
        # For beginners: We save these so the agent can use them later in .run()
        self.retriever = retriever      # Retrieval tool
        self.tokenizer = tokenizer      # NLI tokenizer
        self.nli_model = nli_model      # NLI model

    def run(
        self,
        claim: str,
        top_k: int = 5,
        use_llm: bool = True
    ) -> FactCheckResult:
        """Run the fact-checking pipeline on a claim.

        This is the main method - it orchestrates the three-stage pipeline and
        returns the complete result. The agent follows a fixed plan (always the
        same three steps) which is why we call it a "deterministic planner".

        Pipeline:
        1. **Retrieve**: Find top-k evidence sentences using retriever
        2. **Verify**: Run NLI on each (evidence, claim) pair and aggregate
        3. **Explain**: Generate explanation using LLM or template

        For beginners: This is like following a recipe - do step 1, then step 2,
        then step 3. The agent doesn't "think" about what to do next; it just
        follows the predefined plan.

        Parameters
        ----------
        claim : str
            Claim to verify
        top_k : int, default=5
            How many evidence sentences to retrieve
            For beginners: Higher k = more evidence but slower
        use_llm : bool, default=True
            Whether to use LLM for explanation (vs template)

        Returns
        -------
        FactCheckResult
            Complete result including verdict, evidence, explanation, and trace

        Example
        -------
        >>> result = agent.run("Einstein was a scientist")
        >>> print(result.verdict)
        'SUPPORTS'
        >>> print(result.explanation)
        'The evidence supports the claim. According to [E1], Einstein was a physicist...'
        >>> print(result.trace)
        [ToolTraceStep(tool='retrieve', ...), ToolTraceStep(tool='verify_nli', ...), ...]
        """
        # Initialize trace (we'll record each step)
        # For beginners: A trace is like a log showing what the agent did
        trace: List[ToolTraceStep] = []

        # ====== STAGE 1: Retrieve Evidence ======
        # For beginners: Use the retrieval tool to find candidate evidence sentences

        # Call retriever to get top-k evidence
        # For beginners: self.retriever is the function we saved in __init__
        # It returns a list of (key, score, text) tuples
        retrieved = self.retriever(claim, top_k=top_k)

        # Record this step in the trace
        # For beginners: We log what tool we used, what we asked for, and what we got
        trace.append(ToolTraceStep(
            tool="retrieve",
            input=f"claim={claim} top_k={top_k}",
            output_summary=f"retrieved {len(retrieved)} sentences"
        ))

        # ====== STAGE 2: Verify with NLI ======
        # For beginners: For each evidence sentence, check if it supports/contradicts/is-neutral to the claim

        # Run NLI on each (evidence, claim) pair
        # For beginners: We loop through all retrieved sentences and verify each one
        nli_preds = []
        for key, score, sent in retrieved:
            # Run NLI: Does this evidence (premise) entail/contradict the claim (hypothesis)?
            # For beginners: premise=evidence, hypothesis=claim
            p = nli_predict(
                premise=sent,           # Evidence sentence
                hypothesis=claim,       # Claim to verify
                tokenizer=self.tokenizer,
                model=self.nli_model
            )
            nli_preds.append(p)

        # Aggregate NLI predictions into final verdict
        # For beginners: Combine all the individual (evidence, claim) verdicts
        # into one overall verdict using the aggregation strategy
        verdict = aggregate_verdict_from_nli(nli_preds)

        # Record this step in the trace
        trace.append(ToolTraceStep(
            tool="verify_nli",
            input=f"{len(retrieved)} candidates",
            output_summary=f"verdict={verdict}"
        ))

        # ====== STAGE 3: Explain ======
        # For beginners: Generate a natural language explanation of the verdict

        # Extract just the text from top 3 evidence sentences for explanation
        # For beginners: We use top 3 (not all) to keep explanation concise
        # List comprehension: [text for (key, score, text) in retrieved[:3]]
        # retrieved[:3] takes first 3 items, then we extract just the text (t)
        evidence_texts = [t for (_, _, t) in retrieved[:3]]

        # Generate explanation using LLM or template
        # For beginners: explain() tries LLM first, falls back to template if use_llm=False or LLM fails
        expl = explain(claim, verdict, evidence_texts, use_llm=use_llm)

        # Record this step in the trace
        trace.append(ToolTraceStep(
            tool="explain",
            input="top3 evidence",
            output_summary=f"model={expl.model_name}"
        ))

        # ====== Return Complete Result ======
        # For beginners: Package everything into a FactCheckResult object
        return FactCheckResult(
            claim=claim,              # Original claim
            verdict=verdict,          # Final verdict from NLI aggregation
            evidence=retrieved,       # All retrieved evidence (with scores)
            explanation=expl.text,    # Generated explanation
            trace=trace               # Step-by-step trace of what agent did
        )
