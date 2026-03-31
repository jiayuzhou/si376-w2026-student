"""LangChain-based fact-checking agent.

This module provides an alternative implementation of the fact-checking agent
using LangChain, a popular framework for building LLM-powered applications.

Why LangChain?
- **Industry standard**: Widely used in production systems
- **Built-in tracing**: LangSmith integration for debugging
- **Tool abstraction**: Easy to define and compose tools
- **Active community**: Lots of examples and documentation

Why compare with our custom agent?
- **Understanding vs convenience**: Custom code shows how agents really work
- **Flexibility vs features**: LangChain has more features but less control
- **Learning vs building**: Students should understand both approaches

For beginners: LangChain is like a "framework" that does a lot of the work for
you. Our custom agent (src/agents.py) shows the same concepts but with explicit
code you can read and understand. Learning both helps you:
1. Understand the fundamentals (custom agent)
2. Use industry tools effectively (LangChain agent)

Used in Week 13 Part 10 for comparison with custom agent.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple, Optional

# Check if langchain is available
try:
    from langchain.tools import tool
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False


# ============================================================
# Data Classes (same as custom agent for compatibility)
# ============================================================
@dataclass
class LangChainTraceStep:
    """Record of a single tool use in LangChain agent.

    For beginners: This is similar to ToolTraceStep in our custom agent.
    LangChain has its own tracing (via callbacks), but we create this
    for easier comparison with our custom implementation.

    Attributes
    ----------
    tool_name : str
        Name of the LangChain tool that was called
    input : Any
        What was passed to the tool
    output : Any
        What the tool returned
    """
    tool_name: str
    input: Any
    output: Any


@dataclass
class LangChainFactCheckResult:
    """Result from LangChain fact-checking agent.

    For beginners: This matches the structure of FactCheckResult
    from our custom agent, so we can compare them easily.
    """
    claim: str
    verdict: str
    evidence: List[Tuple[str, float, str]]
    explanation: str
    trace: List[LangChainTraceStep]
    confidence: float = 0.5


# ============================================================
# LangChain Tools Factory
# ============================================================
def create_langchain_tools(
    retriever_fn: Callable[[str, int], List[Tuple[str, float, str]]],
    tokenizer: Any,
    nli_model: Any,
    top_k: int = 1
):
    """Create LangChain tools from our existing components.

    For beginners: LangChain uses a decorator pattern to define tools.
    The @tool decorator tells LangChain "this function can be called by an agent."

    We wrap our existing retriever and NLI functions as LangChain tools,
    so the agent can use them just like our custom agent does.

    Parameters
    ----------
    retriever_fn : Callable
        Our retrieval function (e.g., tfidf_retrieve wrapper)
    tokenizer : Any
        HuggingFace tokenizer for NLI
    nli_model : Any
        HuggingFace NLI model
    top_k : int
        Number of evidence sentences to retrieve

    Returns
    -------
    tuple
        (tools_list, state_dict) where state_dict stores intermediate results
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError(
            "LangChain is not installed. Install with:\n"
            "  pip install langchain langchain-community"
        )

    # Import NLI functions
    from .nli import nli_predict, aggregate_verdict_from_nli

    # State to store intermediate results (for trace)
    state = {
        "evidence": [],
        "nli_preds": [],
        "verdict": None,
        "trace": []
    }

    @tool
    def retrieve_evidence(claim: str) -> str:
        """Retrieve relevant evidence sentences for a claim.

        This tool searches a Wikipedia corpus to find sentences that might
        be relevant to verifying the claim. Returns top-k evidence sentences.

        Args:
            claim: The claim to find evidence for

        Returns:
            A formatted string with retrieved evidence sentences
        """
        results = retriever_fn(claim, top_k)
        state["evidence"] = results

        # Record in trace
        state["trace"].append(LangChainTraceStep(
            tool_name="retrieve_evidence",
            input={"claim": claim, "top_k": top_k},
            output=results
        ))

        # Format output for LLM
        evidence_text = []
        for i, (key, score, text) in enumerate(results, 1):
            evidence_text.append(f"[E{i}] (score={score:.3f}): {text}")
        return "\n".join(evidence_text) if evidence_text else "No evidence found."

    @tool
    def verify_with_nli(claim: str) -> str:
        """Verify a claim against retrieved evidence using NLI.

        This tool uses a Natural Language Inference model to check if
        the retrieved evidence supports, refutes, or is neutral to the claim.

        Args:
            claim: The claim to verify

        Returns:
            The verdict (SUPPORTS, REFUTES, or NOT ENOUGH INFO)
        """
        if not state["evidence"]:
            return "NOT ENOUGH INFO (no evidence retrieved)"

        # Run NLI on each evidence piece
        nli_preds = []
        for key, score, sent in state["evidence"]:
            pred = nli_predict(
                premise=sent,
                hypothesis=claim,
                tokenizer=tokenizer,
                model=nli_model
            )
            nli_preds.append(pred)

        state["nli_preds"] = nli_preds

        # Aggregate to final verdict
        verdict = aggregate_verdict_from_nli(nli_preds)
        state["verdict"] = verdict

        # Record in trace
        state["trace"].append(LangChainTraceStep(
            tool_name="verify_with_nli",
            input={"claim": claim, "n_evidence": len(state["evidence"])},
            output={"nli_preds": [p.label for p in nli_preds], "verdict": verdict}
        ))

        return f"Verdict: {verdict}"

    @tool
    def explain_verdict(claim: str) -> str:
        """Generate an explanation for the verdict.

        This tool creates a human-readable explanation based on the
        evidence and NLI results.

        Args:
            claim: The original claim

        Returns:
            A natural language explanation
        """
        verdict = state.get("verdict", "NOT ENOUGH INFO")
        evidence = state.get("evidence", [])

        # Generate template-based explanation
        if verdict == "SUPPORTS":
            base = "The claim is SUPPORTED by the evidence."
        elif verdict == "REFUTES":
            base = "The claim is REFUTED by the evidence."
        else:
            base = "There is NOT ENOUGH INFO to verify this claim."

        # Add evidence summary
        if evidence:
            top_evidence = evidence[0][2][:100]
            explanation = f"{base} Key evidence: {top_evidence}..."
        else:
            explanation = base

        # Record in trace
        state["trace"].append(LangChainTraceStep(
            tool_name="explain_verdict",
            input={"claim": claim, "verdict": verdict},
            output=explanation
        ))

        return explanation

    tools = [retrieve_evidence, verify_with_nli, explain_verdict]
    return tools, state


# ============================================================
# LangChain Fact-Check Agent
# ============================================================
class LangChainFactCheckAgent:
    """Fact-checking agent built with LangChain.

    For beginners: This is the same three-stage pipeline as our custom agent:
    Retrieve → Verify → Explain

    But instead of explicit Python code, we use LangChain's:
    - @tool decorator to define tools
    - create_react_agent to build a ReAct-style agent
    - AgentExecutor to run the agent

    The LangChain version is more "declarative" - you tell it what tools
    exist, and it figures out how to use them. Our custom agent is more
    "imperative" - you write the exact steps.

    Comparison:
    - Custom: More explicit, easier to understand, fewer dependencies
    - LangChain: More features, industry standard, built-in tracing

    Attributes
    ----------
    tools : List
        LangChain tools (retrieve, verify, explain)
    agent_executor : AgentExecutor
        LangChain agent that orchestrates tool use
    """

    def __init__(
        self,
        retriever: Callable[[str, int], List[Tuple[str, float, str]]],
        tokenizer: Any,
        nli_model: Any,
        top_k: int = 1
    ):
        """Initialize LangChain fact-checking agent.

        For beginners: Same parameters as our custom FactCheckAgent!
        We're just wrapping them in LangChain's framework.

        Parameters
        ----------
        retriever : Callable
            Function that takes (claim, top_k) and returns evidence
        tokenizer : Any
            HuggingFace tokenizer for NLI
        nli_model : Any
            HuggingFace NLI model
        top_k : int
            Number of evidence sentences to retrieve
        """
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain is not installed. Install with:\n"
                "  pip install langchain langchain-community"
            )

        self.top_k = top_k

        # Create LangChain tools
        self.tools, self._state = create_langchain_tools(
            retriever_fn=retriever,
            tokenizer=tokenizer,
            nli_model=nli_model,
            top_k=top_k
        )

        # For educational purposes, we use a simple scripted approach
        # (Real LangChain apps would use GPT-4/Claude here)
        self._setup_scripted_agent()

    def _setup_scripted_agent(self):
        """Set up a scripted agent for deterministic behavior.

        For beginners: LangChain agents usually use an LLM to decide
        which tools to call. For educational purposes, we use a
        "scripted" approach that always follows the same plan:
        1. retrieve_evidence
        2. verify_with_nli
        3. explain_verdict

        This makes it comparable to our custom agent.
        """
        # We'll use direct tool calls instead of LLM-based agent
        # This keeps behavior predictable for teaching
        self._use_scripted = True

    def run(self, claim: str, use_llm: bool = True) -> LangChainFactCheckResult:
        """Run fact-checking on a claim.

        For beginners: This is the main method - just like agent.run() in
        our custom agent. It orchestrates: Retrieve → Verify → Explain

        Parameters
        ----------
        claim : str
            The claim to fact-check
        use_llm : bool
            Whether to use LLM for explanation (kept for API compatibility)

        Returns
        -------
        LangChainFactCheckResult
            Complete result with verdict, evidence, explanation, and trace
        """
        # Reset state for new claim
        self._state["evidence"] = []
        self._state["nli_preds"] = []
        self._state["verdict"] = None
        self._state["trace"] = []

        # Step 1: Retrieve evidence
        retrieve_tool = self.tools[0]
        retrieve_tool.invoke(claim)

        # Step 2: Verify with NLI
        verify_tool = self.tools[1]
        verify_tool.invoke(claim)

        # Step 3: Generate explanation
        explain_tool = self.tools[2]
        explanation = explain_tool.invoke(claim)

        # Build result
        verdict = self._state.get("verdict", "NOT ENOUGH INFO")
        evidence = self._state.get("evidence", [])
        trace = self._state.get("trace", [])

        # Calculate confidence from NLI predictions
        confidence = 0.5
        if self._state.get("nli_preds"):
            # Use max confidence from predictions
            confidences = [p.probs[p.label] for p in self._state["nli_preds"]]
            confidence = max(confidences) if confidences else 0.5

        return LangChainFactCheckResult(
            claim=claim,
            verdict=verdict,
            evidence=evidence,
            explanation=explanation,
            trace=trace,
            confidence=confidence
        )


# ============================================================
# Comparison Helper
# ============================================================
def compare_agents(
    custom_result,
    langchain_result,
    claim: str
) -> str:
    """Compare results from custom agent and LangChain agent.

    For beginners: This function helps you see how the two implementations
    compare - do they produce the same verdict? Same trace steps?

    Parameters
    ----------
    custom_result : FactCheckResult
        Result from our custom agent (src/agents.py)
    langchain_result : LangChainFactCheckResult
        Result from LangChain agent
    claim : str
        The claim that was verified

    Returns
    -------
    str
        Formatted comparison string
    """
    lines = [
        "=" * 60,
        "AGENT COMPARISON",
        "=" * 60,
        f"Claim: {claim}",
        "",
        "VERDICTS:",
        f"  Custom Agent:    {custom_result.verdict}",
        f"  LangChain Agent: {langchain_result.verdict}",
        f"  Match: {'YES' if custom_result.verdict == langchain_result.verdict else 'NO'}",
        "",
        "TRACE STEPS:",
        "  Custom Agent:",
    ]

    for step in custom_result.trace:
        lines.append(f"    - {step.tool}: {step.output_summary}")

    lines.append("  LangChain Agent:")
    for step in langchain_result.trace:
        output_str = str(step.output)[:50] + "..." if len(str(step.output)) > 50 else str(step.output)
        lines.append(f"    - {step.tool_name}: {output_str}")

    lines.extend([
        "",
        "KEY DIFFERENCES:",
        "  - Custom: Explicit Python code, full control",
        "  - LangChain: Framework-based, more features",
        "=" * 60
    ])

    return "\n".join(lines)
