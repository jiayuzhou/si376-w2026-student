"""LangGraph-based multi-agent fact-checking pipeline.

This module provides an alternative implementation of Week 14's multi-agent
orchestration using LangGraph, an industry framework for building stateful
agent workflows as graphs.

Why LangGraph?
- **Visual workflow**: Pipeline is defined as nodes + edges (a graph)
- **State management**: Shared state flows between agents automatically
- **Conditional routing**: Skeptic's decision becomes a conditional edge
- **Industry standard**: Used in production for complex agent workflows

Comparison with our custom implementation:
- Custom (Week 14): Manual message passing, explicit if/else orchestration
- LangGraph: Declarative graph definition, framework handles routing

For beginners: LangGraph lets you define your multi-agent pipeline as a
flowchart. Each agent is a "node", and the connections between them are
"edges". Conditional edges (like the Skeptic's pass/retry decision) are
handled by the framework instead of manual if/else code.

Used in Week 14 as an optional comparison with custom multi-agent code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

# Check if langgraph is available
try:
    from langgraph.graph import StateGraph, END
    LANGGRAPH_AVAILABLE = True
except ImportError:
    LANGGRAPH_AVAILABLE = False


# ============================================================
# State Definition
# ============================================================
class FactCheckState(TypedDict):
    """Shared state that flows through the graph.

    For beginners: This is the "message board" that all agents read from
    and write to. Each node (agent) can access and update this state.

    Compare to Week 14's manual message passing:
    - Week 14: messages = []; messages.append(Message(...))
    - LangGraph: state["evidence"] = [...] (automatic!)
    """
    claim: str
    evidence: List[Tuple[str, float, str]]
    is_suspicious: bool
    retry_count: int
    verdict: str
    explanation: str
    messages: List[str]  # Trace log


# ============================================================
# Node Functions (one per agent)
# ============================================================
def create_graph_nodes(
    retrieve_fn: Callable,
    tokenizer: Any,
    nli_model: Any,
    min_score: float = 0.15,
    top_k: int = 1
):
    """Create node functions for the LangGraph pipeline.

    For beginners: Each function below is a "node" in the graph.
    Compare to Week 14's agent classes:
      RetrieverAgent.act()  → retrieve_node()
      SkepticAgent.act()    → skeptic_node()
      VerifierAgent.act()   → verify_node()
      ExplainerAgent.act()  → explain_node()

    Parameters
    ----------
    retrieve_fn : Callable
        Retrieval function (BM25 or Embeddings)
    tokenizer, nli_model : Any
        NLI model components
    min_score : float
        Skeptic's threshold for suspicious evidence
    top_k : int
        Number of evidence sentences to retrieve
    """
    from .nli import nli_predict, aggregate_verdict_from_nli

    def retrieve_node(state: FactCheckState) -> dict:
        """Retriever agent: find evidence for the claim."""
        evidence = retrieve_fn(state["claim"], top_k=top_k)
        return {
            "evidence": evidence,
            "messages": state["messages"] + [
                f"[Retriever] Found {len(evidence)} evidence sentences"
            ]
        }

    def skeptic_node(state: FactCheckState) -> dict:
        """Skeptic agent: check if evidence quality is sufficient."""
        if not state["evidence"]:
            is_suspicious = True
            msg = "[Skeptic] No evidence found — suspicious!"
        else:
            top_score = state["evidence"][0][1]
            is_suspicious = top_score < min_score
            if is_suspicious:
                msg = f"[Skeptic] Top score {top_score:.3f} < {min_score} — requesting retry"
            else:
                msg = f"[Skeptic] Top score {top_score:.3f} — evidence looks good"

        return {
            "is_suspicious": is_suspicious,
            "retry_count": state["retry_count"] + (1 if is_suspicious else 0),
            "messages": state["messages"] + [msg]
        }

    def verify_node(state: FactCheckState) -> dict:
        """Verifier agent: use NLI to check claim against evidence."""
        nli_preds = []
        for key, score, sent in state["evidence"]:
            pred = nli_predict(
                premise=sent,
                hypothesis=state["claim"],
                tokenizer=tokenizer,
                model=nli_model
            )
            nli_preds.append(pred)

        verdict = aggregate_verdict_from_nli(nli_preds)
        return {
            "verdict": verdict,
            "messages": state["messages"] + [
                f"[Verifier] Verdict: {verdict}"
            ]
        }

    def explain_node(state: FactCheckState) -> dict:
        """Explainer agent: generate explanation for the verdict."""
        verdict = state["verdict"]
        evidence = state["evidence"]

        if verdict == "SUPPORTS":
            base = "The claim is SUPPORTED by the evidence."
        elif verdict == "REFUTES":
            base = "The claim is REFUTED by the evidence."
        else:
            base = "There is NOT ENOUGH INFO to verify this claim."

        if evidence:
            top_ev = evidence[0][2][:100]
            explanation = f"{base} Key evidence: {top_ev}..."
        else:
            explanation = base

        return {
            "explanation": explanation,
            "messages": state["messages"] + [
                f"[Explainer] {explanation[:60]}..."
            ]
        }

    # Routing function for the Skeptic's conditional edge
    def should_retry(state: FactCheckState) -> str:
        """Decide whether to retry retrieval or proceed to verification.

        For beginners: This is the conditional edge in the graph.
        Compare to Week 14's manual if/else:
          if skeptic.act(evidence): retry
          else: proceed to verifier
        """
        if state["is_suspicious"] and state["retry_count"] <= 1:
            return "retry"
        return "proceed"

    return retrieve_node, skeptic_node, verify_node, explain_node, should_retry


# ============================================================
# Graph Builder
# ============================================================
def build_fact_check_graph(
    retrieve_fn: Callable,
    tokenizer: Any,
    nli_model: Any,
    min_score: float = 0.15,
    top_k: int = 1
):
    """Build a LangGraph fact-checking pipeline.

    For beginners: This function creates the graph (flowchart):

        retrieve → skeptic → [retry?] → verify → explain
                      ↑         |
                      └─── yes ─┘

    Compare to Week 14's manual orchestration:
        evidence = retriever.act(claim)
        if skeptic.act(evidence):
            evidence = retriever.act(claim, top_k=3)
        verdict = verifier.act(claim, evidence)
        explanation = explainer.act(claim, verdict, evidence)

    The LangGraph version does the SAME THING, but as a declarative graph.

    Returns
    -------
    CompiledGraph
        A compiled LangGraph that can be invoked with .invoke()
    """
    if not LANGGRAPH_AVAILABLE:
        raise ImportError(
            "LangGraph is not installed. Install with:\n"
            "  pip install langgraph"
        )

    # Create node functions
    retrieve_node, skeptic_node, verify_node, explain_node, should_retry = \
        create_graph_nodes(retrieve_fn, tokenizer, nli_model, min_score, top_k)

    # Build the graph
    graph = StateGraph(FactCheckState)

    # Add nodes (one per agent)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("skeptic", skeptic_node)
    graph.add_node("verify", verify_node)
    graph.add_node("explain", explain_node)

    # Add edges (connections between agents)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "skeptic")

    # Conditional edge: Skeptic decides whether to retry or proceed
    graph.add_conditional_edges(
        "skeptic",
        should_retry,
        {
            "retry": "retrieve",   # Retry retrieval with more results
            "proceed": "verify"    # Evidence is good enough → verify
        }
    )

    graph.add_edge("verify", "explain")
    graph.add_edge("explain", END)

    # Compile the graph
    return graph.compile()


# ============================================================
# Runner
# ============================================================
def run_langgraph_pipeline(graph, claim: str) -> Dict[str, Any]:
    """Run the LangGraph fact-checking pipeline on a claim.

    Parameters
    ----------
    graph : CompiledGraph
        The compiled LangGraph from build_fact_check_graph()
    claim : str
        The claim to fact-check

    Returns
    -------
    dict
        Final state with verdict, evidence, explanation, and message log
    """
    initial_state = {
        "claim": claim,
        "evidence": [],
        "is_suspicious": False,
        "retry_count": 0,
        "verdict": "NOT ENOUGH INFO",
        "explanation": "",
        "messages": []
    }

    result = graph.invoke(initial_state)
    return result


# ============================================================
# Comparison Helper
# ============================================================
def compare_custom_vs_langgraph(
    custom_verdict: str,
    custom_messages: list,
    langgraph_result: dict,
    claim: str
) -> str:
    """Compare custom multi-agent vs LangGraph multi-agent.

    For beginners: This shows that both approaches produce the same
    result, but with different code styles:
    - Custom: Manual orchestration with Python if/else
    - LangGraph: Declarative graph with nodes and edges
    """
    lg_verdict = langgraph_result["verdict"]
    lg_messages = langgraph_result["messages"]

    lines = [
        "=" * 60,
        "MULTI-AGENT COMPARISON: Custom vs LangGraph",
        "=" * 60,
        f"Claim: {claim[:70]}...",
        "",
        "VERDICTS:",
        f"  Custom (Week 14):  {custom_verdict}",
        f"  LangGraph:         {lg_verdict}",
        f"  Match: {'YES' if custom_verdict == lg_verdict else 'NO'}",
        "",
        "MESSAGE LOG (Custom):",
    ]
    for msg in custom_messages:
        lines.append(f"  {msg}")

    lines.append("")
    lines.append("MESSAGE LOG (LangGraph):")
    for msg in lg_messages:
        lines.append(f"  {msg}")

    lines.extend([
        "",
        "CODE COMPARISON:",
        "  Custom:    ~30 lines of if/else orchestration",
        "  LangGraph: ~10 lines of graph.add_node() + graph.add_edge()",
        "=" * 60
    ])

    return "\n".join(lines)
