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

LLM-Based Agent Architecture (Week 13):

The **LLMFactCheckAgent** uses a language model (FLAN-T5) as its "brain" to decide
which tool to call next. Instead of hardcoded if/else logic, the LLM reads the
current state and outputs a decision. This is the closest to a "real" agent:
- Scripted agent: fixed pipeline (no decisions)
- Adaptive agent: rule-based decisions (threshold checks)
- LLM agent: model-based decisions (LLM reads state and chooses)

Used in Weeks 13-14 for single-agent and multi-agent fact-checking systems.

Adaptive Agent Architecture (Week 13):

The **AdaptiveFactCheckAgent** extends the scripted agent with genuine decision-making:
1. **Retrieval Quality Check**: If top retrieval score is too low, tries a fallback retriever
2. **NLI Confidence Check**: If all NLI predictions are neutral/low-confidence, abstains
3. **Conflict Detection**: If evidence both supports AND contradicts, flags uncertainty

These decisions appear as "decide" steps in the tool trace, making the ReAct loop
tangible — the agent truly reasons about its observations before proceeding.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, List, Tuple

from .llm import explain
from .nli import NLIPrediction, aggregate_verdict_from_nli, nli_predict


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


# ============================================================
# Adaptive Fact-Checking Agent
# ============================================================
class AdaptiveFactCheckAgent(FactCheckAgent):
    """A fact-checking agent that adapts based on observations.

    Unlike FactCheckAgent (which always follows retrieve -> verify -> explain),
    this agent makes genuine decisions at runtime:

    1. **Retrieval Quality Check + Fallback**: After retrieving evidence, checks if
       the top retrieval score is below a threshold. If so, tries a different
       retriever (e.g., TF-IDF fails -> try BM25).

    2. **NLI Confidence Check + Abstention**: After NLI verification, if all
       predictions are neutral/low-confidence, the agent abstains rather than
       making a guess.

    3. **Conflict Detection**: If NLI returns both ENTAILMENT and CONTRADICTION
       across evidence pieces, flags the conflict in the trace.

    This is a real ReAct agent: the trace includes "decide" steps that show the
    agent's reasoning, not just tool calls.

    For beginners: The scripted agent (FactCheckAgent) is like following a recipe
    step by step no matter what. The adaptive agent is like a chef who tastes the
    food at each step and adjusts — if the evidence is bad, try another source;
    if nothing is clear, say "I'm not sure" instead of guessing.

    Attributes
    ----------
    retrievers : List[Tuple[str, Callable]]
        List of (name, retriever_fn) tuples to try in order.
        For example: [("tfidf", tfidf_fn), ("bm25", bm25_fn)]
    retrieval_threshold : float
        Minimum top retrieval score to proceed without fallback.
    abstention_threshold : float
        Maximum NLI confidence below which the agent abstains.
    """

    def __init__(
        self,
        retrievers: List[Tuple[str, Callable[[str, int], List[Tuple[str, float, str]]]]],
        tokenizer: Any,
        nli_model: Any,
        retrieval_threshold: float = 0.15,
        abstention_threshold: float = 0.6
    ):
        """Initialize adaptive fact-checking agent with multiple retrievers.

        Parameters
        ----------
        retrievers : List[Tuple[str, Callable]]
            Ordered list of (name, retriever_function) tuples.
            The agent tries each in order until retrieval quality is acceptable.
            For example: [("tfidf", tfidf_fn), ("bm25", bm25_fn)]
        tokenizer : Any
            HuggingFace tokenizer for NLI model
        nli_model : Any
            HuggingFace NLI model
        retrieval_threshold : float, default=0.15
            If top retrieval score is below this, try the next retriever.
        abstention_threshold : float, default=0.6
            If max NLI confidence is below this AND all predictions are neutral,
            the agent abstains instead of generating an explanation.
        """
        # Initialize parent with the first retriever
        super().__init__(retrievers[0][1], tokenizer, nli_model)
        self.retrievers = retrievers
        self.retrieval_threshold = retrieval_threshold
        self.abstention_threshold = abstention_threshold

    def run(
        self,
        claim: str,
        top_k: int = 5,
        use_llm: bool = True
    ) -> FactCheckResult:
        """Run the adaptive fact-checking pipeline on a claim.

        Unlike the scripted agent, this method makes decisions based on
        observations at each stage. The trace includes "decide" steps
        that show the agent's reasoning.

        Pipeline with decision points:
        1. **Retrieve** with quality check (may retry with fallback)
        2. **Verify** with NLI
        3. **Decide**: Check for abstention or conflict
        4. **Explain** (only if confident enough)

        Parameters
        ----------
        claim : str
            Claim to verify
        top_k : int, default=5
            How many evidence sentences to retrieve
        use_llm : bool, default=True
            Whether to use LLM for explanation

        Returns
        -------
        FactCheckResult
            Complete result including verdict, evidence, explanation, and trace.
            Trace includes "decide" steps showing agent reasoning.
        """
        trace: List[ToolTraceStep] = []

        # ====== STAGE 1: Retrieve with Quality Check + Fallback ======
        retrieved = []
        retriever_used = self.retrievers[0][0]

        for i, (name, retriever_fn) in enumerate(self.retrievers):
            retrieved = retriever_fn(claim, top_k=top_k)
            top_score = retrieved[0][1] if retrieved else 0.0

            trace.append(ToolTraceStep(
                tool="retrieve",
                input=f"claim={claim} top_k={top_k} retriever={name}",
                output_summary=f"retrieved {len(retrieved)} sentences, top_score={top_score:.3f}"
            ))

            if top_score >= self.retrieval_threshold:
                retriever_used = name
                break  # Good enough — proceed
            elif i + 1 < len(self.retrievers):
                next_name = self.retrievers[i + 1][0]
                trace.append(ToolTraceStep(
                    tool="decide",
                    input=f"top_score={top_score:.3f}, threshold={self.retrieval_threshold}",
                    output_summary=f"RETRY: score {top_score:.3f} < {self.retrieval_threshold}, trying {next_name}"
                ))
            else:
                # Last retriever — proceed with what we have
                retriever_used = name
                trace.append(ToolTraceStep(
                    tool="decide",
                    input=f"top_score={top_score:.3f}, threshold={self.retrieval_threshold}",
                    output_summary=f"PROCEED: score {top_score:.3f} < {self.retrieval_threshold}, no more retrievers"
                ))

        # ====== STAGE 2: Verify with NLI ======
        nli_preds: List[NLIPrediction] = []
        for key, score, sent in retrieved:
            p = nli_predict(
                premise=sent,
                hypothesis=claim,
                tokenizer=self.tokenizer,
                model=self.nli_model
            )
            nli_preds.append(p)

        verdict = aggregate_verdict_from_nli(nli_preds)

        trace.append(ToolTraceStep(
            tool="verify_nli",
            input=f"{len(retrieved)} candidates",
            output_summary=f"verdict={verdict}"
        ))

        # ====== STAGE 2.5: Decision — Check for Abstention and Conflict ======

        # Conflict detection: both ENTAILMENT and CONTRADICTION present
        has_entail = any(
            p.probs.get("ENTAILMENT", 0.0) >= 0.5 for p in nli_preds
        )
        has_contra = any(
            p.probs.get("CONTRADICTION", 0.0) >= 0.5 for p in nli_preds
        )

        if has_entail and has_contra:
            n_entail = sum(1 for p in nli_preds if p.probs.get("ENTAILMENT", 0.0) >= 0.5)
            n_contra = sum(1 for p in nli_preds if p.probs.get("CONTRADICTION", 0.0) >= 0.5)
            trace.append(ToolTraceStep(
                tool="decide",
                input=f"entailments={n_entail}, contradictions={n_contra}",
                output_summary=f"CONFLICT: {n_entail} entail, {n_contra} contradict"
            ))

        # Abstention check: all neutral AND low confidence
        all_neutral = all(p.label in ("NEUTRAL", "NOT ENOUGH INFO") for p in nli_preds)
        max_conf = max(
            (max(p.probs.get("ENTAILMENT", 0.0), p.probs.get("CONTRADICTION", 0.0))
             for p in nli_preds),
            default=0.0
        )

        if all_neutral and max_conf < self.abstention_threshold:
            trace.append(ToolTraceStep(
                tool="decide",
                input=f"all_neutral={all_neutral}, max_conf={max_conf:.3f}, threshold={self.abstention_threshold}",
                output_summary=f"ABSTAIN: all NLI neutral, max_conf {max_conf:.3f} < {self.abstention_threshold}"
            ))
            return FactCheckResult(
                claim=claim,
                verdict="NOT ENOUGH INFO",
                evidence=retrieved,
                explanation="Agent abstained: evidence not relevant or confident enough to make a determination.",
                trace=trace
            )

        # ====== STAGE 3: Explain (only reached if confident enough) ======
        evidence_texts = [t for (_, _, t) in retrieved[:3]]
        expl = explain(claim, verdict, evidence_texts, use_llm=use_llm)

        trace.append(ToolTraceStep(
            tool="explain",
            input="top3 evidence",
            output_summary=f"model={expl.model_name}"
        ))

        return FactCheckResult(
            claim=claim,
            verdict=verdict,
            evidence=retrieved,
            explanation=expl.text,
            trace=trace
        )


# ============================================================
# LLM-Based Fact-Checking Agent
# ============================================================
class LLMFactCheckAgent(FactCheckAgent):
    """A fact-checking agent that uses an LLM to decide what to do next.

    Unlike the scripted agent (fixed pipeline) or adaptive agent (rule-based
    decisions), this agent uses FLAN-T5 as its "brain" — the LLM reads the
    current situation and decides which tool to call.

    This is the closest to a "real" agent:
    - Scripted: always retrieve → verify → explain (no decisions)
    - Adaptive: if score < threshold → retry (hardcoded rules)
    - LLM-based: "Given this evidence quality, what should I do?" (model decides)

    For beginners: Think of the three agents as:
    - Scripted = following a recipe exactly
    - Adaptive = following a recipe but checking food temperature
    - LLM-based = a chef who reads the situation and improvises

    The LLM agent sometimes makes brilliant decisions and sometimes makes
    mistakes — that's the key teaching moment! Real LLM agents (like Claude
    Code) use much larger models, but the pattern is identical.

    Attributes
    ----------
    planner : pipeline
        FLAN-T5 text generation pipeline used for planning decisions
    max_steps : int
        Maximum number of steps to prevent infinite loops
    """

    def __init__(
        self,
        retriever: Callable[[str, int], List[Tuple[str, float, str]]],
        tokenizer: Any,
        nli_model: Any,
        planner_model: str = "google/flan-t5-small",
        max_steps: int = 8
    ):
        """Initialize LLM-based fact-checking agent.

        Parameters
        ----------
        retriever : Callable
            Retrieval function
        tokenizer, nli_model : Any
            NLI model components
        planner_model : str
            HuggingFace model for planning decisions (default: flan-t5-small)
        max_steps : int
            Maximum steps to prevent infinite loops
        """
        super().__init__(retriever, tokenizer, nli_model)
        self.max_steps = max_steps

        # Load the planner LLM (seq2seq model like FLAN-T5)
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer as AutoTok
        self.planner_tokenizer = AutoTok.from_pretrained(planner_model)
        self.planner_model_obj = AutoModelForSeq2SeqLM.from_pretrained(planner_model)
        self.planner_model = planner_model

    def _ask_planner(self, prompt: str) -> str:
        """Ask the LLM planner what to do next.

        Parameters
        ----------
        prompt : str
            Description of current state and available actions

        Returns
        -------
        str
            The LLM's decision (parsed to extract action)
        """
        import torch
        inputs = self.planner_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.planner_model_obj.generate(**inputs, max_new_tokens=30, do_sample=False)
        result = self.planner_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return result.strip()

    def run(
        self,
        claim: str,
        top_k: int = 1,
        use_llm: bool = True
    ) -> FactCheckResult:
        """Run the LLM-based agent on a claim.

        The agent follows a ReAct loop:
        1. Ask LLM: "What should I do next?"
        2. Execute the chosen tool
        3. Observe the result
        4. Repeat until LLM says "done" or max_steps reached

        Parameters
        ----------
        claim : str
            Claim to fact-check
        top_k : int
            Number of evidence sentences to retrieve
        use_llm : bool
            Whether to use LLM for final explanation

        Returns
        -------
        FactCheckResult
            Complete result with verdict, evidence, explanation, and trace
        """
        trace: List[ToolTraceStep] = []
        retrieved = []
        nli_preds = []
        verdict = None

        for step_num in range(self.max_steps):
            # Build the planning prompt based on current state
            prompt = self._build_planning_prompt(claim, step_num, retrieved, nli_preds, verdict)

            # Ask the LLM what to do
            decision = self._ask_planner(prompt)

            # Record the planning step
            trace.append(ToolTraceStep(
                tool="plan",
                input=f"step={step_num}, state={'has evidence' if retrieved else 'no evidence'}, verdict={verdict}",
                output_summary=f"LLM decided: {decision[:60]}"
            ))

            # Parse and execute the decision
            action = self._parse_action(decision, retrieved, verdict)

            if action == "retrieve":
                retrieved = self.retriever(claim, top_k=top_k)
                trace.append(ToolTraceStep(
                    tool="retrieve",
                    input=f"claim={claim[:40]}... top_k={top_k}",
                    output_summary=f"retrieved {len(retrieved)} sentences"
                ))

            elif action == "verify":
                if not retrieved:
                    # LLM tried to verify without evidence — record the mistake
                    trace.append(ToolTraceStep(
                        tool="decide",
                        input="LLM chose verify but no evidence available",
                        output_summary="ERROR: no evidence to verify — will retrieve first"
                    ))
                    retrieved = self.retriever(claim, top_k=top_k)
                    trace.append(ToolTraceStep(
                        tool="retrieve",
                        input=f"claim={claim[:40]}... top_k={top_k}",
                        output_summary=f"retrieved {len(retrieved)} sentences (recovery)"
                    ))

                nli_preds = []
                for key, score, sent in retrieved:
                    p = nli_predict(
                        premise=sent,
                        hypothesis=claim,
                        tokenizer=self.tokenizer,
                        model=self.nli_model
                    )
                    nli_preds.append(p)
                verdict = aggregate_verdict_from_nli(nli_preds)
                trace.append(ToolTraceStep(
                    tool="verify_nli",
                    input=f"{len(retrieved)} candidates",
                    output_summary=f"verdict={verdict}"
                ))

            elif action == "explain":
                evidence_texts = [t for (_, _, t) in retrieved[:3]]
                expl = explain(claim, verdict or "NOT ENOUGH INFO", evidence_texts, use_llm=use_llm)
                trace.append(ToolTraceStep(
                    tool="explain",
                    input="top3 evidence",
                    output_summary=f"model={expl.model_name}"
                ))
                # Done!
                return FactCheckResult(
                    claim=claim,
                    verdict=verdict or "NOT ENOUGH INFO",
                    evidence=retrieved,
                    explanation=expl.text,
                    trace=trace
                )

            elif action == "done" or action == "abstain":
                trace.append(ToolTraceStep(
                    tool="decide",
                    input=f"LLM said: {decision[:40]}",
                    output_summary=f"LLM chose to stop: {action}"
                ))
                break

        # If we hit max_steps or broke out, generate final result
        if verdict is None and retrieved:
            nli_preds = []
            for key, score, sent in retrieved:
                p = nli_predict(premise=sent, hypothesis=claim,
                                tokenizer=self.tokenizer, model=self.nli_model)
                nli_preds.append(p)
            verdict = aggregate_verdict_from_nli(nli_preds)

        evidence_texts = [t for (_, _, t) in retrieved[:3]] if retrieved else []
        expl = explain(claim, verdict or "NOT ENOUGH INFO", evidence_texts, use_llm=use_llm)

        return FactCheckResult(
            claim=claim,
            verdict=verdict or "NOT ENOUGH INFO",
            evidence=retrieved,
            explanation=expl.text,
            trace=trace
        )

    def _build_planning_prompt(self, claim, step_num, retrieved, nli_preds, verdict):
        """Build a prompt for the LLM planner.

        For beginners: This is the "Reason" step of ReAct. We describe the
        current state to the LLM and ask it what to do next.
        """
        if step_num == 0:
            return (
                f"You are a fact-checking agent. You need to verify this claim: \"{claim}\"\n"
                f"Available tools: retrieve (find evidence), verify (check with NLI), explain (write explanation).\n"
                f"What should you do first? Answer with one word: retrieve, verify, or explain."
            )

        if retrieved and verdict is None:
            return (
                f"I have evidence for the claim \"{claim[:50]}\". "
                f"I have not verified it yet. "
                f"Should I verify or explain? Answer with one word."
            )

        if retrieved and verdict is not None:
            return (
                f"I verified the claim \"{claim[:50]}\" and the verdict is {verdict}. "
                f"The next step is to explain the verdict. "
                f"What should I do? Answer with one word: explain or done."
            )

    def _parse_action(self, decision, retrieved, verdict):
        """Parse the LLM's decision into an action.

        For beginners: The LLM outputs free text, so we need to extract
        the action. Small models sometimes output unexpected text —
        that's why we have fallback logic.
        """
        decision_lower = decision.lower().strip()

        # Direct match
        for action in ["retrieve", "verify", "explain", "done", "abstain"]:
            if action in decision_lower:
                return action

        # Fallback: if LLM output is unclear, use heuristics
        if not retrieved:
            return "retrieve"
        elif verdict is None:
            return "verify"
        else:
            return "explain"
