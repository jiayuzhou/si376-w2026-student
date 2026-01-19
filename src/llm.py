"""LLM-based explanation generation for fact-checking verdicts.

After determining a verdict (SUPPORTS/REFUTES/NOT ENOUGH INFO), we want to
explain WHY to users. This module provides two approaches:

1. **LLM-based (FLAN-T5)**: Use a small language model to generate explanations
   - Pros: Natural, fluent language; can cite evidence; adaptable
   - Cons: Requires model download (~250MB); slower; may hallucinate
   - Used in Week 11 for explanation quality experiments

2. **Template-based**: Simple fill-in-the-blank templates
   - Pros: Fast, deterministic, no hallucinations, no model needed
   - Cons: Rigid, less natural, limited expressiveness
   - Used as fallback or when LLM is unavailable

For beginners: LLMs (Large Language Models) can write human-like text based on
prompts. For fact-checking, we give them a prompt like "Explain why this claim
is supported by this evidence" and they generate a concise explanation.

Why explanations matter:
- Transparency: Users understand how the system reached its verdict
- Trust: Cited evidence makes verdicts more credible
- Debugging: Reveals when the system makes mistakes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


# ============================================================
# Data Classes
# ============================================================
@dataclass
class Explanation:
    """Container for a generated explanation.

    For beginners: This simple object holds an explanation text plus metadata
    about how it was generated (which model/template).

    Attributes
    ----------
    text : str
        The explanation text
        For example: "The evidence supports the claim. For example: [E1] Einstein
        was a theoretical physicist, which confirms he was a scientist."
    model_name : str
        How this explanation was generated
        For example: "google/flan-t5-small" (LLM) or "template" (rule-based)
    """
    text: str         # The generated explanation
    model_name: str   # Model/method used to generate it


# ============================================================
# Prompt Engineering
# ============================================================
def make_explainer_prompt(
    claim: str,
    verdict: str,
    evidence_sentences: List[str]
) -> str:
    """Create a prompt for LLM to generate a fact-checking explanation.

    Prompt engineering is the art of writing instructions that get good outputs
    from LLMs. This prompt is carefully designed to encourage:
    - Grounded explanations (only use provided evidence, don't invent facts)
    - Conciseness (3-5 sentences, not essays)
    - Citations (refer to evidence bullets as [E1], [E2], etc.)

    For beginners: Think of this as writing instructions for a smart assistant.
    Clear, specific instructions → better results. We tell the LLM exactly what
    to do and what not to do.

    Parameters
    ----------
    claim : str
        The claim being verified
    verdict : str
        The verdict: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
    evidence_sentences : List[str]
        Evidence sentences (already formatted with [E1], [E2] prefixes)

    Returns
    -------
    str
        Formatted prompt ready for LLM

    Example
    -------
    >>> prompt = make_explainer_prompt(
    ...     claim="Einstein was a scientist",
    ...     verdict="SUPPORTS",
    ...     evidence_sentences=["[E1] Einstein was a physicist", "[E2] He won Nobel Prize"]
    ... )
    >>> print(prompt)
    You are a careful fact-checking assistant.
    ...
    """
    # Format evidence sentences as bullet list
    # For beginners: "\n".join() combines list items with newlines
    # f"- {s}" adds a bullet point (-) before each sentence
    ev = "\n".join([f"- {s}" for s in evidence_sentences])

    # Construct the prompt with clear instructions
    # For beginners: Triple quotes (""") allow multi-line strings
    # f-strings (f"...{variable}...") insert variables into the text
    prompt = f"""You are a careful fact-checking assistant.

Task:
Explain the verdict for the claim using ONLY the evidence bullets.
- If the evidence is insufficient, say so clearly.
- Do not add new facts.
- Write 3-5 sentences.
- Include short citations like [E1], [E2] referring to the evidence bullets.

Claim: {claim}
Verdict: {verdict}

Evidence bullets:
{ev}

Explanation:"""
    return prompt


# ============================================================
# LLM-based Explanation
# ============================================================
def _extract_prompt_components(prompt: str) -> tuple:
    """Extract claim, verdict, and evidence from a make_explainer_prompt() output.

    Returns (claim, verdict, evidence_text) tuple, or (None, None, None) if parsing fails.
    """
    import re

    claim_match = re.search(r"Claim:\s*(.+?)(?:\n|Verdict)", prompt, re.DOTALL)
    verdict_match = re.search(r"Verdict:\s*(\w+)", prompt)
    evidence_match = re.search(r"Evidence bullets:\s*(.+?)(?:\n\nExplanation|\Z)", prompt, re.DOTALL)

    if not (claim_match and verdict_match):
        return None, None, None

    claim = claim_match.group(1).strip()
    verdict = verdict_match.group(1).strip()

    evidence_text = ""
    if evidence_match:
        evidence_text = evidence_match.group(1).strip()
        # Remove "- [E1]" style prefixes, keep just the text
        evidence_text = re.sub(r"-\s*\[E\d+\]\s*", "", evidence_text)
        evidence_text = evidence_text.replace("\n", " ").strip()

    return claim, verdict, evidence_text


def _generate_small_model_explanation(
    claim: str,
    verdict: str,
    evidence_text: str,
    gen_pipeline
) -> str:
    """Generate explanation using a hybrid template + model approach for small models.

    Small models (flan-t5-small, flan-t5-base) struggle with complex instructions.
    This function uses a hybrid approach:
    1. Start with a template-based prefix explaining the verdict
    2. Use the model to summarize/explain the key evidence

    For beginners: Think of this as giving the model an easier task - instead of
    writing a full explanation, it just needs to summarize what the evidence says.
    """
    # Have the model summarize the evidence (a task small models handle well)
    summary_prompt = f"summarize: {evidence_text}"
    evidence_summary = gen_pipeline(summary_prompt, max_new_tokens=80, do_sample=False)[0]["generated_text"]

    # Build explanation with template prefix + model-generated evidence summary
    if verdict == "SUPPORTS":
        explanation = f"The evidence supports this claim. According to the evidence: {evidence_summary}"
    elif verdict == "REFUTES":
        explanation = f"The evidence contradicts this claim. The evidence shows that {evidence_summary}, which conflicts with the claim that \"{claim}\""
    else:  # NOT ENOUGH INFO
        explanation = f"The evidence is insufficient to verify this claim. The evidence only states that {evidence_summary}, which does not directly address whether \"{claim}\""

    return explanation


def explain_with_flan(
    prompt: str,
    model_name: str = "google/flan-t5-small",
    max_new_tokens: int = 180
) -> Explanation:
    """Generate explanation using FLAN-T5 language model.

    FLAN-T5 is a family of instruction-tuned language models from Google. They're
    trained to follow instructions in prompts, making them good for task-specific
    text generation like explanations.

    For beginners: FLAN-T5-small is a "small" version (~250MB) that runs on CPUs.
    Larger versions (base, large, xl, xxl) are more capable but slower and require
    more memory. The "small" version is fine for classroom use.

    Note: Small models (flan-t5-small, flan-t5-base) have limited instruction-following
    capabilities. For best results, use flan-t5-large or paste the prompt into ChatGPT.
    The function automatically simplifies prompts for small models.

    Why "offline-friendly"?
    - Model downloads once, then cached locally
    - No API calls to external services (like OpenAI)
    - Works without internet after initial download
    - Free and open-source

    Parameters
    ----------
    prompt : str
        Instruction prompt from make_explainer_prompt()
    model_name : str, default="google/flan-t5-small"
        HuggingFace model name. Options:
        - "google/flan-t5-small" (~250MB, fast, limited quality)
        - "google/flan-t5-base" (~900MB, moderate quality)
        - "google/flan-t5-large" (~3GB, good quality)
    max_new_tokens : int, default=180
        Maximum number of tokens (words) to generate
        For beginners: Higher = longer explanations but slower. 180 tokens ≈ 3-5 sentences

    Returns
    -------
    Explanation
        Generated explanation object

    Example
    -------
    >>> prompt = make_explainer_prompt(claim, verdict, evidence)
    >>> explanation = explain_with_flan(prompt)
    >>> print(explanation.text)
    "The evidence supports the claim. According to [E1], Einstein was a physicist..."
    """
    # Import HuggingFace transformers pipeline
    # For beginners: pipeline() is a high-level API that simplifies model usage
    # It handles tokenization, inference, and output formatting automatically
    from transformers import pipeline

    # Create text generation pipeline
    # For beginners: A "pipeline" is a pre-configured tool for a specific task
    # - "text2text-generation": Input text → Output text (seq2seq models like T5)
    # - model=model_name: Which model to use (downloads first time, then cached)
    gen = pipeline("text2text-generation", model=model_name)

    # For small models, use hybrid template + summarization approach
    # For beginners: Small models can't follow complex instructions well,
    # so we use a simpler approach: template prefix + model-generated summary
    if "small" in model_name or "base" in model_name:
        claim, verdict, evidence_text = _extract_prompt_components(prompt)
        if claim and verdict:
            out = _generate_small_model_explanation(claim, verdict, evidence_text, gen)
            return Explanation(text=out, model_name=model_name)
        # Fall through to standard generation if parsing fails

    # Generate explanation using the full prompt (for larger models)
    # For beginners: gen(prompt, ...) runs the model with these settings:
    # - prompt: The input instruction
    # - max_new_tokens: Stop after generating this many tokens
    # - do_sample=False: Use greedy decoding (always pick most likely word, deterministic)
    #   If True, uses sampling (random, more creative but less consistent)
    # [0] extracts first result (pipeline returns a list)
    # ["generated_text"] extracts the text from the result dictionary
    out = gen(prompt, max_new_tokens=max_new_tokens, do_sample=False)[0]["generated_text"]

    # Return as Explanation object
    return Explanation(text=out, model_name=model_name)


# ============================================================
# Template-based Explanation (Fallback)
# ============================================================
def explain_with_template(
    claim: str,
    verdict: str,
    evidence_sentences: List[str]
) -> Explanation:
    """Generate explanation using simple templates (no LLM required).

    Template-based explanations use fill-in-the-blank patterns. They're:
    - Fast (no model loading or inference)
    - Deterministic (same inputs → same output)
    - Honest (never hallucinate or add facts)
    - Simple (easy to understand and debug)

    But they lack the fluency and adaptability of LLM-generated explanations.

    For beginners: Think of this like Mad Libs - we have a template with blanks,
    and we fill in the blanks with the claim, verdict, and evidence.

    Used as fallback when:
    - LLM is unavailable (model download failed, etc.)
    - User explicitly requests template-only mode (use_llm=False)
    - Testing/debugging (deterministic outputs)

    Parameters
    ----------
    claim : str
        The claim being verified
    verdict : str
        The verdict: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
    evidence_sentences : List[str]
        Evidence sentences

    Returns
    -------
    Explanation
        Template-generated explanation

    Example
    -------
    >>> exp = explain_with_template(
    ...     claim="Einstein was a scientist",
    ...     verdict="SUPPORTS",
    ...     evidence_sentences=["Einstein was a physicist"]
    ... )
    >>> print(exp.text)
    'The evidence supports the claim. For example: "Einstein was a physicist"'
    """
    # Handle NOT ENOUGH INFO case separately
    # For beginners: When there's insufficient evidence, we need a different template
    if verdict == "NOT ENOUGH INFO":
        txt = (
            "I cannot verify the claim from the provided evidence. "
            "The evidence does not directly establish whether the claim is true or false."
        )
    else:
        # For SUPPORTS/REFUTES cases
        # Determine the right verb: "supports" or "contradicts"
        # For beginners: We use different wording depending on the verdict
        lead = "supports" if verdict == "SUPPORTS" else "contradicts"

        # Get first evidence sentence as an example (or placeholder if none)
        # For beginners: if/else in one line is called a "ternary operator"
        # It's equivalent to: if evidence_sentences: snippet = evidence_sentences[0] else: snippet = ...
        snippet = evidence_sentences[0] if evidence_sentences else "(no evidence provided)"

        # Format the template
        # For beginners: f-strings insert variables into the text
        txt = f"The evidence {lead} the claim. For example: \"{snippet}\""

    # Return as Explanation object
    # For beginners: model_name="template" indicates this wasn't generated by an LLM
    return Explanation(text=txt, model_name="template")


# ============================================================
# Main Explanation Function
# ============================================================
def explain(
    claim: str,
    verdict: str,
    evidence_sentences: List[str],
    use_llm: bool = True
) -> Explanation:
    """Generate an explanation for a fact-checking verdict.

    This is the main entry point for explanation generation. It tries LLM-based
    explanation first (if use_llm=True), falling back to templates if that fails.

    For beginners: This function handles all the complexity - you just call it
    with a claim, verdict, and evidence, and it returns an explanation.

    Parameters
    ----------
    claim : str
        The claim being verified
    verdict : str
        The verdict: "SUPPORTS" / "REFUTES" / "NOT ENOUGH INFO"
    evidence_sentences : List[str]
        Evidence sentences (without citation prefixes)
        For example: ["Einstein was a physicist", "He won Nobel Prize"]
    use_llm : bool, default=True
        Whether to try LLM-based explanation. If False, uses template immediately.

    Returns
    -------
    Explanation
        Generated explanation (either LLM or template-based)

    Example
    -------
    >>> explanation = explain(
    ...     claim="Einstein was a scientist",
    ...     verdict="SUPPORTS",
    ...     evidence_sentences=["Einstein was a physicist"]
    ... )
    >>> print(explanation.text)
    # LLM-generated explanation with citations
    """
    # Build prompt with evidence citations ([E1], [E2], ...)
    # For beginners: enumerate(list, 1) gives (index, item) pairs starting from index 1
    # f"[E{i}] {s}" formats as "[E1] sentence1", "[E2] sentence2", etc.
    prompt = make_explainer_prompt(
        claim,
        verdict,
        [f"[E{i+1}] {s}" for i, s in enumerate(evidence_sentences)]  # Add [E1], [E2] prefixes
    )

    # If template-only mode, skip LLM
    # For beginners: "not use_llm" means "if use_llm is False"
    if not use_llm:
        return explain_with_template(claim, verdict, evidence_sentences)

    # Try LLM explanation, fall back to template on any error
    # For beginners: try/except handles errors gracefully
    # If explain_with_flan() fails for ANY reason (model download failed,
    # out of memory, etc.), we fall back to the template instead of crashing
    try:
        return explain_with_flan(prompt)
    except Exception:
        # Silently fall back to template
        # For beginners: We don't raise the error - we just use the simpler template instead
        return explain_with_template(claim, verdict, evidence_sentences)
