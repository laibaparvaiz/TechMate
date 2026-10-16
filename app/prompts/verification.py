"""
File: TechMate/app/prompts/verification.py
Description: Prompts for Self-RAG relevance and hallucination grading.
"""

SELF_RAG_HALLUCINATION_PROMPT = """You are a careful evidence checker evaluating an LLM response against the supplied study context.

Context Passages:
{context}

Generated Response:
{response}

Evaluate factual claims in the response using these rules:

1. Judge whether the substantive answer is supported by the context, not whether it repeats the source word for word.
2. Accept faithful paraphrases, concise explanations, headings, and examples when their meaning follows from the supplied context.
3. Mark a response as not grounded only when it contains a material factual claim that is missing from, contradicted by, or cannot reasonably be inferred from the context, or when the context is insufficient to answer the question.
4. Do not treat formatting, organization, or non-factual conversational wording as hallucinations.
5. Set hallucination_detected to true only when the response makes at least one material unsupported or contradicted factual claim.
6. Set is_grounded to true when the answer's material factual content is supported and hallucination_detected is false. Do not reject a supported answer merely because it is paraphrased.
7. In notes, briefly identify the specific unsupported claim when rejecting an answer. If accepting it, briefly state that its substantive claims are supported.

Respond only with a JSON object matching this schema:
{{ "is_grounded": true or false, "hallucination_detected": true or false, "notes": "Brief evidence-based feedback" }}"""
