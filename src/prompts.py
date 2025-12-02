"""Prompt templates for RAG pipeline"""

FEW_SHOT_MEDQA = """You are a medical expert. For each question, select the best answer.
# Format: 
Answer: <LETTER>. <option text>
Explanation: [Brief reasoning]

# Examples:
## Question:
A 35-year-old woman presents with fatigue, weight gain, and cold intolerance. TSH is elevated and free T4 is low. What is the most likely diagnosis?

## Options:
A. Hyperthyroidism
B. Primary hypothyroidism
C. Secondary hypothyroidism
D. Euthyroid sick syndrome

## Response:
Answer: B. Primary hypothyroidism
Explanation: The patient presents with classic symptoms of hypothyroidism (fatigue, weight gain, cold intolerance). The combination of elevated TSH and low free T4 confirms primary hypothyroidism, where the thyroid gland itself fails to produce adequate hormone, triggering compensatory TSH elevation from the pituitary.

## Question:
A 2-year-old boy has recurrent bacterial infections. Labs show low IgG, IgA, and IgM. B cells are absent. What is the diagnosis?

## Options:
A. DiGeorge syndrome
B. Bruton agammaglobulinemia
C. Selective IgA deficiency
D. Chronic granulomatous disease

## Response:
Answer: B. Bruton agammaglobulinemia
Explanation: The complete absence of B cells with low levels of all immunoglobulin classes (IgG, IgA, IgM) is pathognomonic for Bruton agammaglobulinemia, an X-linked agammaglobulinemia caused by defective B cell maturation. This leads to recurrent bacterial infections in early childhood.

## Question:
A 60-year-old smoker has a chronic cough and weight loss. Chest X-ray shows a central mass. Biopsy shows small blue cells. What is the most likely diagnosis?

## Options:
A. Adenocarcinoma
B. Squamous cell carcinoma
C. Small cell lung cancer
D. Large cell carcinoma

## Response:
Answer: C. Small cell lung cancer
Explanation: The key diagnostic features are the central location, smoking history, and histology showing "small blue cells" (small round cells with scant cytoplasm and hyperchromatic nuclei). Small cell lung cancer is strongly associated with smoking, typically presents centrally, and has characteristic neuroendocrine histology.
"""


def format_question_with_options(item: dict) -> str:
    """Extract question and format with options from dataset item.
    
    Args:
        item: Dataset item with 'question' and optionally 'options' fields
        
    Returns:
        Formatted question string with options
    """
    question = item["question"]
    if "options" in item:
        options_text = "\n".join([f"{k}. {v}" for k, v in sorted(item["options"].items())])
        return f"{question}\n\n## Options:\n{options_text}"
    return question


def format_question(question: str, use_few_shot: bool = True) -> str:
    """Format question with optional few-shot examples for MCQ answering.

    Expectation: model should output in the form:
    <LETTER>. <option text>
    Explanation: [Brief reasoning]
    e.g., 'B. Tell the attending that he cannot fail to disclose this mistake'
    """
    if use_few_shot:
        return f"{FEW_SHOT_MEDQA}\n\n## Question:\n{question}\n\n## Response:\n"
    else:
        instruction = (
            "Select the best answer from the options below.\n"
            "## Format:\nAnswer: <LETTER>. <option text>\nExplanation: [Brief reasoning]\n\n"
        )
        return f"{instruction}## Question:\n{question}\n\n## Response:\n"


def format_rag_prompt(question: str, context: str, use_few_shot: bool = True) -> str:
    """Format RAG prompt with retrieved context.

    Expectation: model should output in the form:
    <LETTER>. <option text>
    """
    if use_few_shot:
        return (
            f"{FEW_SHOT_MEDQA}\n\n"
            f"## Context:\n{context}\n\n"
            f"## Question:\n{question}\n\n"
            f"## Response:\n"
        )
    else:
        instruction = (
            "Use the provided context to answer the question.\n"
            "## Format:\nAnswer: <LETTER>. <option text>\nExplanation: [Brief reasoning]\n\n"
        )
        return (
            f"{instruction}"
            f"## Context:\n{context}\n\n"
            f"## Question:\n{question}\n\n"
            f"## Response:\n"
        )


def format_retrieval_decision(question: str) -> str:
    """Format prompt for retrieval decision (Self-BioRAG style)."""
    return (
        "## Instruction:\n"
        "Decide whether external document retrieval is needed to answer the question.\n"
        "Respond with one of: [Retrieval] or [No Retrieval].\n\n"
        f"## Question:\n{question}\n\n"
        "## Response:\n"
    )


# ============================================================================
# DMQR-RAG: Multi-Query Rewriting Prompts (3-Strategy)
# Based on: https://arxiv.org/pdf/2411.13154
# Strategies: GQR (General), KWR (Keyword), PAR (Pseudo-Answer)
# ============================================================================

def format_query_rewrite_gqr(question: str) -> str:
    """
    GQR - General Query Rewrite (Few-shot)
    Clean, well-formed search query version.
    """
    return f"""Rewrite the medical question as a clean search query.

Example 1:
Question: A 35-year-old woman presents with fatigue, weight gain, and cold intolerance. TSH is elevated and free T4 is low. What is the most likely diagnosis?
Search query: fatigue weight gain cold intolerance elevated TSH low T4 diagnosis

Example 2:
Question: A 2-year-old boy has recurrent bacterial infections. Labs show low IgG, IgA, and IgM. B cells are absent. What is the diagnosis?
Search query: recurrent bacterial infections child low immunoglobulins absent B cells diagnosis

Question: {question}
Search query:"""


def format_query_rewrite_kwr(question: str) -> str:
    """
    KWR - Keyword Rewrite (Few-shot)
    Extract key medical terms as keywords.
    """
    return f"""Extract medical keywords from the question.

Example 1:
Question: A 35-year-old woman presents with fatigue, weight gain, and cold intolerance. TSH is elevated and free T4 is low. What is the most likely diagnosis?
Keywords: fatigue weight gain cold intolerance TSH elevated T4 low hypothyroidism

Example 2:
Question: A 2-year-old boy has recurrent bacterial infections. Labs show low IgG, IgA, and IgM. B cells are absent. What is the diagnosis?
Keywords: recurrent infections child immunodeficiency IgG IgA IgM B cells agammaglobulinemia

Question: {question}
Keywords:"""


def format_query_rewrite_par(question: str) -> str:
    """
    PAR - Pseudo-Answer Rewrite (Few-shot)
    Include likely diagnosis/answer terms in search query.
    """
    return f"""Create search query including the likely answer/diagnosis.

Example 1:
Question: A 35-year-old woman presents with fatigue, weight gain, and cold intolerance. TSH is elevated and free T4 is low. What is the most likely diagnosis?
Search query: primary hypothyroidism elevated TSH low T4 fatigue weight gain cold intolerance Hashimoto

Example 2:
Question: A 2-year-old boy has recurrent bacterial infections. Labs show low IgG, IgA, and IgM. B cells are absent. What is the diagnosis?
Search query: Bruton agammaglobulinemia X-linked absent B cells low immunoglobulins recurrent infections

Question: {question}
Search query:"""


# Mapping for easy access
QUERY_REWRITE_STRATEGIES = {
    "gqr": ("General Query Rewrite", format_query_rewrite_gqr),
    "kwr": ("Keyword Rewrite", format_query_rewrite_kwr),
    "par": ("Pseudo-Answer Rewrite", format_query_rewrite_par),
}


def format_critic_prompt(question: str, document: str, title: str = "") -> str:
    """
    Format prompt for document scoring (Self-BioRAG critic reranking).
    
    The model generates reflection tokens to score the document:
    - [Relevant] or [Irrelevant]
    - [Fully supported], [Partially supported], or [No support]
    - [Utility:1-5]
    
    Args:
        question: The question to answer
        document: The document content
        title: Optional document title
        
    Returns:
        Formatted prompt string
    """
    # Format document with title if provided
    if title:
        paragraph = f"{title}\n{document}"
    else:
        paragraph = document
    
    return (
        f"## Instruction:\n{question}\n\n"
        f"## Response:\n"
        f"[Retrieval]<paragraph>{paragraph}</paragraph>\n\n"
    )


def clean_answer(text: str) -> str:
    """
    Remove control tokens / special markers from generated text.

    Note: This does NOT parse MCQ answers; it just strips tags like [Retrieval].
    Downstream evaluation for MCQs should still inspect the cleaned string
    and extract the first occurrence of A/B/C/D as needed.
    """
    tokens = [
        "[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]",
        "[Fully supported]", "[Partially supported]", "[No support]",
        "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
        "</s>", "<|endoftext|>", "<pad>"
    ]
    for token in tokens:
        text = text.replace(token, " ")
    return " ".join(text.split()).strip()


if __name__ == "__main__":
    """Test prompts with sample questions"""
    print("="*100)
    print("PROMPT TESTING - All Prompt Types")
    print("="*100)
    
    # Sample question
    sample_item = {
        "question": "A 45-year-old man with a history of chronic alcohol use presents with confusion and ataxia. Physical exam shows nystagmus and ophthalmoplegia. What vitamin deficiency is most likely?",
        "options": {
            "A": "Vitamin B1 (Thiamine)",
            "B": "Vitamin B12 (Cobalamin)",
            "C": "Vitamin C (Ascorbic acid)",
            "D": "Vitamin D (Cholecalciferol)"
        },
        "answer_idx": "A",
        "answer_text": "Vitamin B1 (Thiamine)"
    }
    
    # Format question with options
    formatted_question = format_question_with_options(sample_item)
    
    print("\n1. FORMATTED QUESTION (what main.py creates):")
    print("-"*100)
    print(formatted_question)
    
    # Test baseline prompt (no context)
    print("\n\n2. BASELINE PROMPT (no RAG, with few-shot):")
    print("-"*100)
    baseline_prompt = format_question(formatted_question, use_few_shot=True)
    print(baseline_prompt)
    
    # Test RAG prompt
    sample_context = "Wernicke encephalopathy is caused by thiamine (vitamin B1) deficiency. Classic triad includes confusion, ataxia, and ophthalmoplegia. It is common in chronic alcoholics due to poor nutrition and impaired thiamine absorption."
    
    print("\n\n3. RAG PROMPT (with context, with few-shot):")
    print("-"*100)
    rag_prompt = format_rag_prompt(formatted_question, sample_context, use_few_shot=True)
    print(rag_prompt)
    
    print("\n\n4. EXPECTED MODEL OUTPUT:")
    print("-"*100)
    print("Answer: A. Vitamin B1 (Thiamine)")
    print("\n(The evaluator looks for 'Answer:' prefix and extracts the letter)")
    
    # ========================================================================
    # DMQR-RAG: Multi-Query Prompts (3-Strategy)
    # ========================================================================
    print("\n\n" + "="*100)
    print("DMQR-RAG: 3-Strategy Query Rewriting")
    print("="*100)
    
    # Medical question for testing
    medical_question = "A 45-year-old man with chronic alcohol use presents with confusion, ataxia, and ophthalmoplegia. What vitamin deficiency is most likely?"
    
    print("\n5. GQR - General Query Rewrite:")
    print("-"*100)
    gqr_prompt = format_query_rewrite_gqr(medical_question)
    print(gqr_prompt)
    print("\nExpected: What vitamin deficiency causes confusion, ataxia, and ophthalmoplegia in chronic alcohol users?")
    
    print("\n\n6. KWR - Keyword Rewrite:")
    print("-"*100)
    kwr_prompt = format_query_rewrite_kwr(medical_question)
    print(kwr_prompt)
    print("\nExpected: vitamin deficiency confusion ataxia ophthalmoplegia chronic alcohol")
    
    print("\n\n7. PAR - Pseudo-Answer Rewrite:")
    print("-"*100)
    par_prompt = format_query_rewrite_par(medical_question)
    print(par_prompt)
    print("\nExpected: thiamine B1 deficiency Wernicke encephalopathy confusion ataxia ophthalmoplegia")
    
    # ========================================================================
    # Self-BioRAG Prompts
    # ========================================================================
    print("\n\n" + "="*100)
    print("Self-BioRAG: Retrieval Decision & Critic Prompts")
    print("="*100)
    
    print("\n8. RETRIEVAL DECISION PROMPT:")
    print("-"*100)
    retrieval_prompt = format_retrieval_decision(formatted_question)
    print(retrieval_prompt)
    print("\nExpected output: [Retrieval] or [No Retrieval]")
    
    print("\n\n9. CRITIC RERANKING PROMPT:")
    print("-"*100)
    critic_prompt = format_critic_prompt(formatted_question, sample_context, "Wernicke Encephalopathy")
    print(critic_prompt)
    print("\nExpected output: [Relevant], [Fully supported], [Utility:5]")
    
    # ========================================================================
    # Summary
    # ========================================================================
    print("\n\n" + "="*100)
    print("PROMPT LENGTH SUMMARY")
    print("="*100)
    print(f"  Baseline prompt:      {len(baseline_prompt):,} chars")
    print(f"  RAG prompt:           {len(rag_prompt):,} chars")
    print(f"  GQR (General):        {len(gqr_prompt):,} chars")
    print(f"  KWR (Keyword):        {len(kwr_prompt):,} chars")
    print(f"  PAR (Pseudo-Answer):  {len(par_prompt):,} chars")
    print(f"  Retrieval decision:   {len(retrieval_prompt):,} chars")
    print(f"  Critic reranking:     {len(critic_prompt):,} chars")
    print("="*100)
