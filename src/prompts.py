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
    print("PROMPT TESTING - Sample Question")
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
    
    print("\n" + "="*100)
    print("Prompt length stats:")
    print(f"  Baseline prompt: {len(baseline_prompt)} characters")
    print(f"  RAG prompt: {len(rag_prompt)} characters")
    print("="*100)
