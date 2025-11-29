"""Prompt templates for RAG pipeline"""

FEW_SHOT_MEDQA = """### Examples:
Question: A 22-year-old male marathon runner presents with right-sided rib pain. Physical examination reveals exhalation dysfunction at ribs 4-5. Which muscle will be most useful in correcting this dysfunction?
A: anterior scalene B: latissimus dorsi C: pectoralis minor D: quadratus lumborum
Explanation: Pectoralis minor originates from ribs 3-5.
Answer: (C) pectoralis minor

Question: A 36-year-old male has low back pain. Examination shows deep sacral sulcus on left, posterior ILA on right. Most likely diagnosis?
A: right-on-right sacral torsion B: left-on-right sacral torsion C: right unilateral sacral flexion D: left-on-left sacral torsion
Explanation: Deep sulcus left, posterior ILA right suggests right-on-right sacral torsion.
Answer: (A) right-on-right sacral torsion

Question: A 44-year-old man has 3-day sore throat, cough, runny nose, headache worse in morning. Vital signs normal. Erythematous throat. No cervical adenopathy. Lungs clear. Most likely cause?
A: Allergic rhinitis B: Epstein-Barr virus C: Mycoplasma pneumonia D: Rhinovirus
Explanation: Symptoms suggest Rhinovirus. No swollen lymph nodes rules out EBV. Clear lungs rules out Mycoplasma.
Answer: (D) Rhinovirus
"""


def format_question(question: str, use_few_shot: bool = True) -> str:
    """Format question with optional few-shot examples"""
    if use_few_shot:
        return f"{FEW_SHOT_MEDQA}\n\nQuestion: {question}\nAnswer:"
    return f"Question: {question}\nAnswer:"


def format_rag_prompt(question: str, context: str, use_few_shot: bool = True) -> str:
    """Format RAG prompt with context"""
    base = FEW_SHOT_MEDQA + "\n\n" if use_few_shot else ""
    return f"{base}Context: {context}\n\nQuestion: {question}\nAnswer:"


def format_retrieval_decision(question: str) -> str:
    """Format prompt for retrieval decision (Self-BioRAG)"""
    return f"### Instruction:\n{question}\n\n### Response:\n"


def format_critic_prompt(question: str, document: str) -> str:
    """Format prompt for document scoring (Self-BioRAG)"""
    return f"{question}[Retrieval]<paragraph>{document}</paragraph>\n\n### Response:\n"


def clean_answer(text: str) -> str:
    """Remove special tokens from generated text"""
    tokens = [
        "[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]",
        "[Fully supported]", "[Partially supported]", "[No support]",
        "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
        "</s>", "<|endoftext|>", "<pad>"
    ]
    for token in tokens:
        text = text.replace(token, " ")
    return " ".join(text.split()).strip()
