"""Prompt templates for RAG pipeline"""

FEW_SHOT_MEDQA = """You are a medical expert answering multiple-choice questions.
For each question, choose the single best answer and respond in the format:
<LETTER>. <option text>

### Examples:

Question:
A 22-year-old male marathon runner presents with right-sided rib pain. Physical examination reveals
exhalation dysfunction at ribs 4-5. Which muscle will be most useful in correcting this dysfunction?

Options:
A. anterior scalene
B. latissimus dorsi
C. pectoralis minor
D. quadratus lumborum

Explanation:
Pectoralis minor originates from ribs 3-5.

Answer: C. pectoralis minor


Question:
A 36-year-old male has low back pain. Examination shows deep sacral sulcus on left,
posterior ILA on right. Most likely diagnosis?

Options:
A. right-on-right sacral torsion
B. left-on-right sacral torsion
C. right unilateral sacral flexion
D. left-on-left sacral torsion

Explanation:
Deep sulcus left, posterior ILA right suggests right-on-right sacral torsion.

Answer: A. right-on-right sacral torsion


Question:
A 44-year-old man has a 3-day history of sore throat, cough, runny nose,
and headache worse in the morning. Vital signs are normal. The throat is erythematous.
There is no cervical adenopathy. Lungs are clear to auscultation. What is the most likely cause?

Options:
A. Allergic rhinitis
B. Epstein-Barr virus
C. Mycoplasma pneumonia
D. Rhinovirus

Explanation:
Symptoms suggest Rhinovirus. No swollen lymph nodes rules out EBV. Clear lungs rules out Mycoplasma.

Answer: D. Rhinovirus
"""


def format_question(question: str, use_few_shot: bool = True) -> str:
    """Format question with optional few-shot examples for MCQ answering.

    Expectation: model should output in the form:
    <LETTER>. <option text>
    e.g., 'B. Tell the attending that he cannot fail to disclose this mistake'
    """
    instruction = (
        "Instructions:\n"
        "- Choose the single best option.\n"
        "- Respond in the format: <LETTER>. <option text>\n"
        "- For example: C. pectoralis minor\n"
    )

    if use_few_shot:
        return f"{FEW_SHOT_MEDQA}\n\nQuestion:\n{question}\n\n{instruction}\nAnswer:"
    return f"Question:\n{question}\n\n{instruction}\nAnswer:"


def format_rag_prompt(question: str, context: str, use_few_shot: bool = True) -> str:
    """Format RAG prompt with retrieved context.

    Expectation: model should output in the form:
    <LETTER>. <option text>
    """
    base = FEW_SHOT_MEDQA + "\n\n" if use_few_shot else ""
    instruction = (
        "Instructions:\n"
        "- Use the context and your medical knowledge to choose the single best option.\n"
        "- Respond in the format: <LETTER>. <option text>\n"
        "- For example: D. Rhinovirus\n"
    )
    return (
        f"{base}"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"{instruction}\n"
        f"Answer:"
    )


def format_retrieval_decision(question: str) -> str:
    """Format prompt for retrieval decision (Self-BioRAG style)."""
    return (
        "### Instruction:\n"
        "Decide whether external document retrieval is needed to answer the question.\n"
        "Respond with one of: [Retrieval] or [No Retrieval].\n\n"
        f"Question:\n{question}\n\n"
        "### Response:\n"
    )


def format_critic_prompt(question: str, document: str) -> str:
    """Format prompt for document scoring (Self-BioRAG)."""
    return (
        f"{question}\n\n"
        "[Retrieval]<paragraph>\n"
        f"{document}\n"
        "</paragraph>\n\n"
        "Score the usefulness of this paragraph for answering the question.\n"
        "Respond with one of: [Relevant] or [Irrelevant].\n\n"
        "### Response:\n"
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
