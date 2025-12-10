from datasets import load_dataset
import json
import os # import the os module

output_dir = "data/eval_datasets/"
os.makedirs(output_dir, exist_ok=True)

# ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

# with open(os.path.join(output_dir, "medqa_train.jsonl"), "w", encoding="utf-8") as f:
#     for ex in ds:
#         f.write(json.dumps({
#             "question": ex["question"],
#             "options": ex["options"],
#             "answer_text": ex.get("answer"),
#             "answer_idx": ex.get("answer_idx")
#         }) + "\n")


# ds_test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

# with open(os.path.join(output_dir, "medqa_test.jsonl"), "w", encoding="utf-8") as f:
#     for ex in ds_test:
#         f.write(json.dumps({
#             "question": ex["question"],
#             "options": ex["options"],
#             "answer_text": ex.get("answer"),
#             "answer_idx": ex.get("answer_idx")
#         }) + "\n")


# PubMedQA dataset (yes/no/maybe over PubMed abstracts)
# Using original dataset: qiaojin/PubMedQA
# Configs: pqa_labeled (1,000 expert-annotated), pqa_unlabeled, pqa_artificial
# We use pqa_labeled for evaluation
ds_pubmedqa = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")


with open(os.path.join(output_dir, "pubmedqa_test.jsonl"), "w", encoding="utf-8") as f:
    for ex in ds_pubmedqa:
        # Map yes/no/maybe to option indices (A=yes, B=no, C=maybe)
        answer_map = {"yes": "A", "no": "B", "maybe": "C"}
        
        # Handle different field names between dataset versions
        # Original qiaojin/PubMedQA uses "final_decision" or "label"
        # bigbio version uses "final_decision"
        final_decision = ex.get("final_decision") or ex.get("label") or ex.get("answer", "")
        if isinstance(final_decision, str):
            final_decision = final_decision.lower()
        else:
            final_decision = str(final_decision).lower()
        
        answer_idx = answer_map.get(final_decision, "")
        
        # Get question - may be in different fields
        question = ex.get("question") or ex.get("QUESTION", "")
        
        # Get context/abstract - may be in different fields
        context = ex.get("context") or ex.get("CONTEXT") or ex.get("abstract", "")
        
        # Get long answer - may be in different fields
        long_answer = ex.get("long_answer") or ex.get("LONG_ANSWER") or ex.get("final_decision", "")
        
        f.write(json.dumps({
            "question": question,
            "context": context,  # PubMed abstract
            "long_answer": long_answer,  # Abstract conclusion
            "options": {"A": "yes", "B": "no", "C": "maybe"},
            "answer_text": final_decision,
            "answer_idx": answer_idx
        }) + "\n")
