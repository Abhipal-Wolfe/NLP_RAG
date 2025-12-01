from datasets import load_dataset
import json

ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="train")

with open("data/eval_datasets/medqa_train.jsonl", "w", encoding="utf-8") as f:
    for ex in ds:
        f.write(json.dumps({
            "question": ex["question"],
            "options": ex["options"],
            "answer_text": ex.get("answer"),
            "answer_idx": ex.get("answer_idx")
        }) + "\n")


ds_test = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")

with open("data/eval_datasets/medqa_test.jsonl", "w", encoding="utf-8") as f:
    for ex in ds_test:
        f.write(json.dumps({
            "question": ex["question"],
            "options": ex["options"],
            "answer_text": ex.get("answer"),
            "answer_idx": ex.get("answer_idx")
        }) + "\n")
