"""Utility functions for Self-BioRAG"""

import re
import string
import json
from typing import List, Dict
from pathlib import Path


def load_reflection_tokens(tokenizer, use_grounding=True, use_utility=True):
    """
    Load reflection token IDs from tokenizer.
    
    Returns: (ret_tokens, rel_tokens, grd_tokens, ut_tokens)
    """
    # Try to get token IDs - handle case where tokens might not exist
    def safe_get_token_id(token_str):
        try:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            if token_id == tokenizer.unk_token_id:
                # Token not found, try adding it
                print(f"WARNING: Token '{token_str}' not found in tokenizer vocabulary")
                return None
            return token_id
        except Exception as e:
            print(f"WARNING: Error loading token '{token_str}': {e}")
            return None
    
    ret_tokens = {
        "[Retrieval]": safe_get_token_id("[Retrieval]"),
        "[No Retrieval]": safe_get_token_id("[No Retrieval]")
    }
    
    # Remove None values and warn
    ret_tokens = {k: v for k, v in ret_tokens.items() if v is not None}
    if len(ret_tokens) < 2:
        print(f"ERROR: Could not load retrieval tokens! Found: {ret_tokens}")
        print(f"Tokenizer vocab size: {len(tokenizer)}")
        print(f"Sample tokens in vocab: {list(tokenizer.get_vocab().keys())[:20]}")
    
    rel_tokens = {
        "[Relevant]": tokenizer.convert_tokens_to_ids("[Relevant]"),
        "[Irrelevant]": tokenizer.convert_tokens_to_ids("[Irrelevant]")
    }
    
    grd_tokens = None
    if use_grounding:
        grd_tokens = {
            "[Fully supported]": tokenizer.convert_tokens_to_ids("[Fully supported]"),
            "[Partially supported]": tokenizer.convert_tokens_to_ids("[Partially supported]"),
            "[No support]": tokenizer.convert_tokens_to_ids("[No support]")
        }
    
    ut_tokens = None
    if use_utility:
        ut_tokens = {
            f"[Utility:{i}]": tokenizer.convert_tokens_to_ids(f"[Utility:{i}]")
            for i in range(1, 6)
        }
    
    return ret_tokens, rel_tokens, grd_tokens, ut_tokens


def normalize_answer(s: str) -> str:
    """Normalize answer for evaluation (removes articles, punctuation, lowercase)"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> Dict:
    """Calculate exact match accuracy with normalization"""
    correct = sum(
        normalize_answer(pred) == normalize_answer(gt)
        for pred, gt in zip(predictions, ground_truths)
    )
    
    return {
        "accuracy": 100 * correct / len(predictions),
        "correct": correct,
        "total": len(predictions)
    }


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file into list of dicts"""
    with open(file_path) as f:
        return [json.loads(line) for line in f]


def save_jsonl(data: List[Dict], file_path: str):
    """Save list of dicts to JSONL file"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')