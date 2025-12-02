# Self-BioRAG Implementation

## ✅ Implemented Features

### 1. **Adaptive Retrieval** (`src/augmentations/retrieval/adaptive_retrieval.py`)

**How it works:**
1. Model generates with `[Retrieval]` or `[No Retrieval]` tokens
2. Extracts log probabilities for both tokens
3. Computes probability ratio: `P([Retrieval]) / (P([Retrieval]) + P([No Retrieval]))`
4. If ratio > threshold (default 0.5), retrieves documents; otherwise skips

**Config:**
```yaml
retrieval:
  - adaptive_retrieval:
      threshold: 0.5  # Probability threshold for retrieval decision
```

**Verbose Output:**
```
[STAGE 2: ADAPTIVE RETRIEVAL DECISION]

Augmentation 1: adaptive_retrieval_0.5
  Decision: RETRIEVE (or NO RETRIEVAL)
  retrieval_score: -2.3456
  no_retrieval_score: -4.1234
  should_retrieve: True
```

---

### 2. **Critic Reranker** (`src/augmentations/reranking/critic_reranker.py`)

**How it works:**
1. For each retrieved document, generates with reflection tokens
2. Extracts scores from special tokens:
   - `[Relevant]` / `[Irrelevant]` → Relevance score (0-1)
   - `[Fully supported]` / `[Partially supported]` / `[No support]` → Groundedness score (0-1)
   - `[Utility:1-5]` → Utility score (-1 to 1)
3. Computes weighted score: `w_rel × rel + w_sup × grd + w_use × util`
4. Reranks documents by score and returns top-k

**Config:**
```yaml
rerank:
  - critic_reranker:
      top_k: 3       # Number of docs to keep after reranking
      w_rel: 1.0     # Weight for relevance
      w_sup: 1.0     # Weight for groundedness  
      w_use: 0.5     # Weight for utility
```

**Verbose Output:**
```
[STAGE 3: RERANKING]
Documents before reranking: 5

Augmentation 1: critic_reranker_top3
  Before:
    Doc 1 - Score: 0.85
    Doc 2 - Score: 0.82
    ...
  After:
    Doc 1 - Score: 0.85 | Rerank: 0.9234 | Rel: 0.95 | Sup: 0.89 | Util: 0.5
    Doc 2 - Score: 0.82 | Rerank: 0.8876 | Rel: 0.92 | Sup: 0.85 | Util: 0.3
    ...

Documents after reranking: 3
```

---

### 3. **Self-Reflection** (`src/augmentations/reflection/self_reflection.py`)

**How it works:**
1. Generates answer with reflection tokens
2. Extracts `[Utility:1-5]` score from model output
3. If utility < 4 (not high quality), regenerates answer
4. Repeats up to `max_iterations` times
5. Returns best answer across all iterations

**Utility Scoring:**
- `[Utility:5]` = Perfect answer
- `[Utility:4]` = Good answer (stops here)
- `[Utility:3]` = Okay answer (regenerate)
- `[Utility:2]` = Poor answer (regenerate)
- `[Utility:1]` = Bad answer (regenerate)

**Config:**
```yaml
reflection:
  - self_reflection:
      max_iterations: 3  # Maximum refinement attempts
```

**Verbose Output:**
```
[STAGE 5: SELF-REFLECTION]
Initial answer: The correct answer is A. Ampicillin...

Augmentation 1: self_reflection_iter3
  Iterations: 2
  Support scores: [2.3, 3.8]
  Improved: True
  Final score: 3.8000
  
  Answer changed after reflection
  New answer: The correct answer is D. Nitrofurantoin...
```

---

## 🔧 Implementation Details

### **Augmentation Loader** (`src/augmentations/loader.py`)

Converts YAML config to callable augmentation functions:

```python
from src.augmentations.loader import load_augmentations

# Load from config
augmentations = load_augmentations(config)

# Returns:
{
    "query": [],
    "retrieval": [adaptive_retrieval_fn],
    "rerank": [critic_reranker_fn],
    "generation": [],
    "reflection": [self_reflection_fn]
}
```

### **Reflection Token Support** (`src/components/generators/selfbiorag_generator.py`)

The Self-BioRAG generator has:
- `generate_with_logprobs()` - Generate with token probabilities
- `get_reflection_tokens()` - Get special token IDs
- Special tokens: `[Retrieval]`, `[Relevant]`, `[Fully supported]`, `[Utility:1-5]`, etc.

---

## 📊 Usage Examples

### **Run Standard RAG (no augmentations):**
```bash
python main.py --config config/standard_rag.yaml
```

### **Run Self-BioRAG (with all augmentations):**
```bash
python main.py --config config/selfbio_rag.yaml
```

### **Debug with verbose mode:**
```yaml
# In config file:
verbose: true  # Shows detailed decision-making at each stage
```

---

## 🎯 Expected Performance

| Configuration | Retrieval | Reranking | Reflection | Expected Accuracy |
|--------------|-----------|-----------|------------|-------------------|
| **Baseline** | None | None | None | ~45% |
| **Standard RAG** | Always | None | None | ~50-55% |
| **Self-BioRAG** | Adaptive | Critic | Self-Reflection | ~60-65%+ |

---

## 🐛 Troubleshooting

### **Prompt too long error:**
```
ValueError: The decoder prompt (length 5960) is longer than the maximum model length of 4096
```

**Fix:** Reduce `top_k` in retriever config:
```yaml
retriever:
  top_k: 1  # Reduce from 5 to 1 or 2
```

### **Augmentations not working:**
- Check that `SelfBioRAGGenerator` is being used (not `VLLMGenerator`)
- Verify model has reflection tokens: `dmis-lab/selfbiorag_7b`
- Enable verbose mode to see augmentation outputs

### **Slow generation:**
- Reflection adds 2-3x overhead (regenerates up to N times)
- Reranking adds 1.5x overhead (scores each document)
- Adaptive retrieval saves time by skipping unnecessary retrievals

---

## 📝 Notes

1. **Reflection tokens** are special tokens added to Self-BioRAG during training
2. **Log probabilities** are used to extract model's confidence in reflection tokens
3. **Adaptive retrieval** can skip retrieval entirely for easy questions
4. **Critic reranking** improves document quality before generation
5. **Self-reflection** iteratively refines answers until high utility

---

## 🚀 Next Steps

- [ ] Implement query decomposition augmentation
- [ ] Add chain-of-thought generation augmentation
- [ ] Optimize batch processing for augmented pipelines
- [ ] Add caching for repeated reflection token extractions

