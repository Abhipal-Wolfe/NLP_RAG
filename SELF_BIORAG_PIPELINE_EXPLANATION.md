# Self-BioRAG Pipeline - How Final Output is Generated

## 📋 Complete Pipeline Flow

```
Query → [Stage 1: Query Processing]
     → [Stage 2: Adaptive Retrieval Decision]
     → [Stage 3: Retrieval (if needed)]
     → [Stage 4: Critic Reranking]
     → [Stage 5: Generation]
     → [Stage 6: Self-Reflection]
     → Final Answer
```

---

## 🔍 Stage-by-Stage Breakdown

### **Stage 1: Query Processing** (Currently: No Augmentations)
```python
processed = [query]  # Query passed through unchanged
```

**Output:** Original formatted query with options

---

### **Stage 2: Adaptive Retrieval Decision** 

**Goal:** Model decides if it needs external knowledge

**Process:**
1. **Format prompt:**
   ```
   ### Instruction:
   [Question with options]
   
   ### Response:
   ```

2. **Generate with logprobs** (max 20 tokens)
   - Model outputs: `[Retrieval]` or `[No Retrieval]` as first token
   - Extract log probabilities for both tokens

3. **Calculate decision:**
   ```python
   P(retrieve) = exp(logprob([Retrieval])) / (exp(logprob([Retrieval])) + exp(logprob([No Retrieval])))
   should_retrieve = P(retrieve) > threshold (default 0.5)
   ```

**Example Output:**
```
[Retrieval] prob: 0.71 → RETRIEVE
[No Retrieval] prob: 0.29
```

**Our Config:** `threshold: 0.5`

---

### **Stage 3: Retrieval** (if Stage 2 decided to retrieve)

**Process:**
1. Embed query using `NeuML/pubmedbert-base-embeddings`
2. FAISS search in 24M document corpus
3. Retrieve top-k documents (config: `top_k: 1`)

**Output:** List of retrieved documents with scores

---

### **Stage 4: Critic Reranking**

**Goal:** Score and rerank retrieved documents using reflection tokens

**Process:**
1. **For each document, generate prompt:**
   ```
   [Question]
   [Retrieval]<paragraph>[document text]</paragraph>
   
   ### Response:
   ```

2. **Extract reflection token probabilities:**
   - `[Relevant]` vs `[Irrelevant]` → **Relevance score** (0-1)
   - `[Fully supported]` / `[Partially supported]` / `[No support]` → **Groundedness score** (0-1)
   - `[Utility:1-5]` → **Utility score** (-1 to 1)

3. **Compute weighted score:**
   ```python
   score = w_rel × relevance + w_sup × groundedness + w_use × utility
   ```

4. **Rerank and keep top-k**

**Our Config:**
```yaml
top_k: 3  # Keep top 3 after reranking
w_rel: 1.0
w_sup: 1.0
w_use: 0.5
```

**Output:** Reranked documents (best quality first)

---

### **Stage 5: Generation**

**Goal:** Generate answer using query + retrieved documents

**Prompt Format:**
```
### Instruction:
[Question with options]

[Retrieval]<paragraph>[Doc 1]</paragraph>

<paragraph>[Doc 2]</paragraph>

### Response:
```

**Model generates:**
```
[No Retrieval][Relevant][Fully supported]The correct answer is D. Nitrofurantoin...[Utility:4]
```

**Note:** 
- `[Retrieval]` prefix tells model context is provided
- Model may generate additional reflection tokens during generation
- These tokens are kept in raw output for reflection stage

**Output:** Raw answer with reflection tokens

---

### **Stage 6: Self-Reflection** (Iterative Refinement)

**Goal:** Evaluate and improve answer quality

**Process:**

**Iteration 1:**
1. **Format prompt with answer:**
   ```
   ### Instruction:
   [Question]
   
   ### Response:
   [Generated answer]
   ```

2. **Generate with logprobs** (max 50 tokens)
   - Look for `[Utility:X]` tokens in output

3. **Extract utility score:**
   ```python
   # Compute expected utility from probabilities
   utility = Σ(i × P([Utility:i])) for i = 1 to 5
   
   # Example:
   [Utility:1]: 0.05 → 1 × 0.05 = 0.05
   [Utility:2]: 0.10 → 2 × 0.10 = 0.20
   [Utility:3]: 0.15 → 3 × 0.15 = 0.45
   [Utility:4]: 0.60 → 4 × 0.60 = 2.40
   [Utility:5]: 0.10 → 5 × 0.10 = 0.50
   Expected utility = 3.60
   ```

4. **Decision:**
   - If `utility >= 4.0` → **STOP** (good answer)
   - If `utility < 4.0` → **REGENERATE** (try again)

**Iteration 2 (if utility < 4.0):**
1. Regenerate answer with same prompt + context
2. Extract new utility score
3. Keep best answer across iterations

**Iteration 3 (if still < 4.0):**
1. Final regeneration attempt
2. Return best answer found

**Our Config:** `max_iterations: 3`

**Output:** Best answer across all iterations

---

## 🎯 Final Output Generation

### **Step 1: Post-processing**
```python
# Remove all reflection tokens
tokens_to_remove = [
    "[Retrieval]", "[No Retrieval]", 
    "[Relevant]", "[Irrelevant]",
    "[Fully supported]", "[Partially supported]", "[No support]",
    "[Utility:1]", "[Utility:2]", "[Utility:3]", "[Utility:4]", "[Utility:5]",
    "</s>", "<|endoftext|>"
]

clean_answer = answer
for token in tokens_to_remove:
    clean_answer = clean_answer.replace(token, " ")
```

### **Step 2: Extract letter answer**
```python
# From src/prompts.py clean_answer()
# Looks for patterns like:
# - "A. Option text"
# - "Answer: A"
# - "The answer is A"

extracted = "A"  # or B, C, D
```

### **Step 3: Evaluation**
```python
predicted = "D"
expected = "D"
correct = (predicted == expected)  # True
```

---

## 📊 Example: Complete Flow

### **Input:**
```
Question: A 23-year-old pregnant woman at 22 weeks gestation presents 
with burning upon urination. Which treatment?

Options:
A. Ampicillin
B. Ceftriaxone
C. Doxycycline
D. Nitrofurantoin

Ground Truth: D
```

### **Processing:**

1. **Adaptive Retrieval:** 
   - P([Retrieval]) = 0.71 → **RETRIEVE**

2. **Retrieval:** 
   - Found 1 document about UTI in pregnancy

3. **Critic Reranking:** 
   - Score: 0.89 (rel=0.95, grd=0.85, util=0.4) → **KEEP**

4. **Generation:**
   ```
   [No Retrieval][Relevant][Fully supported]
   The correct answer is D. Nitrofurantoin is the preferred 
   treatment for uncomplicated UTI in pregnancy because it is 
   safe in the second trimester...[Utility:4]
   ```

5. **Self-Reflection:**
   - Iteration 1: utility = 4.2 → **STOP** (high quality)

6. **Post-process:**
   ```
   Raw: "[No Retrieval][Relevant]...is D. Nitrofurantoin...[Utility:4]"
   Clean: "The correct answer is D. Nitrofurantoin is the preferred..."
   Extracted: "D"
   ```

7. **Result:** 
   - Predicted: D
   - Expected: D
   - **✓ CORRECT**

---

## 🔑 Key Differences from Standard RAG

| Feature | Standard RAG | Self-BioRAG (Ours) |
|---------|-------------|-------------------|
| **Retrieval** | Always retrieves | Adaptive (model decides) |
| **Document Selection** | Top-K by similarity | Critic reranking with reflection tokens |
| **Generation** | One-shot | Iterative with quality checks |
| **Prompt Format** | Simple context injection | Structured with `[Retrieval]` markers |
| **Output** | Raw answer | Cleaned + reflection tokens removed |
| **Quality Control** | None | Utility-based regeneration |

---

## 🎨 Reflection Tokens Explained

### **Retrieval Tokens:**
- `[Retrieval]` - Need external knowledge
- `[No Retrieval]` - Can answer from internal knowledge

### **Relevance Tokens:**
- `[Relevant]` - Document is relevant to query
- `[Irrelevant]` - Document is not relevant

### **Groundedness Tokens:**
- `[Fully supported]` - Answer fully supported by context
- `[Partially supported]` - Partially supported
- `[No support]` - Not supported by context

### **Utility Tokens:**
- `[Utility:5]` - Perfect answer
- `[Utility:4]` - Good answer ← **Threshold for stopping**
- `[Utility:3]` - Acceptable but could improve
- `[Utility:2]` - Poor answer
- `[Utility:1]` - Bad answer

---

## 🐛 Current Implementation Status

✅ **Implemented:**
- Adaptive retrieval decision
- FAISS retrieval with byte-offset index
- Critic reranking with reflection tokens
- Self-reflection with utility-based regeneration
- Augmentation loader from YAML config

⚠️ **Limitations:**
- Prompt format may not exactly match original Self-BioRAG paper
- No query decomposition (yet)
- No chain-of-thought generation (yet)
- Reranking works but may be slow (generates for each document)
- Reflection adds 2-3x generation overhead

🔧 **Next Steps:**
- Verify prompt format matches Self-BioRAG paper
- Optimize batch processing for reranking
- Add caching for reflection token extractions
- Compare results with original Self-BioRAG implementation

---

## 📈 Expected Performance

Based on Self-BioRAG paper:
- **Baseline (Llama-2-7B):** ~45% accuracy
- **Standard RAG:** ~50-55% accuracy
- **Self-BioRAG (full pipeline):** ~60-65% accuracy

Performance gains come from:
- **Adaptive retrieval:** Skip unhelpful retrievals (+2-3%)
- **Critic reranking:** Better document quality (+3-5%)
- **Self-reflection:** Answer refinement (+5-7%)

