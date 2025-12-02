# Self-BioRAG Prompt Format & Reflection Tokens

## 🎯 Key Insight: Self-BioRAG Model is Pre-trained

**The `dmis-lab/selfbiorag_7b` model is specifically trained to generate reflection tokens.**

The model was trained with:
1. **Special tokens added to vocabulary**: `[Retrieval]`, `[Relevant]`, `[Utility:1-5]`, etc.
2. **Training data with reflection tokens**: Examples explicitly showing when to generate these tokens
3. **Instruction-following format**: Using specific prompt structures that trigger token generation

---

## 📋 Self-BioRAG Prompt Format

### **1. For Generation (NO retrieval)**
```
## Question:
[Question text with options]

## Response:
[No Retrieval]Answer: B. Option text
Explanation: [reasoning]
```

The model generates `[No Retrieval]` to indicate it doesn't need external knowledge.

### **2. For Generation (WITH retrieval)**
```
## Context:
[Retrieval]<paragraph>[Document content]</paragraph>

## Question:
[Question text with options]

## Response:
[Relevant][Fully supported]Answer: B. Option text
Explanation: [reasoning][Utility:4]
```

The model generates:
- `[Relevant]` - Document is relevant
- `[Fully supported]` - Answer is supported by document
- `[Utility:4]` - Answer quality score (1-5)

---

## 🔑 When Are Reflection Tokens Generated?

### **[Retrieval] / [No Retrieval]**
- **Position**: First token of generation
- **Purpose**: Model decides if it needs external documents
- **Triggered by**: Empty context or query-only prompts

### **[Relevant] / [Irrelevant]**
- **Position**: After seeing `[Retrieval]<paragraph>` in context
- **Purpose**: Evaluate document relevance
- **Triggered by**: Presence of `[Retrieval]` and `<paragraph>` tags

### **[Fully supported] / [Partially supported] / [No support]**
- **Position**: After generating answer with context
- **Purpose**: Self-evaluate groundedness
- **Triggered by**: Generating answer after seeing context

### **[Utility:1-5]**
- **Position**: END of generation
- **Purpose**: Overall answer quality (1=bad, 5=excellent)
- **Triggered by**: Model completes answer generation

---

## ⚠️ Common Issues

### **Issue 1: Model Not Generating Utility Tokens**

**Symptom**: Utility score always `-1`

**Possible Causes:**
1. **Prompt format doesn't match training**
   - Self-BioRAG was trained with specific format
   - Our format might differ slightly

2. **Few-shot examples don't show utility tokens**
   - Model learns from examples
   - If examples don't have `[Utility:X]`, model won't generate them

3. **Context too long**
   - If max_tokens is reached before utility token, it's truncated
   - Solution: Increase `max_tokens` or reduce context

4. **Wrong model**
   - Using base Llama-2 instead of `dmis-lab/selfbiorag_7b`
   - Only Self-BioRAG model has reflection tokens in vocabulary

### **Issue 2: Tokens Generated But Not Parsed**

**Symptom**: Tokens in raw output but utility score still `-1`

**Possible Causes:**
1. **Tokens stripped before extraction**
   - Using `_postprocess()` removes tokens
   - Solution: Extract from `raw_output` in metadata

2. **Tokens in wrong position**
   - Looking for tokens at start, but they're at end
   - Solution: Search entire token sequence

---

## 🔧 How Our Implementation Works

### **Current Flow:**

1. **Generation** → Model outputs with reflection tokens
   ```
   [Relevant][Fully supported]Answer: D. Nitrofurantoin...[Utility:4]
   ```

2. **Store in metadata** → Before cleaning
   ```python
   metadata = {
       "raw_output": full_text,
       "token_ids": [1234, 5678, ...],
       "logprobs": [{...}, {...}, ...]
   }
   ```

3. **Clean for evaluation** → Remove tokens
   ```python
   cleaned = "Answer: D. Nitrofurantoin..."  # Tokens removed
   ```

4. **Reflection extracts from metadata** → Use raw output
   ```python
   utility_score = extract_from(metadata['token_ids'], metadata['logprobs'])
   ```

---

## 💡 Recommendations

### **For Few-Shot Examples:**

**Option A: No reflection tokens (simpler)**
```
## Response:
Answer: B. Primary hypothyroidism
Explanation: [reasoning]
```
- Model may still generate tokens if trained to do so
- Simpler, cleaner examples

**Option B: With reflection tokens (explicit)**
```
## Response:
[Relevant][Fully supported]Answer: B. Primary hypothyroidism
Explanation: [reasoning][Utility:5]
```
- Shows model exactly what to generate
- May be redundant if model already trained

**Current choice**: Option A (user removed tokens from examples)

### **For Utility Token Generation:**

The model should generate `[Utility:X]` at the END of its response automatically if:
1. ✅ Using `dmis-lab/selfbiorag_7b` model
2. ✅ Prompt format matches training (with `## Question:`, `## Response:`)
3. ✅ `max_tokens` is sufficient (200+ tokens recommended)
4. ✅ Context includes `[Retrieval]<paragraph>` tags

### **Testing:**

Run with `verbose: true` and check:
```
--- RAW MODEL OUTPUT ---
[Relevant][Fully supported]Answer: D. Nitrofurantoin is the preferred...[Utility:4]
```

If you see `[Utility:X]` in raw output → extraction is working!
If you DON'T see it → model isn't generating it (prompt issue or model issue)

---

## 📚 References

- **Paper**: Self-BioRAG: Improving Biomedical RAG via Self-Reflection (https://arxiv.org/abs/2401.15269)
- **Model**: https://huggingface.co/dmis-lab/selfbiorag_7b
- **GitHub**: https://github.com/dmis-lab/self-biorag

**Note**: The exact training prompts and data formats may vary from the paper. The model learns these patterns during training on biomedical instruction-tuning data with reflection token annotations.

