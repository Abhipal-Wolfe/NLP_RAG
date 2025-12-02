# RAG Evaluation Framework for Medical QA

A modular RAG (Retrieval-Augmented Generation) framework for evaluating medical question-answering systems, with full implementation of **Self-BioRAG** augmentations.

## Features

- **Multiple RAG Configurations**: Baseline, Standard RAG, Self-BioRAG
- **Self-BioRAG Augmentations**:
  - Adaptive Retrieval (model decides when to retrieve)
  - Critic-based Reranking (score documents with reflection tokens)
  - Self-Reflection (iteratively improve answers)
- **Large-Scale Corpus**: 24M+ PubMed documents with FAISS indexing
- **Verbose Debugging**: Detailed logging of prompts, tokens, and decisions
- **Batch Processing**: Fast inference with vLLM

---

## Installation

### 1. Install Dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install PyTorch with CUDA (for GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install requirements
pip install -r requirements.txt
```

### 2. HuggingFace Authentication (for gated models)

```bash
# Login to HuggingFace (required for Llama-2 and Self-BioRAG models)
huggingface-cli login
```

### 3. Data Preparation

```bash
# Download evaluation datasets
python data_retrieval_related/get_eval_datasets.py

# Download corpus and build FAISS index (optional - takes time)
python data_retrieval_related/get_corpus_data.py
python data_retrieval_related/build_faiss_index.py
```

---

## Quick Start

### Run Baseline (No Retrieval)

```bash
# Llama-2-7B baseline
python main.py --config config/baseline_llm.yaml --max-samples 100

# Self-BioRAG model baseline (no retrieval)
python main.py --config config/baseline_biollm.yaml --max-samples 100
```

### Run Standard RAG

```bash
python main.py --config config/standard_rag.yaml --max-samples 100
```

### Run Self-BioRAG (Full Pipeline)

```bash
python main.py --config config/selfbio_rag.yaml --max-samples 100
```

---

## Available Configurations

| Config | Retrieval | Model | Augmentations |
|--------|-----------|-------|---------------|
| `baseline_llm.yaml` | ❌ | Llama-2-7B | None |
| `baseline_biollm.yaml` | ❌ | Self-BioRAG-7B | None |
| `standard_rag.yaml` | ✅ | Llama-2-7B | None |
| `selfbio_rag.yaml` | ✅ | Self-BioRAG-7B | Adaptive + Critic + Reflection |

### Ablation Studies

```bash
# Query Processing
python main.py --config config/query_none.yaml      # No query expansion
python main.py --config config/query_multi.yaml     # Multi-query expansion

# Retrieval
python main.py --config config/retrieval_none.yaml      # No retrieval
python main.py --config config/retrieval_adaptive.yaml  # Adaptive retrieval

# Reranking
python main.py --config config/rerank_none.yaml     # No reranking
python main.py --config config/rerank_critic.yaml   # Critic-based reranking

# Reflection
python main.py --config config/reflection_none.yaml # No reflection
python main.py --config config/reflection_self.yaml # Self-reflection
```

---

## Self-BioRAG Pipeline

The Self-BioRAG pipeline consists of 5 stages:

```
Query → [1. Query Processing]
      → [2. Adaptive Retrieval]
      → [3. Retrieval + Critic Reranking]
      → [4. Generation]
      → [5. Self-Reflection]
      → Final Answer
```

### Stage 1: Query Processing
- Optional query expansion/decomposition

### Stage 2: Adaptive Retrieval Decision
- Model generates `[Retrieval]` or `[No Retrieval]` token
- Based on probability threshold (default: 0.5)

### Stage 3: Retrieval + Critic Reranking
- FAISS retrieval from 24M PubMed documents
- Critic reranking using reflection tokens:
  - `[Relevant]` / `[Irrelevant]`
  - `[Fully supported]` / `[Partially supported]` / `[No support]`
  - `[Utility:1-5]`

### Stage 4: Generation
- Generate answer with retrieved context
- Format: `## Context: [Retrieval]<paragraph>...</paragraph>`

### Stage 5: Self-Reflection
- Extract `[Utility:X]` score from generation
- If utility < 4, regenerate (up to N iterations)
- Return best answer

---

## Configuration Options

### Full Self-BioRAG Config

```yaml
# config/selfbio_rag.yaml
retriever:
  type: FAISSRetriever
  top_k: 5
  faiss_index_paths: "data/index/med_faiss.index"
  articles_paths: "data/corpus/med_pure_corpus.jsonl"

generator:
  type: SelfBioRAGGenerator
  model_path: "dmis-lab/selfbiorag_7b"
  max_tokens: 500  # Enough for answer + [Utility:X] token
  temperature: 0.0

augmentations:
  query: []
  
  retrieval:
    - adaptive_retrieval:
        threshold: 0.5  # P([Retrieval]) > 0.5 → retrieve
  
  rerank:
    - critic_reranker:
        top_k: 3        # Keep top 3 after reranking
        w_rel: 1.0      # Weight for relevance
        w_sup: 1.0      # Weight for groundedness
        w_use: 0.5      # Weight for utility
  
  reflection:
    - self_reflection:
        max_iterations: 3

dataset: "eval_datasets/medqa_test.jsonl"
max_samples: null  # null = all samples
verbose: true      # Enable detailed logging
output_path: "predictions/selfbio_rag"
```

---

## Verbose Mode

Enable `verbose: true` in config to see detailed pipeline execution:

```
====================================================================================================
RAG PIPELINE CONFIGURATION
====================================================================================================
Dataset: eval_datasets/medqa_test.jsonl
Samples: 1273
Model: dmis-lab/selfbiorag_7b

PIPELINE STAGES:
  1. Query Processing: 0 augmentation(s)
  2. Retrieval: Enabled
     - Top-K: 5
     - Adaptive: 1 augmentation(s)
  3. Reranking: 1 augmentation(s)
  4. Generation: 0 augmentation(s)
  5. Reflection: 1 augmentation(s)
====================================================================================================

[STAGE 2: ADAPTIVE RETRIEVAL DECISION]
Augmentation 1: adaptive_retrieval_0.5
  Final Decision: ✓ RETRIEVE
  [Retrieval] probability: 0.7103
  [No Retrieval] probability: 0.2897
  Threshold: 0.50

[STAGE 3: RERANKING]
  --- DOCUMENT 1 SCORING ---
  Model output: [Relevant][Fully supported]Answer: D. Nitrofurantoin...[Utility:4]
  Scores:
    Relevance:    0.9200
    Groundedness: 0.8500
    Utility:      0.3000
    Final Score:  0.8467

[STAGE 5: SELF-REFLECTION]
  [Utility:5] (ID=32009): ✓ FOUND at position 130
  ✓ Extracted utility score: 5.0 from token [Utility:5]

====================================================================================================
[FINAL OUTPUT]
Status: CORRECT ✓
Predicted: D | Expected: D
====================================================================================================
```

---

## Results Format

Results are saved as JSON:

```json
{
  "config": "selfbio_rag.yaml",
  "model": "dmis-lab/selfbiorag_7b",
  "num_samples": 1273,
  "metrics": {
    "choice_match": {
      "accuracy": 62.5,
      "correct": 796,
      "total": 1273
    }
  },
  "predictions": ["D", "B", "A", ...]
}
```

---

## Project Structure

```
NLP_RAG/
├── main.py                      # Main entry point
├── requirements.txt
├── config/                      # Configuration files
│   ├── baseline_llm.yaml        # Llama-2 baseline
│   ├── baseline_biollm.yaml     # Self-BioRAG baseline
│   ├── standard_rag.yaml        # Standard RAG
│   ├── selfbio_rag.yaml         # Full Self-BioRAG
│   └── [ablation configs]
├── src/
│   ├── prompts.py               # Prompt templates
│   ├── core/
│   │   ├── config_loader.py     # Config parsing
│   │   ├── interfaces.py        # Abstract base classes
│   │   └── registry.py          # Component registry
│   ├── components/
│   │   ├── generators/          # VLLMGenerator, SelfBioRAGGenerator
│   │   ├── retrievers/          # FAISSRetriever
│   │   └── evaluators/          # AccuracyEvaluator
│   ├── augmentations/
│   │   ├── retrieval/           # Adaptive retrieval
│   │   ├── reranking/           # Critic reranker
│   │   └── reflection/          # Self-reflection
│   └── data/                    # Dataset loaders
├── data/
│   ├── corpus/                  # PubMed corpus (24M docs)
│   └── index/                   # FAISS index files
├── eval_datasets/               # MedQA test/train JSONL
├── predictions/                 # Output results
└── data_retrieval_related/      # Data preparation scripts
```

---

## Expected Performance

| Configuration | MedQA Accuracy |
|--------------|----------------|
| Baseline (Llama-2-7B) | ~45% |
| Baseline (Self-BioRAG-7B) | ~48% |
| Standard RAG | ~52% |
| Self-BioRAG (full) | ~60-65% |

---

## References

- **Self-BioRAG Paper**: [arXiv:2401.15269](https://arxiv.org/abs/2401.15269)
- **Self-BioRAG GitHub**: [dmis-lab/self-biorag](https://github.com/dmis-lab/self-biorag)
- **Self-BioRAG Model**: [HuggingFace](https://huggingface.co/dmis-lab/selfbiorag_7b)
- **MedQA Dataset**: [MedQA USMLE](https://github.com/jind11/MedQA)

---

## Troubleshooting

### Common Issues

1. **`GatedRepoError`**: Run `huggingface-cli login` with valid token
2. **`Out of memory`**: Reduce `batch_size` or `top_k` in config
3. **`Utility score -1`**: Increase `max_tokens` (default 500+ recommended)
4. **`Prompt too long`**: Reduce `top_k` for retrieval

### Recommended Settings

```yaml
generator:
  max_tokens: 500      # Enough for utility tokens
  temperature: 0.0     # Deterministic output

retriever:
  top_k: 3-5          # Balance between context and prompt length

rerank:
  critic_reranker:
    top_k: 1-3        # Keep best documents
```
