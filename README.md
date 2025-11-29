# RAG Evaluation Framework

## Quick Start

```bash
pip install -r requirements.txt

# Baseline (no retrieval)
python main.py --config config/experiments/baseline_unified.yaml --max-samples 10

# Self-BioRAG (full pipeline)
python main.py --config config/experiments/self_biorag_unified.yaml --max-samples 10
```

## Project Structure

```
├── main.py
├── src/
│   ├── prompts.py            # All prompts (separated)
│   ├── core/
│   │   ├── config_loader.py  # Config parsing + component builder
│   │   ├── interfaces.py     # Abstract base classes
│   │   └── registry.py       # Component factory
│   ├── components/           # Retrievers, generators, evaluators
│   ├── augmentations/        # Plugin functions
│   │   ├── query_processing/
│   │   ├── retrieval/
│   │   ├── reranking/
│   │   ├── generation/
│   │   └── reflection/
│   ├── data/                 # Dataset loaders
│   └── utils/                # I/O, embedders
└── config/
    ├── experiments/          # Full pipeline configs
    └── ablations/            # Ablation study configs
```

## Ablation Studies

All ablations use the **same `main.py`** - just different config files!

### Query Processing

```bash
# No query processing
python main.py --config config/ablations/query_none.yaml

# Multi-query expansion
python main.py --config config/ablations/query_multi.yaml
```

### Retrieval

```bash
# No retrieval (baseline)
python main.py --config config/ablations/retrieval_none.yaml

# Adaptive retrieval (Self-BioRAG)
python main.py --config config/ablations/retrieval_adaptive.yaml
```

### Reranking

```bash
# No reranking
python main.py --config config/ablations/rerank_none.yaml

# Critic-based reranking (Self-BioRAG)
python main.py --config config/ablations/rerank_critic.yaml
```

### Reflection

```bash
# No reflection
python main.py --config config/ablations/reflection_none.yaml

# Self-reflection (Self-BioRAG)
python main.py --config config/ablations/reflection_self.yaml
```

See `config/ablations/README.md` for more details.

## Configuration

```yaml
# config/experiments/self_biorag_unified.yaml
generator:
  type: SelfBioRAGGenerator
  model_path: "dmis-lab/selfbiorag_7b"

retriever:
  type: FAISSRetriever
  top_k: 5

augmentations:
  retrieval:
    - adaptive_retrieval:  # Decide if retrieval needed
        threshold: 0.5
  rerank:
    - critic_reranker:     # Score docs with LLM
        top_k: 3
  reflection:
    - self_reflection:     # Validate answer
        max_iterations: 2

dataset: "test_data/med_qa_train.json"
output_path: "experiments/results/self_biorag"
```

## Results

Saved to `experiments/results/`:
```json
{
  "predictions": ["answer1", "answer2", ...],
  "metrics": {
    "substring_match": {"accuracy": 65.5, "correct": 655, "total": 1000}
  }
}
```

## References

- [Self-BioRAG Paper](https://arxiv.org/abs/2401.15269)
- [Original Code](https://github.com/dmis-lab/self-biorag)
