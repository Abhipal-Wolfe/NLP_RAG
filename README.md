# NLP_RAG

 1. Download pre-computed FAISS indices and article jsons from [Download from Google Drive][https://drive.google.com/file/d/1UHDtwcADcPh0AlfesVJValKqUYHrS83S/view?usp=sharing]
 2. Look at src/retriever_models/example_usage.py
=======
# Self-BioRAG Implementation & Evaluation Framework

## Project Structure

```
NLP_RAG/
├── src/              # Main package - modular evaluation framework
│   ├── chains/              # Evaluation chains (baseline, RAG, Self-BioRAG)
│   ├── generator_models/    # LLM wrappers (vLLM-based generators)
│   ├── retriever_models/    # Retrieval systems (biomedical retrievers)
│   ├── infrastructure/      # Utilities (metrics, data loading, configs)
│   └── run_chain.py         # CLI entry point for running evaluations
│
├── predictions/             # Evaluation results (JSON outputs)
├── requirements.txt         # Python dependencies
└── .env.example             # LangSmith tracing configuration
```

---

## Core Components

### **1. src/** - Main Package

#### **chains/** - Evaluation Pipelines
- **`base.py`** - Abstract Chain with LangSmith tracing
- **`baseline_chain.py`** - Standard LLM (zero/few-shot, no retrieval)
- **`basic_rag_chain.py`** - Simple RAG with pre-retrieved evidence
- **`selfbiorag_chain.py`** - Full Self-BioRAG with adaptive retrieval and reflection tokens

#### **generator_models/** - LLM Wrappers
- **`models.py`** - vLLM-based interface supporting HuggingFace models with batch inference and GPU optimization

#### **retriever_models/** - Retrieval Systems
- **`retriever.py`** - Biomedical retriever with pre-retrieved evidence loading and optional FAISS search

#### **infrastructure/** - Utilities
- **`utils.py`** - Metrics (`normalize_answer`, `calculate_accuracy`), data I/O
- **`factory.py`** - Configuration and initialization

#### **Root Files**
- **`run_chain.py`** - CLI interface for running evaluations
- **`__main__.py`** - Enables `python -m selfbiorag` execution
- **`__init__.py`** - Package initialization and exports

---

### **2. predictions/** - Evaluation Outputs

JSON files containing model predictions and evaluation metrics:
```json
{
  "prediction": "Model's answer",
  "ground_truth": "Correct answer",
  "is_correct_substring": true,
  "is_correct_exact": false,
  "metadata": {"question_id": "...", "dataset": "med_qa"}
}
```

---

## Quick Start

### **Installation**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or: venv/Scripts/activate on Windows

# Install dependencies
pip install -r requirements.txt
```

### **Run Evaluation**
```bash
# Baseline LLM (no retrieval)
python -m selfbiorag --chain baseline --max_samples 10

# Basic RAG
python -m selfbiorag --chain basic_rag --max_samples 10

# Full Self-BioRAG
python -m selfbiorag --chain selfbiorag --max_samples 10

# Customize model and settings
python -m selfbiorag --chain baseline \
  --model_path meta-llama/Llama-2-7b-chat-hf \
  --dataset med_qa \
  --max_samples 100 \
  --temperature 0.0
```

### **View Results**
```bash
# Check predictions
cat predictions/baseline_*.json

# View summary
cat predictions/baseline_*_summary.json
```

---

## Key Features

- **Modular Design** - Swap models, chains, and retrievers easily
- **LangSmith Tracing** - Built-in observability (configure via `.env`)
- **Multiple Metrics** - Exact match and substring match evaluation
- **Few-Shot Support** - Automatic few-shot examples from training data
- **Batch Processing** - Efficient GPU utilization with vLLM
- **Pre-Retrieved Evidence** - 208 MB of evidence ready for evaluation

---

## Configuration

### **Environment Variables** (`.env`)
```bash
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=your_api_key
LANGCHAIN_PROJECT=selfbiorag-evaluation
```

### **CLI Arguments**
- `--chain` - baseline | basic_rag | selfbiorag
- `--model_path` - HuggingFace model ID
- `--dataset` - med_qa | medmc_qa | mmlu
- `--max_samples` - Number of questions to evaluate
- `--no_few_shot` - Disable few-shot prompting
- `--max_model_len` - Context window size (default: 4096)
- `--gpu_memory_utilization` - GPU memory fraction (default: 0.7)

---

## Resources

- **Original Paper**: [Self-BioRAG (arXiv)](https://arxiv.org/abs/2401.15269)
- **Original Code**: [dmis-lab/self-biorag](https://github.com/dmis-lab/self-biorag)
- **Models**: [HuggingFace Hub](https://huggingface.co/dmis-lab)

---
>>>>>>> eeb4e1d (initial commit with the baseline, base RAG, selfbio rag implementations)
