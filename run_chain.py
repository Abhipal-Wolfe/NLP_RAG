"""Unified runner for all chains (Baseline and Self-BioRAG)

Usage:
  # Quick test with 3 samples
  python -m selfbiorag --chain baseline --dataset path/to/data.jsonl --max_samples 3
  
  # Full evaluation with few-shot
  python -m selfbiorag --chain baseline --use_few_shot
  
  # Self-BioRAG evaluation
  python -m selfbiorag --chain selfbiorag --mode adaptive_retrieval
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict, Optional, Any
import os

# Load .env file for LangSmith configuration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not required, env vars can be set manually

# Import all chains
from src.chains.baseline_chain import BaselineChain, BaselineConfig
from src.chains.basic_rag_chain import BasicRAGChain, BasicRAGConfig
from src.chains.selfbiorag_chain import SelfBioRAGChain, SelfBioRAGConfig
from src.infrastructure.utils import load_jsonl, load_reflection_tokens
from src.retriever_models import FaissJsonRetriever
from src.generator_models import GeneratorModel


def run_chain(
    chain_type: str,
    dataset_path: str,
    output_path: str,
    verbose: bool = True,
    **config_kwargs
) -> Dict[str, Any]:
    """
    Unified function to run any chain.
    
    Args:
        chain_type: "baseline" or "selfbiorag"
        dataset_path: Path to test dataset
        output_path: Where to save results
        verbose: Print progress and results
        **config_kwargs: Configuration for the selected chain
    
    Returns:
        Dictionary with 'results' and 'metrics'
    """
    if verbose:
        print("=" * 80)
        print(f"RUNNING {chain_type.upper()} CHAIN")
        print("=" * 80)
        print(f"Dataset: {dataset_path}")
        print(f"Output: {output_path}")
        print()
    
    # 1. Load dataset
    if verbose:
        print("Loading dataset...")
    data = load_jsonl(dataset_path)
    if verbose:
        print(f"Loaded {len(data)} samples\n")
    
    # 2. Initialize chain with config
    if chain_type == "baseline":
        config = BaselineConfig(**config_kwargs)
        chain = BaselineChain(config)
    elif chain_type == "basic_rag":
        # Extract FAISS paths
        faiss_index_paths = config_kwargs.pop("faiss_index_paths", {
            "pubmed": "data/indexes_and_articles/Pubmed/Pubmed_Total_Index.faiss",
            "pmc": "data/indexes_and_articles/PMC/PMC_Total_Index.faiss",
            "cpg": "data/indexes_and_articles/CPG/CPG_Total_Index.faiss",
            "textbook": "data/indexes_and_articles/Textbook/Textbook_Total_Index.faiss"
        })
        articles_paths = config_kwargs.pop("articles_paths", {
            "pubmed": "data/indexes_and_articles/Pubmed/Pubmed_Total_Articles.json",
            "pmc": "data/indexes_and_articles/PMC/PMC_Total_Articles.json",
            "cpg": "data/indexes_and_articles/CPG/CPG_Total_Articles.json",
            "textbook": "data/indexes_and_articles/Textbook/Textbook_Total_Articles.json"
        })
        
        # Create retriever
        retriever = FaissJsonRetriever(faiss_index_paths, articles_paths)
        
        config = BasicRAGConfig(**config_kwargs)
        chain = BasicRAGChain(config=config, retriever=retriever)
    elif chain_type == "selfbiorag":
        # Extract FAISS paths (same as basic_rag)
        faiss_index_paths = config_kwargs.pop("faiss_index_paths", {
            "pubmed": "data/indexes_and_articles/Pubmed/Pubmed_Total_Index.faiss",
            "pmc": "data/indexes_and_articles/PMC/PMC_Total_Index.faiss",
            "cpg": "data/indexes_and_articles/CPG/CPG_Total_Index.faiss",
            "textbook": "data/indexes_and_articles/Textbook/Textbook_Total_Index.faiss"
        })
        articles_paths = config_kwargs.pop("articles_paths", {
            "pubmed": "data/indexes_and_articles/Pubmed/Pubmed_Total_Articles.json",
            "pmc": "data/indexes_and_articles/PMC/PMC_Total_Articles.json",
            "cpg": "data/indexes_and_articles/CPG/CPG_Total_Articles.json",
            "textbook": "data/indexes_and_articles/Textbook/Textbook_Total_Articles.json"
        })
        
        # Create config
        config = SelfBioRAGConfig(**config_kwargs)
        
        # Create generator model
        generator = GeneratorModel(
            model_path=config.model_path,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            dtype="half",  # Use half precision
            tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            enforce_eager=True  # Disable CUDA graphs to save memory
        )
        
        # Load reflection tokens
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_reflection_tokens(
            generator.tokenizer,
            use_grounding=config.use_groundedness,
            use_utility=config.use_utility
        )
        
        # Create retriever
        retriever = FaissJsonRetriever(faiss_index_paths, articles_paths)
        
        # Create chain
        chain = SelfBioRAGChain(
            retriever=retriever,
            generator=generator,
            ret_tokens=ret_tokens,
            rel_tokens=rel_tokens,
            grd_tokens=grd_tokens,
            ut_tokens=ut_tokens,
            config=config
        )
    else:
        raise ValueError(f"Unknown chain type: {chain_type}. Choose 'baseline', 'basic_rag', or 'selfbiorag'")
    
    if verbose:
        print(f"Initialized {chain_type} chain")
        print(f"Config: {config}\n")
    
    # 3. Prepare inputs
    instructions = [item.get("instruction", "") for item in data]
    questions = [item["instances"]["input"] for item in data]
    ground_truths = [item["instances"]["output"] for item in data]
    metadata = [{"id": item.get("id", f"q_{i}")} for i, item in enumerate(data)]
    
    # 4. Run chain (all logic is in the chain itself)
    if verbose:
        print("Running evaluation...")
    
    output = chain.invoke({
        "instructions": instructions,
        "questions": questions,
        "ground_truths": ground_truths,
        "metadata": metadata
    })
    
    # 5. Save summary (only metrics, no full predictions)
    summary_path = output_path.replace('.jsonl', '_summary.json')
    summary = {
        "chain_type": chain_type,
        "dataset": dataset_path,
        "num_samples": len(data),
        "metrics": output["metrics"]
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    if verbose:
        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print(f"{'='*80}")
        metrics = output["metrics"]
        
        if "substring_match" in metrics:
            print(f"\n✓ Substring Match Accuracy: {metrics['substring_match']['accuracy']:.2f}%")
            print(f"  Correct: {metrics['substring_match']['correct']}/{metrics['substring_match']['total']}")
        
        if "exact_match" in metrics:
            print(f"\n✓ Exact Match Accuracy: {metrics['exact_match']['accuracy']:.2f}%")
            print(f"  Correct: {metrics['exact_match']['correct']}/{metrics['exact_match']['total']}")
        
        if "avg_docs_used" in metrics:
            print(f"\n✓ Average Documents Used: {metrics['avg_docs_used']:.2f}")
        
        print(f"\nSummary saved: {summary_path}")
        print()
    
    return output


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="Unified runner for Baseline and Self-BioRAG chains",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 3 samples
  python -m selfbiorag --chain baseline --dataset path/to/data.jsonl --max_samples 3
  
  # Full evaluation with few-shot
  python -m selfbiorag --chain baseline --use_few_shot
  
  # Self-BioRAG evaluation
  python -m selfbiorag --chain selfbiorag --mode adaptive_retrieval
        """
    )
    
    # Chain selection
    parser.add_argument(
        "--chain",
        type=str,
        required=True,
        choices=["baseline", "basic_rag", "selfbiorag"],
        help="Which chain to run: baseline (no RAG), basic_rag (simple RAG), or selfbiorag (advanced RAG with reflection)"
    )
    
    # Dataset settings
    parser.add_argument(
        "--dataset",
        type=str,
        default="evidence_data/med_qa_test.jsonl",
        help="Path to test dataset (JSONL format)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions/chain_predictions.jsonl",
        help="Output file path for predictions"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Limit number of samples (None = all)"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    
    # Model settings (shared)
    parser.add_argument(
        "--model_path",
        type=str,
        default="dmis-lab/selfbiorag_7b",
        help="HuggingFace model path"
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=4096,
        help="Maximum model context length (default: 4096)"
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.7,
        help="GPU memory utilization (0.0-1.0, default: 0.7)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (0.0 = greedy)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=200,
        help="Maximum tokens to generate"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    
    # Baseline-specific settings
    parser.add_argument(
        "--no_few_shot",
        action="store_true",
        help="Disable few-shot examples (few-shot is enabled by default)"
    )
    parser.add_argument(
        "--use_exact_match",
        action="store_true",
        help="Use exact match evaluation (default: substring match)"
    )
    
    # Self-BioRAG-specific settings
    parser.add_argument(
        "--mode",
        type=str,
        default="adaptive_retrieval",
        choices=["adaptive_retrieval", "always_retrieve", "no_retrieval"],
        help="Self-BioRAG retrieval mode"
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
        help="Number of documents to retrieve"
    )
    parser.add_argument(
        "--retriever_path",
        type=str,
        default=None,
        help="Path to retriever"
    )
    
    args = parser.parse_args()
    
    # Load and potentially limit dataset
    dataset_path = args.dataset
    if args.max_samples:
        # Load, limit, and save to temp file
        import tempfile
        data = load_jsonl(dataset_path)[:args.max_samples]
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in data:
                f.write(json.dumps(item) + '\n')
            dataset_path = f.name
    
    # Prepare configuration kwargs based on chain type
    if args.chain == "baseline":
        config_kwargs = {
            "model_path": args.model_path,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "use_few_shot": not args.no_few_shot,  # Re-added
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "use_substring_match": not args.use_exact_match,
            "batch_size": args.batch_size,
        }
    elif args.chain == "basic_rag":
        config_kwargs = {
            "model_path": args.model_path,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "top_k": args.top_k,
            "use_few_shot": not args.no_few_shot,  # Re-added
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "use_substring_match": not args.use_exact_match,
            "batch_size": args.batch_size,
        }
    elif args.chain == "selfbiorag":
        config_kwargs = {
            "model_path": args.model_path,
            "max_model_len": args.max_model_len,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "mode": args.mode,
            "top_k": args.top_k,
            "use_few_shot": not args.no_few_shot,  # Re-added
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "batch_size": args.batch_size,
        }
        if args.retriever_path:
            config_kwargs["retriever_path"] = args.retriever_path
    else:
        config_kwargs = {}
    
    # Run the chain
    try:
        run_chain(
            chain_type=args.chain,
            dataset_path=dataset_path,
            output_path=args.output,
            verbose=not args.quiet,
            **config_kwargs
        )
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    finally:
        # Clean up temp file if created
        if args.max_samples and dataset_path != args.dataset:
            import os
            os.unlink(dataset_path)


if __name__ == "__main__":
    main()

