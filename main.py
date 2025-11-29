"""RAG Evaluation Framework - Domain logic visible in main()"""

from src.core.config_loader import load_config, build_components
from src.data import DatasetLoader
from src.prompts import format_rag_prompt, clean_answer
from src.utils import save_results
from tqdm import tqdm


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-samples", type=int)
    args = parser.parse_args()

    # Load config and build components
    config = load_config(args.config)
    components = build_components(config)
    augmentations = config.get("augmentations", {})

    # Load dataset
    loader = DatasetLoader()
    data = loader.load_dataset(config["dataset"], max_samples=args.max_samples)
    queries = [item["question"] for item in data]
    ground_truths = [item["answer"] for item in data]

    # RAG Pipeline: Query → Retrieval → Rerank → Generate → Reflect
    predictions = []

    for i, query in enumerate(tqdm(queries, desc="Processing")):
        ctx = {"idx": i, "generator": components["generator"], "config": config}

        # Stage 1: Query processing
        processed = [query]
        for aug in augmentations.get("query", []):
            processed = aug(processed, ctx)

        # Stage 2: Retrieval (with adaptive decision)
        documents = None
        if components.get("retriever"):
            should_retrieve = True
            for aug in augmentations.get("retrieval", []):
                should_retrieve, _ = aug(processed, ctx)
                if not should_retrieve:
                    break

            if should_retrieve:
                documents = components["retriever"].retrieve(processed[0], k=config.get("top_k", 5)).documents

        # Stage 3: Reranking
        if documents and augmentations.get("rerank"):
            for aug in augmentations["rerank"]:
                documents, _ = aug(processed[0], documents, ctx)

        # Stage 4: Generation
        context_text = "\n".join([d.page_content for d in documents]) if documents else None
        prompt = format_rag_prompt(processed[0], context_text) if context_text else processed[0]
        answer = components["generator"].generate(prompt, context=documents).answer

        # Stage 5: Reflection
        for aug in augmentations.get("reflection", []):
            answer, _ = aug(processed[0], answer, documents, ctx)

        predictions.append(clean_answer(answer))

    # Evaluate
    metrics = components["evaluator"].evaluate(predictions, ground_truths).metrics
    results = {"predictions": predictions, "metrics": metrics}

    # Save and print
    save_results(results, config.get("output_path", "experiments/results"))
    print(f"\nAccuracy: {results['metrics']['substring_match']['accuracy']:.2f}%")


if __name__ == "__main__":
    main()
