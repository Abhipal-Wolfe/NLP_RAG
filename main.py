"""RAG Evaluation Framework - Domain logic visible in main()"""

from src.core.config_loader import load_config, build_components
from src.data import DatasetLoader
from src.prompts import format_question_with_options, clean_answer
from src.utils import save_results
from src.augmentations.loader import load_augmentations, print_loaded_augmentations
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
    
    # Load and instantiate augmentations from config
    augmentations = load_augmentations(config)
    if config.get("verbose", False):
        print_loaded_augmentations(augmentations)
    verbose = config.get("verbose", False)

    # Load dataset
    loader = DatasetLoader()
    # Use command-line arg if provided, otherwise use config value
    max_samples = args.max_samples if args.max_samples is not None else config.get("max_samples")
    data = loader.load_dataset(config["dataset"], max_samples=max_samples)
    data = data[2:10]
    
    # Format questions with options (moved to prompts.py)
    queries = [format_question_with_options(item) for item in data]
    ground_truths = data

    # RAG Pipeline: Query → Retrieval → Rerank → Generate → Reflect
    predictions = []
    
    # Check if we can use batch processing (no retrieval, no augmentations)
    has_augmentations = any(augmentations.get(k) for k in ["query", "retrieval", "rerank", "reflection"])
    has_retriever = components.get("retriever") is not None
    use_batch = not has_augmentations and not has_retriever
    
    # Print configuration if verbose
    if verbose:
        print("\n" + "="*100)
        print("RAG PIPELINE CONFIGURATION")
        print("="*100)
        print(f"Dataset: {config['dataset']}")
        print(f"Samples: {len(queries)}")
        print(f"Model: {config['generator']['model_path']}")
        print(f"Max Tokens: {config['generator'].get('max_tokens', 200)}")
        print(f"\nPIPELINE STAGES:")
        print(f"  1. Query Processing: {len(augmentations.get('query', []))} augmentation(s)")
        print(f"  2. Retrieval: {'Enabled' if has_retriever else 'Disabled'}")
        if has_retriever:
            print(f"     - Top-K: {config.get('retriever', {}).get('top_k', 5)}")
            print(f"     - Adaptive: {len(augmentations.get('retrieval', []))} augmentation(s)")
        print(f"  3. Reranking: {len(augmentations.get('rerank', []))} augmentation(s)")
        print(f"  4. Generation: {len(augmentations.get('generation', []))} augmentation(s)")
        print(f"  5. Reflection: {len(augmentations.get('reflection', []))} augmentation(s)")
        print("="*100 + "\n")
    
    if use_batch:
        # Fast path: Batch process all queries at once
        print(f"Using batch inference for {len(queries)} queries...")
        batch_size = config.get("batch_size", 32)
        
        for batch_start in tqdm(range(0, len(queries), batch_size), desc="Batches"):
            batch_end = min(batch_start + batch_size, len(queries))
            batch_queries = queries[batch_start:batch_end]
            
            if verbose and batch_start == 0:
                print("\n[QUERY 0]")
                print(batch_queries[0])
                print(f"\nGround truth: {ground_truths[0].get('answer_idx')}")
                
                # Show the formatted prompt (same as what generator does internally)
                print("\n[PROMPT 0]")
                formatted_prompt = components["generator"]._format_prompt(batch_queries[0], None)
                print(formatted_prompt)
            
            # Batch generate
            batch_results = components["generator"].generate_batch(batch_queries)
            batch_predictions = [clean_answer(result.answer) for result in batch_results]
            
            if verbose and batch_start == 0:
                print("\n[OUTPUT 0]")
                print(batch_results[0].answer)
                sample_data = ground_truths[0]
                correct = batch_predictions[0] == sample_data.get('answer_idx')
                print(f"\nPredicted: {batch_predictions[0]} | Expected: {sample_data.get('answer_idx')} | {'✓' if correct else '✗'}\n")
            
            predictions.extend(batch_predictions)
    else:
        # Standard path: Process one by one with retrieval/augmentations
        for i, query in enumerate(tqdm(queries, desc="Processing")):
            ctx = {"idx": i, "generator": components["generator"], "config": config, "verbose": verbose}

            if verbose and i == 0:
                print(f"\n[QUERY {i}]")
                print(query)
                print(f"\nGround truth: {ground_truths[i].get('answer_idx')}")

            # Stage 1: Query processing
            processed = [query]
            if verbose and i == 0 and augmentations.get("query"):
                print(f"\n[STAGE 1: QUERY PROCESSING]")
                print(f"Applying {len(augmentations.get('query', []))} query augmentation(s)...")
            
            for aug_idx, aug in enumerate(augmentations.get("query", [])):
                old_processed = processed.copy()
                processed = aug(processed, {}, ctx)  # (queries, metadata, pipeline_context)
                if verbose and i == 0:
                    aug_name = getattr(aug, '__name__', str(aug))
                    print(f"\nAugmentation {aug_idx+1}: {aug_name}")
                    if processed != old_processed:
                        print(f"  Generated {len(processed)} queries")
                        for q_idx, q in enumerate(processed[:3]):  # Show first 3
                            print(f"  Query {q_idx+1}: {q[:100]}...")
                    else:
                        print(f"  No changes")

            # Stage 2: Retrieval (with adaptive decision)
            documents = None
            if components.get("retriever"):
                should_retrieve = True
                has_callable_retrieval_augs = False
                
                if verbose and i == 0 and augmentations.get("retrieval"):
                    print(f"\n[STAGE 2: ADAPTIVE RETRIEVAL DECISION]")
                
                for aug_idx, aug in enumerate(augmentations.get("retrieval", [])):
                    has_callable_retrieval_augs = True
                    should_retrieve, aug_metadata = aug(processed, {}, ctx)
                    
                    if verbose and i == 0:
                        aug_name = getattr(aug, '__name__', str(aug))
                        print(f"\nAugmentation {aug_idx+1}: {aug_name}")
                        print(f"  Final Decision: {'✓ RETRIEVE' if should_retrieve else '✗ NO RETRIEVAL'}")
                        
                        if aug_metadata and isinstance(aug_metadata, dict):
                            # Show summary of decision
                            if 'retrieval_prob' in aug_metadata and 'no_retrieval_prob' in aug_metadata:
                                print(f"  [Retrieval] probability: {aug_metadata['retrieval_prob']:.4f}")
                                print(f"  [No Retrieval] probability: {aug_metadata['no_retrieval_prob']:.4f}")
                                print(f"  Threshold: {aug_metadata.get('threshold', 0.5):.2f}")
                    
                    if not should_retrieve:
                        break
                
                if verbose and i == 0:
                    if not augmentations.get("retrieval") or not has_callable_retrieval_augs:
                        print(f"\n[STAGE 2: RETRIEVAL - Always On]")
                    else:
                        print(f"Final Decision: {'RETRIEVING' if should_retrieve else 'SKIPPING RETRIEVAL'}")

                if should_retrieve:
                    # Get top_k from retriever config or global config
                    top_k = config.get("retriever", {}).get("top_k", config.get("top_k", 5))
                    
                    # DMQR-RAG: Multi-query retrieval with passage merging
                    if len(processed) > 1:
                        # Multiple sub-queries: retrieve for each and merge
                        if verbose and i == 0:
                            print(f"\n" + "="*100)
                            print("MULTI-QUERY RETRIEVAL (DMQR-RAG)")
                            print("="*100)
                            print(f"Number of sub-queries: {len(processed)}")
                            print(f"Top-K per query: {top_k}")
                            print(f"\n--- GENERATED SUB-QUERIES (FULL TEXT) ---")
                            for q_idx, sq in enumerate(processed):
                                print(f"\n[Sub-query {q_idx + 1}]")
                                print(sq)
                        
                        all_documents = []
                        seen_content = set()  # For deduplication
                        
                        for q_idx, sub_query in enumerate(processed):
                            sub_docs = components["retriever"].retrieve(sub_query, k=top_k).documents
                            
                            if verbose and i == 0:
                                print(f"\n" + "-"*80)
                                print(f"RETRIEVAL FOR SUB-QUERY {q_idx + 1}")
                                print("-"*80)
                                print(f"Query: {sub_query}")
                                print(f"Documents retrieved: {len(sub_docs)}")
                                
                                for doc_idx, doc in enumerate(sub_docs):
                                    print(f"\n  [Doc {doc_idx + 1}] Score: {doc.metadata.get('score', 'N/A'):.4f}")
                                    print(f"  Content preview: {doc.page_content[:200]}...")
                            
                            # Add unique documents (deduplicate by content)
                            added_count = 0
                            for doc in sub_docs:
                                content_hash = hash(doc.page_content[:500])  # Hash first 500 chars
                                if content_hash not in seen_content:
                                    seen_content.add(content_hash)
                                    doc.metadata['source_query'] = sub_query
                                    doc.metadata['query_idx'] = q_idx
                                    all_documents.append(doc)
                                    added_count += 1
                            
                            if verbose and i == 0:
                                if added_count < len(sub_docs):
                                    print(f"\n  -> Added {added_count}/{len(sub_docs)} (duplicates removed)")
                                else:
                                    print(f"\n  -> Added {added_count} unique documents")
                        
                        documents = all_documents
                        
                        if verbose and i == 0:
                            print(f"\n" + "="*100)
                            print(f"MERGED DOCUMENT SET: {len(documents)} unique documents")
                            print("="*100)
                    else:
                        # Single query: standard retrieval
                        if verbose and i == 0:
                            print(f"  Retrieving top_k={top_k} documents...")
                        
                        documents = components["retriever"].retrieve(processed[0], k=top_k).documents
                    
                    if verbose and i == 0:
                        if documents:
                            print(f"\n[RETRIEVED {len(documents)} DOCUMENTS]")
                            for j, doc in enumerate(documents[:5]):  # Show first 5
                                print(f"\n--- Document {j+1} ---")
                                print(f"Score: {doc.metadata.get('score', 'N/A')}")
                                if doc.metadata.get('source_query'):
                                    print(f"Source Query: {doc.metadata.get('source_query')}...")
                                print(f"Content (first 300 chars):")
                                print(f"{doc.page_content[:300]}...")
                            if len(documents) > 5:
                                print(f"\n... and {len(documents) - 5} more documents")
                        else:
                            print("[WARNING] No documents retrieved!")
                else:
                    if verbose and i == 0:
                        print("  Skipping retrieval (model decided not needed)")

            # Stage 3: Reranking
            if documents and augmentations.get("rerank"):
                if verbose and i == 0:
                    print(f"\n[STAGE 3: RERANKING]")
                    print(f"Documents before reranking: {len(documents)}")
                
                for aug_idx, aug in enumerate(augmentations["rerank"]):
                    if verbose and i == 0:
                        aug_name = getattr(aug, '__name__', str(aug))
                        print(f"\nAugmentation {aug_idx+1}: {aug_name}")
                        # Show documents before reranking
                        print("  Before:")
                        for j, doc in enumerate(documents[:5]):
                            print(f"    Doc {j+1} - Score: {doc.metadata.get('score', 'N/A')}")
                    
                    documents, aug_metadata = aug(processed[0], documents, {}, ctx)
                    
                    if verbose and i == 0:
                        print("  After:")
                        for j, doc in enumerate(documents[:5]):
                            # Show reranking scores if available
                            score_info = f"Score: {doc.metadata.get('score', 'N/A')}"
                            if doc.metadata.get('rerank_score'):
                                score_info += f" | Rerank: {doc.metadata.get('rerank_score'):.4f}"
                            if doc.metadata.get('relevance_score'):
                                score_info += f" | Rel: {doc.metadata.get('relevance_score'):.3f}"
                            if doc.metadata.get('support_score'):
                                score_info += f" | Sup: {doc.metadata.get('support_score'):.3f}"
                            if doc.metadata.get('utility_score'):
                                score_info += f" | Util: {doc.metadata.get('utility_score'):.3f}"
                            print(f"    Doc {j+1} - {score_info}")
                        
                        if aug_metadata and isinstance(aug_metadata, dict):
                            print(f"\n  --- RERANKING SUMMARY ---")
                            if 'original_count' in aug_metadata:
                                print(f"  Documents evaluated: {aug_metadata['original_count']}")
                            if 'top_k' in aug_metadata:
                                print(f"  Documents kept: {aug_metadata['top_k']}")
                            if 'avg_relevance' in aug_metadata:
                                print(f"  Average Relevance: {aug_metadata['avg_relevance']:.4f}")
                            if 'avg_groundedness' in aug_metadata:
                                print(f"  Average Groundedness: {aug_metadata['avg_groundedness']:.4f}")
                            if 'avg_utility' in aug_metadata:
                                print(f"  Average Utility: {aug_metadata['avg_utility']:.4f}")
                
                if verbose and i == 0:
                    print(f"\nDocuments after reranking: {len(documents)}")

            # Stage 4: Generation
            if verbose and i == 0:
                print(f"Context: {'WITH DOCUMENTS' if documents else 'NO DOCUMENTS'}")
                print(f"\n[PROMPT]")
                # Show the formatted prompt (generator will format it internally)
                formatted_prompt = components["generator"]._format_prompt(processed[0], documents)
                print(formatted_prompt)
            
            # Pass query and documents to generator (it will format internally)
            generation_result = components["generator"].generate(processed[0], context=documents)
            answer = generation_result.answer

            # Stage 5: Reflection
            if augmentations.get("reflection"):
                if verbose and i == 0:
                    print(f"\n[STAGE 5: SELF-REFLECTION]")
                    print(f"Initial answer: {answer[:200]}...")
                
                # Add generation result to context for reflection to access raw tokens
                ctx["generation_result"] = generation_result
                
                for aug_idx, aug in enumerate(augmentations["reflection"]):
                    if verbose and i == 0:
                        aug_name = getattr(aug, '__name__', str(aug))
                        print(f"\nAugmentation {aug_idx+1}: {aug_name}")
                    
                    old_answer = answer
                    answer, aug_metadata = aug(processed[0], answer, documents, {}, ctx)
                    
                    if verbose and i == 0:
                        if aug_metadata and isinstance(aug_metadata, dict):
                            # Show reflection metadata (iterations, scores, decisions)
                            if 'iterations' in aug_metadata:
                                print(f"  Iterations: {aug_metadata['iterations']}")
                            if 'utility_scores' in aug_metadata:
                                print(f"  Utility scores: {aug_metadata['utility_scores']}")
                            if 'refined' in aug_metadata:
                                print(f"  Refined: {aug_metadata['refined']}")
                            if 'final_utility' in aug_metadata:
                                print(f"  Final utility: {aug_metadata['final_utility']:.2f}")
                            if 'stopped_reason' in aug_metadata:
                                print(f"  Stopped: {aug_metadata['stopped_reason']}")
                            # Print any other metadata
                            for key, value in aug_metadata.items():
                                if key not in ['iterations', 'utility_scores', 'refined', 'final_utility', 'stopped_reason']:
                                    print(f"  {key}: {value}")
                        
                        if answer != old_answer:
                            print(f"  Answer changed after reflection")
                            print(f"  New answer: {answer[:200]}...")
                        else:
                            print(f"  Answer unchanged")

            cleaned = clean_answer(answer)
            predictions.append(cleaned)
            
            if verbose and i == 0:
                print(f"\n" + "="*100)
                print(f"[FINAL OUTPUT - Query {i}]")
                print("="*100)
                
                print("\n--- RAW MODEL OUTPUT ---")
                print(answer)
                
                print("\n--- CLEANED PREDICTION ---")
                print(f"Extracted Answer: {cleaned}")
                
                print("\n--- EVALUATION ---")
                sample_data = ground_truths[i]
                correct = cleaned == sample_data.get('answer_idx')
                status = "CORRECT" if correct else "INCORRECT"
                print(f"Status: {status}")
                print(f"Predicted: {cleaned}")
                print(f"Expected:  {sample_data.get('answer_idx')} ({sample_data.get('answer_text')})")
                
                if "options" in sample_data and not correct:
                    pred_text = sample_data["options"].get(cleaned, "N/A")
                    print(f"\nPredicted was: {pred_text}")
                    print(f"Should have been: {sample_data.get('answer_text')}")
                
                print("\n--- PIPELINE SUMMARY ---")
                print(f"Query Augmentations: {len(augmentations.get('query', []))}")
                print(f"Retrieved: {len(documents) if documents else 0} documents")
                print(f"Reranking: {'Applied' if augmentations.get('rerank') else 'None'}")
                print(f"Reflection: {'Applied' if augmentations.get('reflection') else 'None'}")
                print("="*100 + "\n")

    # Evaluate
    metrics = components["evaluator"].evaluate(predictions, ground_truths).metrics
    results = {"predictions": predictions, "metrics": metrics}

    # Save and print
    save_results(results, config.get("output_path", "experiments/results"))
    
    # Print results
    if "choice_match" in metrics:
        acc = metrics['choice_match']['accuracy']
        correct = metrics['choice_match']['correct']
        total = metrics['choice_match']['total']
        print(f"\nAccuracy: {acc:.2f}% ({correct}/{total})")
    elif "substring_match" in metrics:
        acc = metrics['substring_match']['accuracy']
        print(f"\nAccuracy: {acc:.2f}%")
    else:
        print(f"\nMetrics: {metrics}")
    
    print(f"Saved: {config.get('output_path', 'experiments/results')}\n")


if __name__ == "__main__":
    main()
