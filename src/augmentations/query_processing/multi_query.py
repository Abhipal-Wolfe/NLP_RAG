"""
Multi-Query Decomposition RAG (DMQR-RAG)

Based on: "DMQR-RAG: Diverse Multi-Query Rewriting for Retrieval-Augmented Generation"
Paper: https://arxiv.org/pdf/2411.13154

3-Strategy Pipeline:
1. GQR (General Query Rewrite) - Clean, well-formed version
2. KWR (Keyword Rewrite) - Keyword/entity heavy version  
3. PAR (Pseudo-Answer Rewrite) - Expand with guessed answer

Pipeline:
1. Call LLM 3 times (once per strategy) to get 3 rewritten queries
2. Run retrieval for each query (+ original = 4 queries total)
3. Merge retrieved documents (deduplicate)
4. Feed top docs + original question to LLM to answer
"""

from typing import List, Dict, Any
from ...prompts import (
    format_query_rewrite_gqr,
    format_query_rewrite_kwr, 
    format_query_rewrite_par,
    QUERY_REWRITE_STRATEGIES
)


def generate_rewritten_queries(
    query: str,
    generator,
    strategies: List[str] = ["gqr", "kwr", "par"],
    include_original: bool = True,
    verbose: bool = False
) -> List[str]:
    """
    Generate diverse queries using DMQR-RAG 3-strategy approach.
    
    Strategies:
    - GQR: General Query Rewrite - clean, well-formed version
    - KWR: Keyword Rewrite - keyword/entity heavy for search
    - PAR: Pseudo-Answer Rewrite - expand with guessed answer
    
    Args:
        query: Original question
        generator: LLM generator
        strategies: Which strategies to use (default: all 3)
        include_original: Whether to include original query
        verbose: Print debug info
        
    Returns:
        List of queries [original (optional), gqr, kwr, par]
    """
    queries = []
    
    # Strip out "## Options:" section from query for rewriting
    # The options confuse the LLM into answering instead of rewriting
    query_for_rewrite = query
    if "## Options:" in query:
        query_for_rewrite = query.split("## Options:")[0].strip()
    
    # Optionally include original query first
    if include_original:
        queries.append(query)
    
    if verbose:
        print(f"\n  --- DMQR-RAG QUERY REWRITING ---")
        print(f"  Original (full): {query[:200]}...")
        print(f"  Query for rewrite (no options): {query_for_rewrite[:200]}...")
        print(f"  Strategies: {', '.join(s.upper() for s in strategies)}")
    
    # Strategy -> prompt function mapping
    strategy_prompts = {
        "gqr": format_query_rewrite_gqr,
        "kwr": format_query_rewrite_kwr,
        "par": format_query_rewrite_par,
    }
    
    for strategy in strategies:
        if strategy not in strategy_prompts:
            continue
            
        prompt_fn = strategy_prompts[strategy]
        strategy_name = QUERY_REWRITE_STRATEGIES[strategy][0]
        prompt = prompt_fn(query_for_rewrite)  # Use query WITHOUT options
        
        if verbose:
            print(f"\n  " + "="*70)
            print(f"  [{strategy.upper()}] {strategy_name}")
            print(f"  " + "="*70)
            print(f"  --- PROMPT ---")
            print(f"  {prompt}")
            print(f"  " + "-"*70)
        
        try:
            # Use raw=True to bypass few-shot formatting (avoid "Answer:" pattern)
            result = generator.generate(prompt, context=None, raw=True)
            raw_output = result.answer
            rewritten = raw_output.strip()
            
            if verbose:
                print(f"  --- RAW LLM OUTPUT ---")
                print(f"  {repr(raw_output)}")
                print(f"  " + "-"*70)
            
            # Clean up - take first line only, remove quotes
            rewritten = rewritten.split('\n')[0].strip()
            if rewritten.startswith('"') and rewritten.endswith('"'):
                rewritten = rewritten[1:-1]
            if rewritten.startswith("'") and rewritten.endswith("'"):
                rewritten = rewritten[1:-1]
            
            # Skip if it looks like an answer (starts with "Answer:" or single letter)
            is_answer = (
                rewritten.lower().startswith("answer:") or
                (len(rewritten) <= 3 and rewritten[0].upper() in "ABCD")
            )
            
            if verbose:
                print(f"  --- CLEANED OUTPUT ---")
                print(f"  {rewritten}")
                print(f"  Is answer (skipped): {is_answer}")
            
            # Only add if valid, different, and not an answer
            if rewritten and len(rewritten) > 10 and rewritten not in queries and not is_answer:
                queries.append(rewritten)
                if verbose:
                    print(f"  STATUS: ADDED to query list")
            else:
                if verbose:
                    reason = "is answer" if is_answer else "empty/short/duplicate"
                    print(f"  STATUS: SKIPPED ({reason})")
                    
        except Exception as e:
            if verbose:
                print(f"  STATUS: FAILED - {e}")
    
    return queries


def multi_query_augmentation(
    queries: List[str],
    metadata: Dict[str, Any],
    pipeline_context: Dict[str, Any],
    max_sub_queries: int = 4
) -> List[str]:
    """
    DMQR-RAG: Generate diverse queries using 3-strategy rewriting.
    
    Args:
        queries: Input queries (typically 1 query)
        metadata: Query metadata
        pipeline_context: Context with generator, config, etc.
        max_sub_queries: Max total queries (original + rewrites)
        
    Returns:
        List of queries [original, gqr, kwr, par] (up to max_sub_queries)
        
    Example:
        Input: "What vitamin deficiency causes confusion, ataxia, ophthalmoplegia?"
        Output: [
            "What vitamin deficiency causes confusion, ataxia, ophthalmoplegia?",  # Original
            "What vitamin deficiency causes the triad of confusion, ataxia, and ophthalmoplegia?",  # GQR
            "vitamin deficiency confusion ataxia ophthalmoplegia Wernicke",  # KWR
            "Thiamine B1 deficiency Wernicke encephalopathy symptoms treatment"  # PAR
        ]
    """
    generator = pipeline_context.get("generator")
    config = pipeline_context.get("config", {})
    verbose = config.get("verbose", False)
    idx = pipeline_context.get("idx", 0)
    
    if not generator:
        return queries
    
    query = queries[0] if queries else ""
    if not query:
        return queries
    
    # Use all 3 strategies: GQR + KWR + PAR
    # Don't include original query - only use rewrites for retrieval
    all_strategies = ["gqr", "kwr", "par"]
    num_rewrites = min(max_sub_queries, 3)
    strategies = all_strategies[:num_rewrites]
    
    # Generate rewritten queries (no original)
    sub_queries = generate_rewritten_queries(
        query, generator,
        strategies=strategies,
        include_original=False,  # Don't use original for retrieval
        verbose=(verbose and idx == 0)
    )
    
    # Limit to max
    sub_queries = sub_queries[:max_sub_queries]
    
    if verbose and idx == 0:
        print(f"\n  Total queries for retrieval: {len(sub_queries)}")
        for i, q in enumerate(sub_queries):
            label = "Original" if i == 0 else strategies[i-1].upper() if i-1 < len(strategies) else f"Query {i}"
            print(f"    [{label}] {q[:80]}{'...' if len(q) > 80 else ''}")
    
    return sub_queries


def create_multi_query_augmentation(max_sub_queries: int = 4):
    """
    Factory function to create DMQR-RAG multi-query augmentation.
    
    Args:
        max_sub_queries: Total queries (original + rewrites, max 4)
        
    Returns:
        Augmentation function
    """
    def augmentation(queries, metadata, pipeline_context):
        return multi_query_augmentation(
            queries, metadata, pipeline_context,
            max_sub_queries=max_sub_queries
        )
    
    augmentation.__name__ = f"dmqr_rag_{max_sub_queries}queries"
    return augmentation
