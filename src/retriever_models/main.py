import os
import json
import argparse
import logging
import gc
from tqdm import tqdm
import torch
import query_encode as qe
import retrieve as rt
import rerank as rr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--embeddings_dir', default='embeddings', help='Embeddings directory')
    parser.add_argument('-a', '--articles_dir', default='articles', help='Articles directory')
    parser.add_argument('-i', '--input_path', default='input/all_biomedical_instruction.json', help='Input file path')
    parser.add_argument('-o', '--output_path', default='output/medcpt_top10_evidence.jsonl', help='Output JSONL path')
    parser.add_argument('-spc', '--use_spacy', default='True', help='Use scispacy [SEP] insertion')
    parser.add_argument('-k', '--topk', type=int, default=10, help='Top-k chunks per source')
    parser.add_argument('-b', '--batch_size', type=int, default=256, help='Query batch size for retrieval')
    parser.add_argument('-m', '--model', choices=['medcpt', 'bert'], required=True, help='Reranker model')
    args = parser.parse_args()

    embeddings_dir = args.embeddings_dir
    articles_dir = args.articles_dir
    input_path = args.input_path
    output_path = args.output_path
    use_spacy = args.use_spacy.lower() == 'true'
    topk = args.topk
    batch_size = args.batch_size
    reranker_type = args.model  # rename to avoid overwriting model object

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logging.info("Loading queries...Yo")
    input_list = qe.query_preprocess(input_path, use_spacy=use_spacy)
    logging.info(f"Total queries: {len(input_list)}")

    # Load reranker
    if reranker_type == 'medcpt':
        logging.info("Loading MedCPT reranker model...")
        tokenizer = AutoTokenizer.from_pretrained("ncbi/MedCPT-Cross-Encoder")
        model = AutoModelForSequenceClassification.from_pretrained("ncbi/MedCPT-Cross-Encoder").to(device).eval()
    elif reranker_type == 'bert':
        logging.info("Loading Sentence-BERT model...")
        model = SentenceTransformer("sentence-transformers/all-distilroberta-v1", device=device)
    print(model)

    # Preload indexes
    logging.info("Loading PubMed and PMC indexes once...")
    pubmed_index, pubmed_mapping = rt.pubmed_index_create(os.path.join(embeddings_dir, "pubmed"))
    pmc_index, pmc_mapping = rt.pmc_index_create(os.path.join(embeddings_dir, "pmc"))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load input as JSON objects (list of dicts)
    with open(input_path, 'r', encoding='utf-8') as f:
        input_objs = json.load(f)

    with open(output_path, 'w', encoding='utf-8') as f_out:
        for start_idx in range(0, len(input_list), batch_size):
            batch_objs = input_objs[start_idx:start_idx + batch_size]
            batch_queries = input_list[start_idx:start_idx + batch_size]
            logging.info(f"Processing queries {start_idx}–{start_idx + len(batch_queries) - 1}")

            # Encode queries
            if reranker_type == 'medcpt':
                q_embs = qe.query_encode(batch_queries)
            elif reranker_type == 'bert':
                q_embs = model.encode(
                    batch_queries,
                    batch_size=batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )

            # Retrieve from PubMed and PMC
            pubmed_topk_all = rt.retrieve_topk(pubmed_index, pubmed_mapping, q_embs, topk=topk,
                                               source_dir=articles_dir, source_type='pubmed')
            pmc_topk_all = rt.retrieve_topk(pmc_index, pmc_mapping, q_embs, topk=topk,
                                            source_dir=articles_dir, source_type='pmc')

            # Combine evidence
            q_evidence_pairs_all = []
            combined_evidences_all = []

            for q_text, pubmed_evs, pmc_evs in zip(batch_queries, pubmed_topk_all, pmc_topk_all):
                q_pairs, combined = rr.combine_query_evidence([q_text], [pubmed_evs], [pmc_evs])
                q_evidence_pairs_all.extend(q_pairs)
                combined_evidences_all.extend(combined)

            # Rerank
            reranked_all = []
            if reranker_type == 'medcpt':
                batch_size_rerank = 8
                for start_r in range(0, len(q_evidence_pairs_all), batch_size_rerank):
                    batch_q_pairs = q_evidence_pairs_all[start_r:start_r + batch_size_rerank]
                    batch_combined = combined_evidences_all[start_r:start_r + batch_size_rerank]

                    reranked_batch = rr.rerank(
                        batch_q_pairs,
                        batch_combined,
                        tokenizer=tokenizer,
                        model=model,
                        device=device
                    )
                    reranked_all.extend(reranked_batch)
                    torch.cuda.empty_cache()
                    gc.collect()
            
            elif reranker_type == 'bert':
                batch_size_rerank = 8  # adjust as per GPU memory
                for start_r in range(0, len(q_evidence_pairs_all), batch_size_rerank):
                    batch_q_pairs = q_evidence_pairs_all[start_r:start_r + batch_size_rerank]
                    batch_combined = combined_evidences_all[start_r:start_r + batch_size_rerank]

                    reranked_batch = rr.rerank_sbert(
                        batch_q_pairs,
                        batch_combined,
                        sbert_model=model,  # keep argument names consistent with rerank_sbert
                        device=device,
                        topk=topk
                    )
                    reranked_all.extend(reranked_batch)
                    torch.cuda.empty_cache()
                    gc.collect()

            # Write output: add evidence key to original objects
            for obj, reranked in zip(batch_objs, reranked_all):
                obj['evidence'] = reranked[:topk]
                f_out.write(json.dumps(obj) + "\n")
            f_out.flush()

            # Cleanup
            del q_embs, pubmed_topk_all, pmc_topk_all, q_evidence_pairs_all, combined_evidences_all, reranked_all
            torch.cuda.empty_cache()
            gc.collect()

    del pubmed_index, pmc_index
    gc.collect()
    logging.info(f"Finished all queries. Results saved to {output_path}")


if __name__ == "__main__":
    main()