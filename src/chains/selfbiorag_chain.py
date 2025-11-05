"""Self-BioRAG chain"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import re
import string
from tqdm import tqdm

from .base import Chain
from langchain_core.documents import Document

from ..generator_models import GeneratorModel, ScoringResult
from ..retriever_models import BiomedicalRetriever, RetrievalResult
from ..retriever_models import FaissJsonRetriever


# Few-shot examples
FEW_SHOT_EXAMPLES = {
    "med_qa": (
        "Given four answer candidates, A, B, C and D, choose the best answer choice.\n\n"
        "###QUESTION: A 22-year-old male marathon runner presents to the office with the complaint of right-sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?\n"
        "Option A: anterior scalene\nOption B: latissimus dorsi\nOption C: pectoralis minor\nOption D: quadratus lumborum\n"
        "Explanation: We refer to Wikipedia articles on medicine for help. Among the options, only pectoralis minor muscle origins from the outer surfaces of the 3rd to 5th ribs.\n"
        "Answer: (C)pectoralis minor\n\n"
        "###QUESTION: A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is\n"
        "Option A: right-on-right sacral torsion\nOption B: left-on-right sacral torsion\nOption C: right unilateral sacral flexion\nOption D: left-on-left sacral torsion\n"
        "Explanation: We refer to Wikipedia articles on medicine for help. The deep sulcus on the left, a posterior ILA on the right, with a negative spring test suggests a right-on-right sacral torsion. All other options have a deep sulcus on the right.\n"
        "Answer: (A)right-on-right sacral torsion\n\n"
        "###QUESTION: A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mm Hg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient's symptoms?\n"
        "Option A: Allergic rhinitis\nOption B: Epstein-Barr virus\nOption C: Mycoplasma pneumonia\nOption D: Rhinovirus\n"
        "Explanation: We refer to Wikipedia articles on medicine for help. The symptoms, especially the headache, suggest that the most likely cause is Rhinovirus. Epstein-Barr virus will cause swollen lymph nodes but there is no palpable cervical adenopathy. Lungs are clear to auscultation suggests it's not Mycoplasma pneumonia.\n"
        "Answer: (D)Rhinovirus\n\n"
        "###QUESTION: A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?\n"
        "Option A: Dopamine\nOption B: Glutamate\nOption C: Norepinephrine\nOption D: Serotonin\n"
        "Explanation: We refer to Wikipedia articles on medicine for help. The patient feels sad and among the options, only Dopamine and Serotonin can help increase positive emotions. Serotonin also affects digestion and metabolism, which can help the patient's decreased appetite and sleep difficulty.\n"
        "Answer: (D)Serotonin\n\n"
    )
}

@dataclass
class SelfBioRAGConfig:
    """Configuration for Self-BioRAG chain"""
    mode: str = "adaptive_retrieval"  # adaptive_retrieval, always_retrieve, no_retrieval
    top_k: int = 5
    use_groundedness: bool = True
    use_utility: bool = True
    use_seqscore: bool = False  # Use sequence score in final scoring
    threshold: float = 0.5  # Threshold for retrieval decision
    closed_domain: bool = True  # For answer aggregation in multiple choice
    w_rel: float = 1.0  # Weight for relevance score
    w_sup: float = 1.0  # Weight for groundedness/support score
    w_use: float = 0.5  # Weight for utility score
    
    # Prompt settings (consistent with other chains)
    use_few_shot: bool = True  # Use few-shot examples (default: enabled)
    dataset_name: str = "med_qa"  # Dataset for few-shot examples
    
    # Model settings
    model_path: str = "dmis-lab/selfbiorag_7b"
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.7
    temperature: float = 0.0
    max_tokens: int = 200
    batch_size: int = 8
    

@dataclass
class SelfBioRAGEvaluationResult:
    """Result from Self-BioRAG evaluation"""
    prediction: str
    ground_truth: str
    is_correct_substring: bool
    is_correct_exact: bool
    retrieval_used: bool
    retrieved_docs: List[str]
    num_docs_used: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "is_correct_substring": self.is_correct_substring,
            "is_correct_exact": self.is_correct_exact,
            "retrieval_used": self.retrieval_used,
            "retrieved_docs": self.retrieved_docs,
            "num_docs_used": self.num_docs_used,
            "metadata": self.metadata
        }


class SelfBioRAGChain(Chain):
    """
    Pipeline:
    1. Generate without retrieval to get [Retrieval] vs [No Retrieval] token logprobs
    2. Decide if retrieval is needed based on token probability ratio
    3. If needed, retrieve documents
    4. Generate with each retrieved document to get reflection token logprobs
    5. Score documents based on [Relevant], [Fully supported], [Utility:X] tokens
    6. Aggregate answers (for closed-domain QA) or return best answer
    """
    
    retriever: BiomedicalRetriever
    generator: GeneratorModel  # Self-BioRAG model that generates reflection tokens
    config: SelfBioRAGConfig
    
    """
    Special token IDs for reflection tokens
    
    Let's say
    x: the problem statement
    y: the answer to the problem
    e: the evidence
    r: the retrieval

    ret_tokens: Decide if retrieval is needed
    rel_tokens: Extract relevance score of e to x
    grd_tokens: Extract groundedness scores of y to e
    ut_tokens: Extract utility scores of y to x
    """

    ret_tokens: Dict[str, int]  # {[Retrieval]: id, [No Retrieval]: id}
    rel_tokens: Dict[str, int]  # {[Relevant]: id, [Irrelevant]: id}
    grd_tokens: Optional[Dict[str, int]]  # {[Fully supported]: id, [Partially supported]: id, [No support]: id}
    ut_tokens: Optional[Dict[str, int]]  # {[Utility:1]: id, ..., [Utility:5]: id}
    
    def __init__(
        self,
        retriever: FaissJsonRetriever,
        generator: GeneratorModel,
        ret_tokens: Dict[str, int],
        rel_tokens: Dict[str, int],
        grd_tokens: Optional[Dict[str, int]] = None,
        ut_tokens: Optional[Dict[str, int]] = None,
        config: Optional[SelfBioRAGConfig] = None,
    ):
        super().__init__()
        self.retriever = retriever
        self.generator = generator
        self.ret_tokens = ret_tokens
        self.rel_tokens = rel_tokens
        self.grd_tokens = grd_tokens
        self.ut_tokens = ut_tokens
        self.config = config or SelfBioRAGConfig()
    
    @property
    def input_keys(self) -> List[str]:
        # Support both "query" (single) and "instructions"/"questions" (batch)
        return ["query", "instructions", "questions"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["results", "metrics"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager=None
    ) -> Dict[str, Any]:
        """Execute Self-BioRAG pipeline on batch of questions"""
        # Handle batch format (instructions/questions/ground_truths)
        if "instructions" in inputs and "questions" in inputs:
            instructions = inputs["instructions"]
            questions = inputs["questions"]
            ground_truths = inputs.get("ground_truths", [])
            metadata_list = inputs.get("metadata", [{}] * len(questions))
            
            # Process each question with progress bar
            results = []
            total = len(questions)
            for i, (instruction, question, gt) in enumerate(tqdm(zip(instructions, questions, ground_truths), total=total, desc="Processing queries")):
                retrieval_query = f"### Instruction:\n{question}\n\n### Response:\n"
                answer_query = f"### Instruction:\n{instruction}\n\n" + f"### Examples:\n{FEW_SHOT_EXAMPLES[self.config.dataset_name]}\n\n" + f"### Question (Always answer with both option alphabets and the named answer):\n{question}\n\n"
                result = self._process_single_query(retrieval_query, answer_query, gt, metadata_list[i] if i < len(metadata_list) else {}, query_idx=i)
                results.append(result)
            
            # Calculate metrics
            metrics = self.calculate_accuracy(results)
            
            return {
                "results": results,
                "metrics": metrics
            }
        elif "query" in inputs:
            # Single query format (for backward compatibility)
            query = inputs["query"]
            gt = inputs.get("ground_truth", "")
            result = self._process_single_query(query, gt, inputs.get("metadata", {}))
            return {
                "results": [result],
                "metrics": self.calculate_accuracy([result])
            }
        else:
            raise ValueError("Must provide either 'query' or both 'instructions' and 'questions'")
    
    def _process_single_query(
        self,
        query: str,
        answer_query: str,
        ground_truth: str = "",
        metadata: Dict[str, Any] = {},
        query_idx: Optional[int] = None
    ) -> SelfBioRAGEvaluationResult:
        """Process a single query through Self-BioRAG pipeline"""
        # Phase 1: Decide retrieval
        needs_retrieval, retrieval_info = self._decide_retrieval(query)
        
        # Log 1: 1st generated answer
        first_answer = retrieval_info.get("no_retrieval_text", "")
        if first_answer:
            first_answer = self._postprocess(first_answer)
        
        if not needs_retrieval:
            answer = first_answer or \
                     self.generator.generate([f"### Question (Always answer with both option alphabets and the named answer):\n{query}\n\n### Response:\n"])[0].get("text", "")
            answer = self._postprocess(answer)
            
            # Evaluate
            is_correct_substring, is_correct_exact = self._evaluate(answer, ground_truth)
            
            return SelfBioRAGEvaluationResult(
                prediction=answer,
                ground_truth=ground_truth,
                is_correct_substring=is_correct_substring,
                is_correct_exact=is_correct_exact,
                retrieval_used=False,
                retrieved_docs=[],
                num_docs_used=0,
                metadata={**metadata, "retrieval_decision": retrieval_info, "mode": "no_retrieval"}
            )
        
        # Phase 2: Retrieve
        from ..retriever_models import query_encode
        query_embedding = query_encode([query])
        retrieval_result = self.retriever.retrieve(query_embedding, k=self.config.top_k)
        
        # Log 2: Retrieved docs
        for i, (doc, score) in enumerate(zip(retrieval_result.documents, retrieval_result.scores), 1):
            doc_preview = doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        
        # Phase 3: Generate + Score
        scored_docs = self._score_and_filter_documents(answer_query, retrieval_result)
        if not scored_docs:
            answer = self.generator.generate([answer_query + "[No Retrieval]"])
            # Extract text from generator result
            answer = answer[0].get("text", "") if answer and isinstance(answer[0], dict) else (answer[0] if answer else "")
            answer = self._postprocess(answer)
            
            # Evaluate
            is_correct_substring, is_correct_exact = self._evaluate(answer, ground_truth)
            
            return SelfBioRAGEvaluationResult(
                prediction=answer,
                ground_truth=ground_truth,
                is_correct_substring=is_correct_substring,
                is_correct_exact=is_correct_exact,
                retrieval_used=True,
                retrieved_docs=[doc.page_content for doc in retrieval_result.documents],
                num_docs_used=0,
                metadata={
                    **metadata,
                    "retrieval_decision": retrieval_info,
                    "mode": "retrieval_failed",
                    "message": "No relevant documents"
                }
            )
        
        
        # Phase 4: Select best answer
        # Log 3: Final RAG query (the prompt with context used for final generation)
        if scored_docs:
            # Get the best doc and show the final RAG query
            best_doc = scored_docs[0][0]
            final_rag_query = self._format_retrieval_prompt(answer_query, best_doc)
            
            # Generate final answer with context
            answer = self.generator.generate([final_rag_query])
            # Extract text from generator result (returns list of dicts)
            answer = answer[0].get("text", "") if answer and isinstance(answer[0], dict) else (answer[0] if answer else "")
            answer = self._postprocess(answer)
            
        else:
            answer = ""
        # Evaluate
        is_correct_substring, is_correct_exact = self._evaluate(answer, ground_truth)
        
        return SelfBioRAGEvaluationResult(
            prediction=answer,
            ground_truth=ground_truth,
            is_correct_substring=is_correct_substring,
            is_correct_exact=is_correct_exact,
            retrieval_used=True,
            retrieved_docs=[doc.page_content for doc, _, _ in scored_docs],
            num_docs_used=len(scored_docs),
            metadata={
                **metadata,
                "retrieval_decision": retrieval_info,
                "mode": "retrieval_success",
                "num_retrieved": len(retrieval_result.documents),
                "num_used": len(scored_docs)
            }
        )
    
    def _decide_retrieval(self, query: str) -> Tuple[bool, Dict]:
        """
        Decide if retrieval is needed by generating and checking 
        [Retrieval] vs [No Retrieval] token probabilities.
        
        The query must be formatted as: "### Instruction:\n{instruction}\n\n### Response:\n"
        to trigger the model to generate [Retrieval] tokens.
        """        
        # Format query for retrieval decision (same format as original Self-BioRAG)
        # If query doesn't already have the format, add it
        if not query.startswith("### Instruction:"):
            formatted_query = f"### Instruction:\n{query}\n\n### Response:\n"
        else:
            formatted_query = query
        
        # Adaptive retrieval: generate and check token logprobs
        result = self.generator.generate_with_logprobs([formatted_query])[0]
        
        # Get probabilities of retrieval tokens from first position
        pred_log_probs = result["logprobs"][0] if result["logprobs"] else {}
        pred_token_ids = result.get("token_ids", [])
        
        # Check if retrieval tokens appear in generated sequence (fallback)
        generated_text = result.get("text", "")
        has_retrieval_token = "[Retrieval]" in generated_text or any(
            self.ret_tokens["[Retrieval]"] == tid for tid in pred_token_ids[:5]
        )
        has_no_retrieval_token = "[No Retrieval]" in generated_text or any(
            self.ret_tokens["[No Retrieval]"] == tid for tid in pred_token_ids[:5]
        )
        
        # Try to get logprobs from first position
        # vLLM returns Logprob objects, need to extract .logprob value
        score_dict = {}
        for tok, token_id in self.ret_tokens.items():
            if token_id in pred_log_probs:
                logprob_obj = pred_log_probs[token_id]
                # Extract logprob value from Logprob object
                if hasattr(logprob_obj, 'logprob'):
                    score_dict[tok] = float(logprob_obj.logprob)
                elif isinstance(logprob_obj, (int, float)):
                    score_dict[tok] = float(logprob_obj)
                else:
                    score_dict[tok] = -100
            else:
                score_dict[tok] = -100
        
        ret_prob = score_dict.get("[Retrieval]", -100)
        no_ret_prob = score_dict.get("[No Retrieval]", -100)
        
        # If tokens not in top-20 logprobs, check if they were generated
        if ret_prob == -100 and has_retrieval_token:
            ret_prob = 0.0  # Token was generated, assign minimal score
        if no_ret_prob == -100 and has_no_retrieval_token:
            no_ret_prob = 0.0
        
        # Decide based on probability ratio
        if ret_prob == -100 and no_ret_prob == -100:
            # No retrieval tokens found - default to no retrieval
            do_retrieve = False
        else:
            # Use logprobs if available, otherwise use presence check
            if ret_prob != -100 and no_ret_prob != -100:
                # Both logprobs available - convert to probabilities and use ratio
                ret_prob_exp = np.exp(ret_prob)
                no_ret_prob_exp = np.exp(no_ret_prob)
                total_prob = ret_prob_exp + no_ret_prob_exp
                if total_prob > 0:
                    ratio = ret_prob_exp / total_prob
                    do_retrieve = ratio > self.config.threshold
                else:
                    do_retrieve = False
            elif ret_prob != -100:
                # Only [Retrieval] found - retrieve if prob > threshold
                ret_prob_exp = np.exp(ret_prob)
                do_retrieve = ret_prob_exp > self.config.threshold
            elif no_ret_prob != -100:
                # Only [No Retrieval] found - don't retrieve
                do_retrieve = False
            else:
                # Fallback: use token presence
                do_retrieve = has_retrieval_token and not has_no_retrieval_token
        
        return do_retrieve, {
            "mode": "adaptive",
            "retrieval_score": ret_prob,
            "no_retrieval_score": no_ret_prob,
            "no_retrieval_text": result.get("text", "")
        }
    
    def _score_and_filter_documents(
        self,
        query: str,
        retrieval_result: RetrievalResult
    ) -> List[Tuple[Document, ScoringResult, str]]:
        """Generate with all documents and score using reflection token logprobs"""
        prompts = [self._format_retrieval_prompt(query, doc) for doc in retrieval_result.documents]
        results = self.generator.generate_with_logprobs(prompts)
        
        scored_docs = [
            (doc, self._compute_scores(result), result["text"])
            for doc, result in zip(retrieval_result.documents, results)
        ]
        
        return sorted(scored_docs, key=lambda x: x[1].final_score, reverse=True)
    
    def _compute_scores(self, result: Dict[str, Any]) -> ScoringResult:
        """Extract reflection token scores from generation result"""
        pred_token_ids = result["token_ids"]
        pred_log_probs = result["logprobs"]
        
        # Sequence score (optional)
        seq_score = 0.0
        if self.config.use_seqscore and pred_token_ids:
            seq_score = result.get("cumulative_logprob", 0) / max(len(pred_token_ids), 1)
        
        # Extract scores from reflection tokens
        relevance_score = self._extract_relevance_score(pred_log_probs)
        ground_score = self._extract_groundedness_score(pred_token_ids, pred_log_probs)
        utility_score = self._extract_utility_score(pred_token_ids, pred_log_probs)
        
        # Calculate final weighted score
        final_score = (
            (np.exp(seq_score) if self.config.use_seqscore else 0) +
            self.config.w_rel * relevance_score +
            self.config.w_sup * ground_score +
            self.config.w_use * utility_score
        )
        
        return ScoringResult(
            relevance_score=relevance_score,
            groundedness_score=ground_score,
            utility_score=utility_score,
            final_score=final_score
        )
    
    def _extract_relevance_score(self, pred_log_probs: List[Dict]) -> float:
        """Extract [Relevant] vs [Irrelevant] score from first token position"""
        if not pred_log_probs:
            return 0.0
        
        first_token_probs = pred_log_probs[0]
        # vLLM returns Logprob objects, need to extract .logprob value
        score_dict = {}
        for tok, token_id in self.rel_tokens.items():
            if token_id in first_token_probs:
                logprob_obj = first_token_probs[token_id]
                if hasattr(logprob_obj, 'logprob'):
                    score_dict[tok] = np.exp(float(logprob_obj.logprob))
                elif isinstance(logprob_obj, (int, float)):
                    score_dict[tok] = np.exp(float(logprob_obj))
                else:
                    score_dict[tok] = 0.0
            else:
                score_dict[tok] = 0.0
        
        total = sum(score_dict.values())
        return score_dict.get("[Relevant]", 0) / total if total > 0 else 0.0
    
    def _extract_groundedness_score(self, pred_token_ids: List[int], pred_log_probs: List[Dict]) -> float:
        """Extract [Fully supported]/[Partially supported] score from generation"""
        if not (self.config.use_groundedness and self.grd_tokens):
            return 0.0
        
        # Find where groundedness token appears
        score_dict = self._find_and_extract_token_probs(
            pred_token_ids, pred_log_probs, self.grd_tokens
        )
        
        if len(score_dict) != 3:
            return 0.0
        
        total = sum(score_dict.values())
        if total == 0:
            return 0.0
        
        # Weighted: 1.0 * fully + 0.5 * partially
        return (
            score_dict.get("[Fully supported]", 0) / total +
            0.5 * score_dict.get("[Partially supported]", 0) / total
        )
    
    def _extract_utility_score(self, pred_token_ids: List[int], pred_log_probs: List[Dict]) -> float:
        """Extract [Utility:1-5] score from generation"""
        if not (self.config.use_utility and self.ut_tokens):
            return 0.0
        
        score_dict = self._find_and_extract_token_probs(
            pred_token_ids, pred_log_probs, self.ut_tokens
        )
        
        if len(score_dict) != 5:
            return 0.0
        
        total = sum(score_dict.values())
        if total == 0:
            return 0.0
        
        # Map to [-1, 1] scale
        weights = [-1, -0.5, 0, 0.5, 1]
        return sum(
            weights[i] * score_dict.get(f"[Utility:{i+1}]", 0) / total
            for i in range(5)
        )
    
    def _find_and_extract_token_probs(
        self, 
        pred_token_ids: List[int], 
        pred_log_probs: List[Dict],
        token_dict: Dict[str, int]
    ) -> Dict[str, float]:
        """Find where special tokens appear in generation and extract their probabilities"""
        for tok_idx, token_id in enumerate(pred_token_ids):
            if token_id in token_dict.values() and tok_idx < len(pred_log_probs):
                token_probs = pred_log_probs[tok_idx]
                # vLLM returns Logprob objects, need to extract .logprob value
                result = {}
                for tok, tok_id in token_dict.items():
                    if tok_id in token_probs:
                        logprob_obj = token_probs[tok_id]
                        if hasattr(logprob_obj, 'logprob'):
                            result[tok] = np.exp(float(logprob_obj.logprob))
                        elif isinstance(logprob_obj, (int, float)):
                            result[tok] = np.exp(float(logprob_obj))
                        else:
                            result[tok] = 0.0
                    else:
                        result[tok] = 0.0
                return result
        return {}
    
    def _generate_with_context(
        self,
        query: str,
        scored_docs: List[Tuple[Document, ScoringResult, str]]
    ) -> str:
        """Select best answer from already-generated results"""
        if self.config.closed_domain:
            return self._aggregate_closed_domain(scored_docs)
        return scored_docs[0][2]  # Return best document's answer
    
    def _aggregate_closed_domain(self, scored_docs: List[Tuple[Document, ScoringResult, str]]) -> str:
        """Aggregate answers by score for multiple-choice QA"""
        answer_scores = {}
        for _, score, text in scored_docs:
            answer = self._postprocess(text)
            answer_scores[answer] = answer_scores.get(answer, 0) + score.final_score
        
        return max(answer_scores.items(), key=lambda x: x[1])[0] if answer_scores else ""
    
    def _format_retrieval_prompt(self, query: str, document: Document) -> str:
        """Format: query[Retrieval]<paragraph>title\ncontent</paragraph>"""
        title = document.metadata.get("title", "")
        content = document.page_content
        paragraph = f"{title}\n{content}" if title else content
        return f"{query}[Retrieval]<paragraph>{paragraph}</paragraph>\n\n### Response:\n"
    
    def _postprocess(self, answer: str) -> str:
        """Remove reflection tokens and special tokens from answer"""
        tokens = [
            "[Retrieval]", "[No Retrieval]", "[Relevant]", "[Irrelevant]",
            "[Fully supported]", "[Partially supported]", "[No support]",
            *[f"[Utility:{i}]" for i in range(1, 6)],
            "</s>", "<|endoftext|>", "\n"
        ]
        for token in tokens:
            answer = answer.replace(token, "")
        return answer.strip()
    
    def _evaluate(self, prediction: str, ground_truth: str) -> Tuple[bool, bool]:
        """Evaluate prediction against ground truth"""
        def normalize_answer(s: str) -> str:
            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)
            def white_space_fix(text):
                return ' '.join(text.split())
            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)
            def lower(text):
                return text.lower()
            return white_space_fix(remove_articles(remove_punc(lower(s))))
        
        # Substring match
        is_correct_substring = ground_truth.lower() in prediction.lower()
        
        # Normalized exact match
        norm_pred = normalize_answer(prediction)
        norm_gt = normalize_answer(ground_truth)
        is_correct_exact = norm_pred == norm_gt
        
        return is_correct_substring, is_correct_exact
    
    def calculate_accuracy(self, results: List[SelfBioRAGEvaluationResult]) -> Dict[str, Any]:
        """Calculate accuracy metrics"""
        if not results:
            return {
                "substring_match": {"accuracy": 0.0, "correct": 0, "total": 0},
                "exact_match": {"accuracy": 0.0, "correct": 0, "total": 0}
            }
        
        substring_correct = sum(1 for r in results if r.is_correct_substring)
        exact_correct = sum(1 for r in results if r.is_correct_exact)
        total = len(results)
        
        # Calculate average number of documents used
        avg_docs = sum(r.num_docs_used for r in results) / total if total > 0 else 0
        retrieval_used_count = sum(1 for r in results if r.retrieval_used)
        
        return {
            "substring_match": {
                "accuracy": 100 * substring_correct / total,
                "correct": substring_correct,
                "total": total
            },
            "exact_match": {
                "accuracy": 100 * exact_correct / total,
                "correct": exact_correct,
                "total": total
            },
            "retrieval_stats": {
                "avg_docs_per_question": avg_docs,
                "questions_with_retrieval": retrieval_used_count,
                "retrieval_rate": 100 * retrieval_used_count / total if total > 0 else 0
            },
            "primary_method": "substring_match"
        }
