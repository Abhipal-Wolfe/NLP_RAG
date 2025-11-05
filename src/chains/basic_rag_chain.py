"""Basic RAG Chain - Simple Retrieval-Augmented Generation

This chain implements a straightforward RAG approach:
1. Retrieve relevant documents for the question
2. Add documents to the prompt context
3. Generate answer using LLM
4. Evaluate results
"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import re
import string

from tqdm import tqdm
from .base import Chain
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from ..retriever_models import FaissJsonRetriever, query_encode

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
class BasicRAGConfig:
    """Configuration for Basic RAG chain"""
    # Model settings
    model_path: str = "dmis-lab/selfbiorag_7b"
    dtype: str = "half"
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 4096  # Increased from 2048 to support few-shot + retrieval context
    enforce_eager: bool = True
    
    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 200
    top_p: float = 1.0
    
    # RAG settings
    top_k: int = 1  # Number of documents to retrieve
    use_reranking: bool = False
    
    # Prompt settings
    use_few_shot: bool = True  # Changed to True by default
    dataset_name: str = "med_qa"
    
    # Evaluation settings
    use_substring_match: bool = True
    batch_size: int = 8
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "model_path": self.model_path,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "max_model_len": self.max_model_len,
            "enforce_eager": self.enforce_eager,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "use_reranking": self.use_reranking,
            "use_few_shot": self.use_few_shot,
            "dataset_name": self.dataset_name,
            "use_substring_match": self.use_substring_match,
            "batch_size": self.batch_size
        }


@dataclass
class RAGEvaluationResult:
    """Result from Basic RAG evaluation"""
    prediction: str
    ground_truth: str
    is_correct_substring: bool
    is_correct_exact: bool
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
            "retrieved_docs": self.retrieved_docs,
            "num_docs_used": self.num_docs_used,
            "metadata": self.metadata
        }


class RAGPromptFormatter:
    """Formats prompts with retrieved documents"""

    def __init__(self, config: BasicRAGConfig):
        self.config = config
    
    def format_prompt_with_context(
        self,
        instruction: str,
        question: str,
        documents: List[str]
    ) -> str:
        """
        Create RAG prompt with retrieved documents.
        
        Format:
        ### Instruction: {instruction}
        
        ### Context:
        [1] {doc1}
        [2] {doc2}
        ...
        
        ### Question: {question}
        
        ### Response:
        """
        # Format context from retrieved documents
        if documents:
            context_parts = [f"[{i+1}] {doc}" for i, doc in enumerate(documents)]
            context = "\n".join(context_parts)
            
            prompt = f"### Instruction:\n{instruction}\n\n" + f"### Examples:\n{FEW_SHOT_EXAMPLES[self.config.dataset_name]}\n\n" + f"### Question (Always answer with both option alphabets and the named answer):\n{question}\n\n ### Context:\n{context}\n\n ### Response:\n"
        else:
            # No documents retrieved - fall back to no-context prompt
            prompt = f"### Instruction:\n{instruction}\n\n" + f"###Examples:\n{FEW_SHOT_EXAMPLES[self.config.dataset_name]}\n\n" + f"### Question (Always answer with both option alphabets and the named answer):\n{question}\n\n### Response:\n"
        return prompt


class Evaluator:
    """Handles evaluation metrics"""
    
    @staticmethod
    def normalize_answer(s: str) -> str:
        """Normalize answer for strict comparison"""
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
    
    @staticmethod
    def substring_match(prediction: str, ground_truth: str) -> bool:
        """Check if ground truth is in prediction"""
        return ground_truth.lower() in prediction.lower()
    
    def evaluate(
        self,
        prediction: str,
        ground_truth: str,
        use_substring: bool = True
    ) -> Tuple[bool, bool]:
        """
        Evaluate prediction against ground truth.
        
        Returns:
            (is_correct_substring, is_correct_exact)
        """
        # Substring match
        is_correct_substring = self.substring_match(prediction, ground_truth)
        
        # Normalized exact match
        norm_pred = self.normalize_answer(prediction)
        norm_gt = self.normalize_answer(ground_truth)
        is_correct_exact = norm_pred == norm_gt
        
        return is_correct_substring, is_correct_exact


class BasicRAGGenerator:
    """Handles LLM loading and generation"""
    
    def __init__(self, config: BasicRAGConfig):
        self.config = config
        
        print(f"Loading model: {config.model_path}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        
        # Initialize vLLM
        self.model = LLM(
            model=config.model_path,
            dtype=config.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager,
            disable_log_stats=True
        )
        
        # Sampling parameters
        self.sampling_params = SamplingParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            top_p=config.top_p
        )
        
        print("Model loaded successfully!")
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate responses for a batch of prompts"""
        # Disable vLLM's internal progress bar
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=False)
        predictions = [output.outputs[0].text.strip() for output in outputs]
        return predictions


class BasicRAGChain(Chain):
    """
    Basic RAG Chain - Simple Retrieval + Generation
    
    Flow:
    1. For each question:
       - Retrieve top-k relevant documents
       - Format prompt with question + documents
       - Generate answer with LLM
       - Evaluate result
    2. Return predictions and metrics
    """
    
    def __init__(self, config: BasicRAGConfig, retriever: FaissJsonRetriever):
        self.config = config
        
        # Initialize components
        self.formatter = RAGPromptFormatter(self.config)
        self.retriever = retriever
        if not self.retriever:
            raise ValueError("BasicRAGChain requires a FaissJsonRetriever instance")
        self.generator = BasicRAGGenerator(self.config)
        self.evaluator = Evaluator()
    
    @property
    def input_keys(self) -> List[str]:
        return ["instructions", "questions", "ground_truths"]
    
    @property
    def output_keys(self) -> List[str]:
        return ["results", "metrics"]
    
    def _call(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run Basic RAG evaluation"""
        instructions = inputs["instructions"]
        questions = inputs["questions"]
        ground_truths = inputs["ground_truths"]
        metadata_list = inputs.get("metadata", [{}] * len(questions))
        
        # Process in batches
        all_results = []
        batch_size = self.config.batch_size
        total_batches = (len(questions) + batch_size - 1) // batch_size
        
        for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches", total=total_batches):
            batch_instructions = instructions[i:i + batch_size]
            batch_questions = questions[i:i + batch_size]
            batch_ground_truths = ground_truths[i:i + batch_size]
            batch_metadata = metadata_list[i:i + batch_size]
            
            # Retrieve documents for each question
            batch_docs = []
            for j, question in enumerate(batch_questions):
                # Encode query to embedding (FaissJsonRetriever expects embeddings)
                query_embedding = query_encode([question])
                # Retrieve using FaissJsonRetriever
                retrieval_result = self.retriever.retrieve(query_embedding, k=self.config.top_k)
                # Extract document texts
                docs = [doc.page_content for doc in retrieval_result.documents]
                batch_docs.append(docs)
            

            # Format prompts with retrieved documents
            batch_prompts = [
                self.formatter.format_prompt_with_context(
                    instruction,
                    question,
                    docs
                )
                for instruction, question, docs in zip(
                    batch_instructions,
                    batch_questions,
                    batch_docs
                )
            ]

            # Generate predictions
            batch_predictions = self.generator.generate(batch_prompts)
            # Evaluate each prediction
            for j, (prediction, ground_truth, docs, meta) in enumerate(zip(
                batch_predictions,
                batch_ground_truths,
                batch_docs,
                batch_metadata
            )):
                is_correct_substring, is_correct_exact = self.evaluator.evaluate(
                    prediction,
                    ground_truth,
                    use_substring=self.config.use_substring_match
                )
                
                result = RAGEvaluationResult(
                    prediction=prediction,
                    ground_truth=ground_truth,
                    is_correct_substring=is_correct_substring,
                    is_correct_exact=is_correct_exact,
                    retrieved_docs=docs,
                    num_docs_used=len(docs),
                    metadata={
                        **meta,
                        "index": i + j,
                        "prompt_length": len(batch_prompts[j]),
                        "num_retrieved_docs": len(docs)
                    }
                )
                all_results.append(result)
        
        # Calculate metrics
        metrics = self.calculate_accuracy(all_results)
        
        return {
            "results": all_results,
            "metrics": metrics
        }
    
    def calculate_accuracy(self, results: List[RAGEvaluationResult]) -> Dict[str, Any]:
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
                "questions_with_retrieval": sum(1 for r in results if r.num_docs_used > 0)
            },
            "primary_method": "substring_match" if self.config.use_substring_match else "exact_match"
        }

