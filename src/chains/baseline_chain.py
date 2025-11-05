"""Baseline LLM chain"""

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import re
import string

from .base import Chain
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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
class BaselineConfig:
    """Configuration for baseline LLM evaluation"""
    # Model settings
    model_path: str = "dmis-lab/selfbiorag_7b"
    dtype: str = "half"
    gpu_memory_utilization: float = 0.7
    max_model_len: int = 4096  # Increased from 2048 to support few-shot prompts
    enforce_eager: bool = True
    
    # Generation settings
    temperature: float = 0.0
    max_tokens: int = 200
    top_p: float = 1.0
    
    # Prompting strategy
    use_few_shot: bool = True  # Changed to True by default
    dataset_name: str = "med_qa"
    
    # Evaluation settings
    use_substring_match: bool = True  # Original Self-BioRAG method
    batch_size: int = 8


@dataclass
class EvaluationResult:
    """Result from baseline evaluation"""
    prediction: str
    ground_truth: str
    is_correct_substring: bool
    is_correct_exact: bool
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "prediction": self.prediction,
            "ground_truth": self.ground_truth,
            "is_correct_substring": self.is_correct_substring,
            "is_correct_exact": self.is_correct_exact,
            "metadata": self.metadata
        }


class PromptFormatter:
    """Handles different prompt formatting strategies"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
    
    def format_prompt(self, instruction: str, question: str) -> str:
        """
        Create prompt using different strategies.
        
        Strategies:
        1. Zero-shot: Just instruction + question
        2. Few-shot: Examples + question
        """
        if self.config.use_few_shot and self.config.dataset_name in FEW_SHOT_EXAMPLES:
            # Few-shot mode: prepend examples
            return f"### Instruction:\n{instruction}\n\n" + f"### Examples:\n{FEW_SHOT_EXAMPLES[self.config.dataset_name]}\n\n" + f"### Question (Always answer with both option alphabets and the named answer):\n{question}\n\n### Response:\n"
        else:
            # Zero-shot mode
            return f"### Instruction:\n{instruction}\n\n### Question (Always answer with both option alphabets and the named answer):\n{question}\n\n### Response:\n"


class Evaluator:
    """Handles different evaluation metrics"""
    
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
    

    # TODO: Revise it to better matching method.
    @staticmethod
    def substring_match(prediction: str, ground_truth: str) -> bool:
        """
        Checks if ground truth is contained in prediction.
        """
        return ground_truth.lower() in prediction.lower()
    
    def evaluate(self, prediction: str, ground_truth: str, use_substring: bool = True) -> Tuple[bool, Dict]:
        """
        Evaluate prediction against ground truth.
        
        Returns:
            (is_correct, metadata)
        """
        # Substring match (original method)
        is_correct_substring = self.substring_match(prediction, ground_truth)
        
        # Normalized exact match (strict)
        norm_pred = self.normalize_answer(prediction)
        norm_gt = self.normalize_answer(ground_truth)
        is_correct_exact = norm_pred == norm_gt
        
        # Use configured method for primary correctness
        is_correct = is_correct_substring if use_substring else is_correct_exact
        
        return is_correct, {
            "substring_match": is_correct_substring,
            "exact_match": is_correct_exact,
            "normalized_prediction": norm_pred,
            "normalized_ground_truth": norm_gt
        }


class BaselineGenerator:
    """Handles LLM loading and generation"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        print(f"Loading model: {config.model_path}")
        print(f"Few-shot mode: {'Enabled (4 examples)' if config.use_few_shot else 'Disabled (zero-shot)'}")
        
        self.model = LLM(
            model=config.model_path,
            dtype=config.dtype,
            tensor_parallel_size=1,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_model_len=config.max_model_len,
            enforce_eager=config.enforce_eager
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_path)
        print("Model loaded successfully!")
    
    def generate(self, prompts: List[str]) -> List[str]:
        """Generate answers for a batch of prompts"""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            max_tokens=self.config.max_tokens
        )
        
        outputs = self.model.generate(prompts, sampling_params)
        return [output.outputs[0].text.strip() for output in outputs]


class BaselineChain(Chain):
    """
    Baseline LLM evaluation chain.
    
    Pipeline:
    1. Format prompts (zero-shot or few-shot)
    2. Generate answers in batches
    3. Evaluate predictions against ground truth
    4. Aggregate metrics
    
    LangChain Interface:
    - Inherits from langchain_core.runnables.Chain
    - Implements input_keys, output_keys, and _call
    - Compatible with LangChain callbacks and tracing
    """
    
    config: BaselineConfig
    formatter: PromptFormatter
    generator: BaselineGenerator
    evaluator: Evaluator
    
    def __init__(self, config: Optional[BaselineConfig] = None):
        super().__init__()
        self.config = config or BaselineConfig()
        self.formatter = PromptFormatter(self.config)
        self.generator = BaselineGenerator(self.config)
        self.evaluator = Evaluator()
    
    @property
    def input_keys(self) -> List[str]:
        """Required input keys for LangChain Chain interface"""
        return ["instructions", "questions", "ground_truths"]
    
    @property
    def output_keys(self) -> List[str]:
        """Required output keys for LangChain Chain interface"""
        return ["results", "metrics"]
    
    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run evaluation on a batch of questions.
        
        Args:
            inputs: Dictionary with required keys:
                - instructions: List[str] - Task instructions for each question
                - questions: List[str] - Questions with options
                - ground_truths: List[str] - Ground truth answers
                - metadata: Optional[List[Dict]] - Optional metadata
            run_manager: Optional callback manager for LangChain
            
        Returns:
            Dictionary with:
                - results: List[EvaluationResult] - Evaluation results
                - metrics: Dict - Aggregate metrics
        """
        instructions = inputs["instructions"]
        questions = inputs["questions"]
        ground_truths = inputs["ground_truths"]
        metadata = inputs.get("metadata")
        
        # Phase 1: Format prompts
        prompts = [
            self.formatter.format_prompt(inst, q)
            for inst, q in zip(instructions, questions)
        ]
        
        # Callback: on_chain_start
        if run_manager:
            run_manager.on_text("Formatting prompts...\n", verbose=self.verbose)
        
        # Phase 2: Generate answers
        if run_manager:
            run_manager.on_text("Generating predictions...\n", verbose=self.verbose)
        
        predictions = self.generator.generate(prompts)
        
        # Phase 3: Evaluate
        if run_manager:
            run_manager.on_text("Evaluating predictions...\n", verbose=self.verbose)
        
        results = []
        for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            is_correct, eval_metadata = self.evaluator.evaluate(
                pred, gt, use_substring=self.config.use_substring_match
            )
            
            result_metadata = {
                "index": i,
                "prompt_length": len(prompts[i]),
                **eval_metadata
            }
            
            if metadata and i < len(metadata):
                result_metadata.update(metadata[i])
            
            results.append(EvaluationResult(
                prediction=pred,
                ground_truth=gt,
                is_correct_substring=eval_metadata["substring_match"],
                is_correct_exact=eval_metadata["exact_match"],
                metadata=result_metadata
            ))
        
        # Calculate metrics
        metrics = self.calculate_accuracy(results)
        
        return {
            "results": results,
            "metrics": metrics
        }
    
    def evaluate_single(
        self,
        instruction: str,
        question: str,
        ground_truth: str
    ) -> EvaluationResult:
        """Evaluate a single question (convenience method)"""
        result = self.invoke({
            "instructions": [instruction],
            "questions": [question],
            "ground_truths": [ground_truth]
        })
        return result["results"][0]
    
    def calculate_accuracy(self, results: List[EvaluationResult]) -> Dict[str, Any]:
        """Calculate aggregate accuracy metrics"""
        if not results:
            return {"accuracy": 0.0, "correct": 0, "total": 0}
        
        # Substring match accuracy (original method)
        correct_substring = sum(1 for r in results if r.is_correct_substring)
        
        # Exact match accuracy (strict)
        correct_exact = sum(1 for r in results if r.is_correct_exact)
        
        total = len(results)
        
        return {
            "substring_match": {
                "accuracy": 100 * correct_substring / total,
                "correct": correct_substring,
                "total": total
            },
            "exact_match": {
                "accuracy": 100 * correct_exact / total,
                "correct": correct_exact,
                "total": total
            },
            "primary_method": "substring_match" if self.config.use_substring_match else "exact_match"
        }

