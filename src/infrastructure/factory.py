"""Factory for creating Self-BioRAG components"""

from typing import Optional, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from ..generator_models import GeneratorModel
from ..retriever_models import VectorStoreRetriever, PreloadedRetriever
from ..chains import SelfBioRAGChain, SelfBioRAGConfig
from .utils import load_reflection_tokens, load_jsonl


class SelfBioRAGFactory:
    """Factory for easy Self-BioRAG pipeline creation"""
    
    @staticmethod
    def create_chain(
        model_path: str,
        retriever_type: str = "vectorstore",
        vectorstore_path: Optional[str] = None,
        evidence_file: Optional[str] = None,
        embeddings_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        config: Optional[Dict] = None,
        **kwargs
    ) -> SelfBioRAGChain:
        """
        Create Self-BioRAG pipeline from configuration.
        
        Args:
            model_path: Path to Self-BioRAG model (e.g., "dmis-lab/selfbiorag_7b")
            retriever_type: "vectorstore" or "preloaded"
            vectorstore_path: Path to FAISS index (for vectorstore)
            evidence_file: Path to evidence JSONL (for preloaded)
            embeddings_model: Embeddings model name
            config: Chain configuration dict
            **kwargs: Additional model args (dtype, tensor_parallel_size, etc.)
        
        Returns:
            Configured SelfBioRAGChain
        """
        
        # 1. Create generator model
        generator = GeneratorModel(
            model_path=model_path,
            max_tokens=kwargs.get("max_tokens", 200),
            temperature=kwargs.get("temperature", 0.0),
            dtype=kwargs.get("dtype", "half"),
            tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
            download_dir=kwargs.get("download_dir")
        )
        
        # 2. Load reflection tokens
        ret_tokens, rel_tokens, grd_tokens, ut_tokens = load_reflection_tokens(
            generator.tokenizer,
            use_grounding=config.get("use_groundedness", True) if config else True,
            use_utility=config.get("use_utility", True) if config else True
        )
        
        # 3. Create retriever
        if retriever_type == "vectorstore":
            if not vectorstore_path:
                raise ValueError("vectorstore_path required for vectorstore retriever")
            
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
            vectorstore = FAISS.load_local(vectorstore_path, embeddings)
            retriever = VectorStoreRetriever(vectorstore=vectorstore)
        
        elif retriever_type == "preloaded":
            if not evidence_file:
                raise ValueError("evidence_file required for preloaded retriever")
            
            evidence_list = load_jsonl(evidence_file)
            evidence_dict = {i: ev for i, ev in enumerate(evidence_list)}
            retriever = PreloadedRetriever(evidence_dict)
        
        else:
            raise ValueError(f"Unknown retriever_type: {retriever_type}")
        
        # 4. Create chain configuration
        chain_config = SelfBioRAGConfig(
            mode=config.get("mode", "adaptive_retrieval") if config else "adaptive_retrieval",
            top_k=config.get("top_k", 10) if config else 10,
            use_groundedness=config.get("use_groundedness", True) if config else True,
            use_utility=config.get("use_utility", True) if config else True,
            use_seqscore=config.get("use_seqscore", False) if config else False,
            threshold=config.get("threshold", 0.5) if config else 0.5,
            closed_domain=config.get("closed_domain", True) if config else True,
            w_rel=config.get("w_rel", 1.0) if config else 1.0,
            w_sup=config.get("w_sup", 1.0) if config else 1.0,
            w_use=config.get("w_use", 0.5) if config else 0.5
        )
        
        # 5. Create chain
        return SelfBioRAGChain(
            generator=generator,
            retriever=retriever,
            ret_tokens=ret_tokens,
            rel_tokens=rel_tokens,
            grd_tokens=grd_tokens,
            ut_tokens=ut_tokens,
            config=chain_config
        )