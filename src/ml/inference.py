"""Inference logic for contract correlation prediction."""
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
import re
from typing import Dict, Any, Optional
from pathlib import Path

from src.ml.rag import SemanticRAG, embed_contracts
from src.config import get_settings


class CorrelationPredictor:
    """Predict correlations between prediction market contracts."""

    def __init__(self, model_path: Optional[str] = None, index_path: str = "./faiss_index"):
        """
        Initialize predictor with QLoRA fine-tuned model and RAG.

        Args:
            model_path: Path to fine-tuned model checkpoint (if available)
            index_path: Path to FAISS index for RAG
        """
        self.settings = get_settings()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load base model with quantization
        print("Loading base Llama 3.1 8B model...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_name,
            token=self.settings.hf_token,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            token=self.settings.hf_token,
            torch_dtype=torch.float16,
        )

        # Load fine-tuned adapter if available
        if model_path and Path(model_path).exists():
            print(f"Loading fine-tuned adapter from {model_path}...")
            self.model = PeftModel.from_pretrained(self.model, model_path)

        self.model.eval()

        # Load RAG system
        self.rag = None
        if Path(index_path).exists():
            print("Loading RAG index...")
            self.rag = SemanticRAG(index_path=index_path)
            self.rag.load()
            print("RAG index loaded successfully")

    def build_prompt(self, contract_a: str, contract_b: str, rag_context: Optional[str] = None) -> str:
        """
        Build prompt for correlation prediction.

        Args:
            contract_a: First contract description
            contract_b: Second contract description
            rag_context: Optional RAG context with similar examples

        Returns:
            Formatted prompt string
        """
        system_prompt = """You are an expert at analyzing prediction market contracts and determining how their outcomes correlate.
Analyze the two contracts and predict their correlation strength, type, and provide reasoning."""

        user_prompt = f"""Analyze these two prediction market contracts:

Contract A: {contract_a}
Contract B: {contract_b}
"""

        if rag_context:
            user_prompt += f"\n{rag_context}\n"

        user_prompt += """
Based on the contracts, provide a JSON response with:
- underlying_correlation: float between -1.0 (perfect negative) and 1.0 (perfect positive)
- correlation_type: "positive", "negative", or "neutral"
- confidence: float between 0.0 and 1.0
- reasoning: explanation of the correlation

Respond with only the JSON object, no additional text."""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

    def predict(self, contract_a: str, contract_b: str, use_rag: bool = True) -> Dict[str, Any]:
        """
        Predict correlation between two contracts.

        Args:
            contract_a: First contract description
            contract_b: Second contract description
            use_rag: Whether to use RAG for similar examples

        Returns:
            Dictionary with prediction results
        """
        rag_context = None
        rag_examples_count = None

        # Get RAG context if enabled and available
        if use_rag and self.rag is not None:
            query_embedding = embed_contracts(
                self.model,
                self.tokenizer,
                contract_a,
                contract_b,
                device=self.device
            )
            similar_examples = self.rag.search(query_embedding, k=3)
            rag_context = self.rag.format_rag_context(similar_examples)
            rag_examples_count = len(similar_examples)

        # Build prompt
        prompt = self.build_prompt(contract_a, contract_b, rag_context)

        # Generate prediction
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract JSON from response
        result = self._parse_response(response)

        # Add RAG metadata
        if rag_examples_count is not None:
            result["rag_context_used"] = rag_examples_count

        return result

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """
        Parse model response and extract JSON.

        Args:
            response: Raw model response

        Returns:
            Parsed prediction dictionary
        """
        # Try to find JSON in the response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)

        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Fallback: return default response
        return {
            "underlying_correlation": 0.0,
            "correlation_type": "neutral",
            "confidence": 0.5,
            "reasoning": "Unable to parse model response. Response: " + response[:200]
        }
