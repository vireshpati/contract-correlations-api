"""Inference module for Llama 3.1 8B with 4-bit quantization."""
import json
import re
from typing import Optional, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from ..config import get_settings
from .prompt_builder import build_system_prompt, build_user_prompt


class CorrelationPredictor:
    """Llama 3.1 8B predictor with 4-bit quantization for M2 Mac."""

    def __init__(self):
        """Initialize model and tokenizer."""
        self.settings = get_settings()
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load_model(self) -> None:
        """Load model with 4-bit quantization for efficient CPU/MPS inference."""
        if self._loaded:
            return

        print("Loading Llama 3.1 8B model with 4-bit quantization...")

        # Configure 4-bit quantization for efficiency
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        ) if self.settings.use_4bit else None

        # Determine device
        if torch.cuda.is_available():
            device_map = "auto"
        elif torch.backends.mps.is_available():
            device_map = "mps"
        else:
            device_map = "cpu"

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.settings.model_name,
            token=self.settings.hf_token,
            cache_dir=self.settings.model_cache_dir
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.settings.model_name,
            token=self.settings.hf_token,
            quantization_config=quantization_config,
            device_map=device_map,
            cache_dir=self.settings.model_cache_dir,
            torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
            low_cpu_mem_usage=True
        )

        self._loaded = True
        print(f"Model loaded successfully on {device_map}")

    def predict(
        self,
        contract_a: str,
        contract_b: str,
        rag_context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Predict correlation between two contracts.

        Args:
            contract_a: First contract description
            contract_b: Second contract description
            rag_context: Optional RAG context

        Returns:
            Prediction dictionary with correlation, type, confidence, reasoning
        """
        if not self._loaded:
            self.load_model()

        # Build prompts
        system_prompt = build_system_prompt()
        user_prompt = build_user_prompt(contract_a, contract_b, rag_context)

        # Format for Llama 3.1 Instruct
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Tokenize
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                max_new_tokens=self.settings.max_new_tokens,
                temperature=self.settings.temperature,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0][inputs.shape[1]:],
            skip_special_tokens=True
        )

        # Parse JSON response
        return self._parse_response(response)

    def _parse_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured format."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{[^}]+\}', response, re.DOTALL)
            if json_match:
                parsed = json.loads(json_match.group())
                return {
                    "underlying_correlation": float(parsed.get("underlying_correlation", 0.0)),
                    "correlation_type": str(parsed.get("correlation_type", "neutral")),
                    "confidence": float(parsed.get("confidence", 0.5)),
                    "reasoning": str(parsed.get("reasoning", "Unable to determine correlation."))
                }
        except (json.JSONDecodeError, ValueError):
            pass

        # Fallback parsing
        return self._fallback_parse(response)

    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parser when JSON extraction fails."""
        response_lower = response.lower()

        # Determine correlation type
        if "positive" in response_lower:
            corr_type = "positive"
            corr_value = 0.5
        elif "negative" in response_lower:
            corr_type = "negative"
            corr_value = -0.5
        else:
            corr_type = "neutral"
            corr_value = 0.0

        return {
            "underlying_correlation": corr_value,
            "correlation_type": corr_type,
            "confidence": 0.6,
            "reasoning": response[:500]  # First 500 chars
        }

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._loaded


# Global predictor instance
_predictor: Optional[CorrelationPredictor] = None


def get_predictor() -> CorrelationPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CorrelationPredictor()
    return _predictor
