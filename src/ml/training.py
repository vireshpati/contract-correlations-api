"""QLoRA training with hybrid loss and RAG."""
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import pandas as pd
from sqlalchemy import text
import json
import re

from src.config import get_settings
from src.db.database import get_engine
from src.ml.rag import SemanticRAG, embed_contracts


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./checkpoints"
    num_train_epochs: int = 5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    max_length: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    correlation_loss_weight: float = 0.3  # Weight for MSE on correlation value


def load_training_data(train_split: float = 0.7, val_split: float = 0.15) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load and split contract correlation data from database.

    Args:
        train_split: Proportion for training (default 0.7)
        val_split: Proportion for validation (default 0.15)
        test_split will be 1 - train_split - val_split (default 0.15)

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    settings = get_settings()
    engine = get_engine()

    query = """
    SELECT
        contract_a_title,
        contract_b_title,
        underlying_event_correlation,
        correlation_type,
        correlation_reasoning,
        analysis_confidence
    FROM contract_correlations
    WHERE is_active = true
    ORDER BY created_at
    """

    with engine.connect() as conn:
        df = pd.read_sql(text(query), conn)

    print(f"Loaded {len(df)} contract pairs from database")

    # Create train/val/test splits
    n = len(df)
    train_end = int(n * train_split)
    val_end = int(n * (train_split + val_split))

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    print(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Convert to message format
    def row_to_example(row) -> Dict[str, Any]:
        system_msg = """You are an expert at analyzing prediction market contracts and determining how their outcomes correlate.
Analyze the two contracts and predict their correlation strength, type, and provide reasoning."""

        user_msg = f"""Analyze these two prediction market contracts:

Contract A: {row['contract_a_title']}
Contract B: {row['contract_b_title']}

Based on the contracts, provide a JSON response with:
- underlying_correlation: float between -1.0 (perfect negative) and 1.0 (perfect positive)
- correlation_type: "positive", "negative", or "neutral"
- confidence: float between 0.0 and 1.0
- reasoning: explanation of the correlation

Respond with only the JSON object, no additional text."""

        assistant_msg = json.dumps({
            "underlying_correlation": float(row['underlying_event_correlation']),
            "correlation_type": row['correlation_type'],
            "confidence": float(row['analysis_confidence']),
            "reasoning": row['correlation_reasoning']
        })

        return {
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg}
            ],
            "contract_a": row['contract_a_title'],
            "contract_b": row['contract_b_title'],
            "true_correlation": float(row['underlying_event_correlation'])
        }

    train_data = [row_to_example(row) for _, row in train_df.iterrows()]
    val_data = [row_to_example(row) for _, row in val_df.iterrows()]
    test_data = [row_to_example(row) for _, row in test_df.iterrows()]

    return train_data, val_data, test_data


class CorrelationDataset(Dataset):
    """Dataset for contract correlation with RAG context."""

    def __init__(self, data: List[Dict], tokenizer, rag: Optional[SemanticRAG], model, max_length: int = 1024):
        """
        Initialize dataset.

        Args:
            data: List of examples
            tokenizer: Tokenizer
            rag: RAG system for retrieving similar examples
            model: Model for computing embeddings (needed for RAG)
            max_length: Maximum sequence length
        """
        self.data = data
        self.tokenizer = tokenizer
        self.rag = rag
        self.model = model
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        messages = example["messages"].copy()

        # Add RAG context to user message if available
        if self.rag is not None:
            # Compute embedding for this pair
            query_embedding = embed_contracts(
                self.model,
                self.tokenizer,
                example["contract_a"],
                example["contract_b"],
                device="cuda" if torch.cuda.is_available() else "cpu"
            )

            # Get similar examples (k=3)
            similar = self.rag.search(query_embedding, k=3)
            rag_context = self.rag.format_rag_context(similar)

            # Add RAG context to user message
            messages[1]["content"] += f"\n\n{rag_context}\n"

        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )

        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze(),
            "true_correlation": torch.tensor(example["true_correlation"], dtype=torch.float32)
        }


class HybridLossTrainer(Trainer):
    """Custom trainer with hybrid loss: language modeling + correlation MSE."""

    def __init__(self, correlation_loss_weight: float = 0.3, *args, **kwargs):
        """
        Initialize trainer with hybrid loss.

        Args:
            correlation_loss_weight: Weight for correlation MSE loss (0-1)
        """
        super().__init__(*args, **kwargs)
        self.correlation_loss_weight = correlation_loss_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Compute hybrid loss: cross-entropy for generation + MSE for correlation value.

        Args:
            model: The model
            inputs: Input batch
            return_outputs: Whether to return model outputs

        Returns:
            Loss value (and optionally outputs)
        """
        true_correlation = inputs.pop("true_correlation")

        # Standard language modeling loss
        outputs = model(**inputs)
        lm_loss = outputs.loss

        # Extract predicted correlation from generated text
        # Generate a short sequence to get the correlation prediction
        with torch.no_grad():
            generated = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id
            )

            # Decode and extract correlation value
            pred_correlations = []
            for gen_seq in generated:
                text = self.tokenizer.decode(gen_seq, skip_special_tokens=True)
                # Try to extract correlation value from JSON
                try:
                    json_match = re.search(r'\{[^{}]*\}', text, re.DOTALL)
                    if json_match:
                        pred_data = json.loads(json_match.group())
                        pred_corr = pred_data.get("underlying_correlation", 0.0)
                    else:
                        pred_corr = 0.0
                except:
                    pred_corr = 0.0
                pred_correlations.append(pred_corr)

            pred_correlation_tensor = torch.tensor(
                pred_correlations,
                dtype=torch.float32,
                device=true_correlation.device
            )

        # MSE loss on correlation values
        mse_loss = F.mse_loss(pred_correlation_tensor, true_correlation)

        # Combined loss
        total_loss = (1 - self.correlation_loss_weight) * lm_loss + self.correlation_loss_weight * mse_loss

        return (total_loss, outputs) if return_outputs else total_loss


def train_model(config: Optional[TrainingConfig] = None, data_limit: Optional[int] = None):
    """
    Train QLoRA model with hybrid loss and RAG.

    Args:
        config: Training configuration
        data_limit: Optional limit on training data size
    """
    if config is None:
        config = TrainingConfig()

    settings = get_settings()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("Training Contract Correlation Model with QLoRA + RAG + Hybrid Loss")
    print("=" * 80)

    # Load data with proper splits
    print("\n1. Loading and splitting data...")
    train_data, val_data, test_data = load_training_data()

    if data_limit:
        train_data = train_data[:data_limit]
        val_data = val_data[:min(data_limit // 5, len(val_data))]

    print(f"Final sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    # Load model and tokenizer
    print("\n2. Loading base model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name,
        token=settings.hf_token,
    )
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=settings.hf_token,
        torch_dtype=torch.float16,
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    print("\n3. Configuring LoRA...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load RAG index
    print("\n4. Loading RAG index...")
    rag = None
    try:
        rag = SemanticRAG(index_path="./faiss_index")
        rag.load()
        print("RAG index loaded successfully - will use during training")
    except Exception as e:
        print(f"Warning: Could not load RAG index: {e}")
        print("Training will proceed without RAG context")

    # Create datasets
    print("\n5. Creating datasets with RAG context...")
    train_dataset = CorrelationDataset(train_data, tokenizer, rag, model, config.max_length)
    val_dataset = CorrelationDataset(val_data, tokenizer, rag, model, config.max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="none"
    )

    # Create trainer with hybrid loss
    print("\n6. Initializing trainer with hybrid loss...")
    trainer = HybridLossTrainer(
        correlation_loss_weight=config.correlation_loss_weight,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer
    )

    # Train
    print("\n7. Starting training...")
    trainer.train()

    # Save final model
    print("\n8. Saving final model...")
    trainer.save_model(f"{config.output_dir}/final_model")

    print("\n" + "=" * 80)
    print("Training complete!")
    print(f"Model saved to: {config.output_dir}/final_model")
    print("=" * 80)
