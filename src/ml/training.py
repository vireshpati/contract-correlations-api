"""QLoRA fine-tuning script for Llama 3.1 8B on GPU machine with hybrid loss."""
import json
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sqlalchemy import select
from ..database import get_db_session
from ..db.schema import ContractCorrelation
from ..config import get_settings
from .prompt_builder import build_training_prompt
from .training_utils import (
    extract_correlation_from_labels,
    extract_correlation_from_logits,
    compute_regression_loss
)
import numpy as np


@dataclass
class TrainingConfig:
    """Training configuration."""
    output_dir: str = "./models/llama-3.1-8b-correlation-qlora"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    logging_steps: int = 10
    save_steps: int = 100
    max_seq_length: int = 2048
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    regression_loss_weight: float = 10.0  # Weight for MSE loss
    regression_loss_type: str = "mse"  # "mse", "mae", or "huber"


class RegressionAugmentedTrainer(Trainer):
    """
    Custom Trainer that adds regression loss on correlation predictions.

    Combines standard generation loss with MSE loss on parsed correlation values.
    """

    def __init__(self, *args, regression_loss_weight=10.0, regression_loss_type="mse", **kwargs):
        super().__init__(*args, **kwargs)
        self.regression_loss_weight = regression_loss_weight
        self.regression_loss_type = regression_loss_type

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Compute hybrid loss: generation loss + weighted regression loss.

        Args:
            model: The model being trained
            inputs: Dict with input_ids, attention_mask, labels
            return_outputs: Whether to return model outputs
            num_items_in_batch: Number of items in batch (ignored, for API compatibility)

        Returns:
            Loss tensor (and optionally outputs)
        """
        # Standard forward pass for generation loss
        outputs = model(**inputs)
        generation_loss = outputs.loss

        # Extract correlation values from labels (ground truth)
        labels = inputs.get("labels")
        if labels is None:
            # No regression loss if no labels
            return (generation_loss, outputs) if return_outputs else generation_loss

        try:
            true_correlations = extract_correlation_from_labels(labels, self.tokenizer)

            # Extract predicted correlations from logits
            logits = outputs.logits
            attention_mask = inputs.get("attention_mask")
            pred_correlations = extract_correlation_from_logits(
                logits, self.tokenizer, attention_mask
            )

            # Compute regression loss
            regression_loss = compute_regression_loss(
                pred_correlations,
                true_correlations,
                loss_type=self.regression_loss_type
            )

            # Combined loss
            total_loss = generation_loss + self.regression_loss_weight * regression_loss

            # Log losses
            if self.state.global_step % self.args.logging_steps == 0:
                self.log({
                    "generation_loss": generation_loss.item(),
                    "regression_loss": regression_loss.item(),
                    "total_loss": total_loss.item()
                })

        except Exception as e:
            # If parsing fails, fall back to generation loss only
            print(f"Warning: Regression loss computation failed: {e}")
            total_loss = generation_loss

        return (total_loss, outputs) if return_outputs else total_loss


def load_training_data(limit: int = None) -> List[Dict[str, Any]]:
    """
    Load training data from contract_correlations table.

    Args:
        limit: Optional limit on number of rows to load

    Returns:
        List of training examples
    """
    print("Loading training data from database...")

    with get_db_session() as session:
        query = select(ContractCorrelation).where(
            ContractCorrelation.is_active == True
        )
        if limit:
            query = query.limit(limit)

        results = session.execute(query).scalars().all()

        training_data = []
        for row in results:
            example = build_training_prompt(
                contract_a_title=row.contract_a_title,
                contract_b_title=row.contract_b_title,
                target_correlation=row.underlying_event_correlation,
                target_type=row.correlation_type,
                target_reasoning=row.correlation_reasoning or "No reasoning provided"
            )
            training_data.append(example)

    print(f"Loaded {len(training_data)} training examples")
    return training_data


def split_data(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train/val/test sets.

    Args:
        data: Full dataset
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1"

    np.random.seed(seed)
    indices = np.random.permutation(len(data))

    train_end = int(len(data) * train_ratio)
    val_end = train_end + int(len(data) * val_ratio)

    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]

    train_data = [data[i] for i in train_indices]
    val_data = [data[i] for i in val_indices]
    test_data = [data[i] for i in test_indices]

    print(f"Split: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    return train_data, val_data, test_data


def prepare_dataset(training_data: List[Dict[str, Any]], tokenizer) -> Dataset:
    """
    Prepare dataset for training.

    Args:
        training_data: List of message dictionaries
        tokenizer: HuggingFace tokenizer

    Returns:
        Formatted dataset
    """
    def format_example(example):
        """Format example for training."""
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return {"text": text}

    formatted_data = [format_example(ex) for ex in training_data]
    return Dataset.from_list(formatted_data)


def train_model(config: TrainingConfig = None, data_limit: int = None):
    """
    Train Llama 3.1 8B with QLoRA on contract correlations.

    Args:
        config: Training configuration
        data_limit: Optional limit on training data size
    """
    if config is None:
        config = TrainingConfig()

    settings = get_settings()

    print("Setting up QLoRA training...")

    # Configure 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # Load model
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        settings.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        token=settings.hf_token,
        torch_dtype=torch.float16,
        cache_dir=settings.model_cache_dir
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Get PEFT model
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        settings.model_name,
        token=settings.hf_token,
        cache_dir=settings.model_cache_dir
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load and split data
    all_data = load_training_data(limit=data_limit)
    train_data, val_data, test_data = split_data(all_data)

    # Prepare datasets
    train_dataset = prepare_dataset(train_data, tokenizer)
    val_dataset = prepare_dataset(val_data, tokenizer)
    test_dataset = prepare_dataset(test_data, tokenizer)

    # Tokenize datasets
    def tokenize_function(examples):
        """Tokenize examples."""
        result = tokenizer(
            examples["text"],
            truncation=True,
            max_length=config.max_seq_length,
            padding="max_length"
        )
        result["labels"] = result["input_ids"].copy()
        return result

    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=train_dataset.column_names
    )
    tokenized_val = val_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=val_dataset.column_names
    )
    tokenized_test = test_dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=test_dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_strategy="steps",  # Changed from evaluation_strategy
        eval_steps=config.save_steps,
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=True,
        optim="paged_adamw_8bit",
        save_total_limit=3,
        push_to_hub=False
    )

    # Create trainer with regression loss
    trainer = RegressionAugmentedTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        regression_loss_weight=config.regression_loss_weight,
        regression_loss_type=config.regression_loss_type
    )

    # Train
    print("Starting training...")
    train_result = trainer.train()

    # Evaluate on validation set
    print("\nEvaluating on validation set...")
    val_metrics = trainer.evaluate(eval_dataset=tokenized_val)
    print(f"Validation metrics: {val_metrics}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized_test)
    print(f"Test metrics: {test_metrics}")

    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    # Save metrics
    import json
    metrics_file = f"{config.output_dir}/metrics.json"
    with open(metrics_file, 'w') as f:
        json.dump({
            "train": train_result.metrics,
            "validation": val_metrics,
            "test": test_metrics
        }, f, indent=2)
    print(f"Metrics saved to {metrics_file}")

    print("\n" + "="*80)
    print("Training complete!")
    print(f"Final test loss: {test_metrics.get('eval_loss', 'N/A'):.4f}")
    print("="*80)


if __name__ == "__main__":
    # Train with default config
    train_model()
