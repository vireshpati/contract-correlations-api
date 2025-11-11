"""QLoRA fine-tuning script for Llama 3.1 8B on GPU machine."""
import json
from typing import List, Dict, Any
from dataclasses import dataclass
import torch
from datasets import Dataset
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

    # Load and prepare data
    training_data = load_training_data(limit=data_limit)
    dataset = prepare_dataset(training_data, tokenizer)

    # Tokenize dataset
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

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_train_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        fp16=True,
        optim="paged_adamw_8bit",
        save_total_limit=3,
        push_to_hub=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    print(f"Saving model to {config.output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(config.output_dir)

    print("Training complete!")


if __name__ == "__main__":
    # Train with default config
    train_model()
