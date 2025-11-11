"""Example script for training the model on GPU machine."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.training import train_model, TrainingConfig


if __name__ == "__main__":
    print("Contract Correlation Model Training")
    print("=" * 80)
    print("\nIMPORTANT: Run this script on a GPU machine!")
    print("This script requires CUDA-capable GPU for efficient training.\n")

    # Custom training config for small test run
    config = TrainingConfig(
        output_dir="./models/llama-3.1-8b-correlation-qlora",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_steps=100,
        logging_steps=10,
        save_steps=100,
        lora_r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )

    print("Training Configuration:")
    print(f"  Output dir: {config.output_dir}")
    print(f"  Epochs: {config.num_train_epochs}")
    print(f"  Batch size: {config.per_device_train_batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  LoRA rank: {config.lora_r}")
    print()

    # Option to run with limited data for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                       help="Limit training data size (for testing)")
    args = parser.parse_args()

    if args.limit:
        print(f"Running with limited dataset: {args.limit} examples\n")

    # Train the model
    train_model(config=config, data_limit=args.limit)

    print("\nTraining complete!")
    print(f"Model saved to: {config.output_dir}")
    print("\nTo use the fine-tuned model:")
    print("1. Copy the model directory to your Mac")
    print("2. Update MODEL_NAME in .env to point to the local path")
