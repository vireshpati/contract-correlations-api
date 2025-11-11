"""Evaluate trained model on test set."""
import sys
import os
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.ml.training import load_training_data, split_data
from src.ml.inference import CorrelationPredictor
from src.ml.prompt_builder import build_user_prompt
import numpy as np


def evaluate_predictions(predictions, ground_truth):
    """
    Calculate evaluation metrics.

    Args:
        predictions: List of predicted values
        ground_truth: List of true values

    Returns:
        Dictionary of metrics
    """
    predictions = np.array(predictions)
    ground_truth = np.array(ground_truth)

    # Mean Absolute Error
    mae = np.mean(np.abs(predictions - ground_truth))

    # Mean Squared Error
    mse = np.mean((predictions - ground_truth) ** 2)

    # RMSE
    rmse = np.sqrt(mse)

    # R-squared (correlation coefficient squared)
    correlation = np.corrcoef(predictions, ground_truth)[0, 1]
    r_squared = correlation ** 2

    # Accuracy within threshold
    threshold_01 = np.mean(np.abs(predictions - ground_truth) <= 0.1)
    threshold_02 = np.mean(np.abs(predictions - ground_truth) <= 0.2)
    threshold_03 = np.mean(np.abs(predictions - ground_truth) <= 0.3)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "rmse": float(rmse),
        "r_squared": float(r_squared),
        "correlation": float(correlation),
        "accuracy_0.1": float(threshold_01),
        "accuracy_0.2": float(threshold_02),
        "accuracy_0.3": float(threshold_03),
    }


def run_evaluation(model_path: str = None, num_samples: int = None):
    """
    Run evaluation on test set.

    Args:
        model_path: Path to trained model (None for base model)
        num_samples: Number of test samples to evaluate (None for all)
    """
    print("=" * 80)
    print("Model Evaluation on Test Set")
    print("=" * 80)

    # Load and split data
    print("\nLoading data...")
    all_data = load_training_data()
    _, _, test_data = split_data(all_data)

    if num_samples:
        test_data = test_data[:num_samples]

    print(f"Evaluating on {len(test_data)} test examples")

    # Load model
    print("\nLoading model...")
    if model_path:
        print(f"Using fine-tuned model: {model_path}")
        # Update config to use fine-tuned model
        os.environ['MODEL_NAME'] = model_path
    else:
        print("Using base model")

    predictor = CorrelationPredictor()
    predictor.load_model()

    # Run predictions
    print("\nRunning predictions...")
    predictions = []
    ground_truth = []

    for i, example in enumerate(test_data):
        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(test_data)}")

        messages = example["messages"]
        user_msg = messages[1]["content"]

        # Extract contract titles from user message
        lines = user_msg.split('\n')
        contract_a = lines[1].replace("Contract A: ", "")
        contract_b = lines[2].replace("Contract B: ", "")

        # Get ground truth
        assistant_msg = messages[2]["content"]
        gt_correlation = eval(assistant_msg)["underlying_correlation"]

        # Predict
        try:
            prediction = predictor.predict(contract_a, contract_b, rag_context=None)
            pred_correlation = prediction["underlying_correlation"]

            predictions.append(pred_correlation)
            ground_truth.append(gt_correlation)
        except Exception as e:
            print(f"  Warning: Prediction failed for example {i}: {e}")
            continue

    # Calculate metrics
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    metrics = evaluate_predictions(predictions, ground_truth)

    print(f"\nNumber of predictions: {len(predictions)}")
    print(f"\nRegression Metrics:")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r_squared']:.4f}")
    print(f"  Correlation: {metrics['correlation']:.4f}")

    print(f"\nAccuracy (within threshold):")
    print(f"  ±0.1: {metrics['accuracy_0.1']:.1%}")
    print(f"  ±0.2: {metrics['accuracy_0.2']:.1%}")
    print(f"  ±0.3: {metrics['accuracy_0.3']:.1%}")

    # Save results
    results_file = "evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "model_path": model_path or "base_model",
            "num_samples": len(predictions),
            "metrics": metrics,
            "predictions": predictions[:10],  # Save first 10
            "ground_truth": ground_truth[:10]
        }, f, indent=2)

    print(f"\nResults saved to: {results_file}")
    print("=" * 80)

    return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate model on test set")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to fine-tuned model (default: use base model)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of test samples (default: all)"
    )

    args = parser.parse_args()

    run_evaluation(model_path=args.model, num_samples=args.samples)
