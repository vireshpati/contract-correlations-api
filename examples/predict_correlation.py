"""Example script for predicting contract correlations via API."""
import requests
import json


def predict_correlation(contract_a: str, contract_b: str, use_rag: bool = True):
    """
    Predict correlation between two contracts.

    Args:
        contract_a: First contract description
        contract_b: Second contract description
        use_rag: Whether to use RAG context
    """
    url = "http://localhost:8000/predict-correlation"

    payload = {
        "contract_a": contract_a,
        "contract_b": contract_b,
        "use_rag": use_rag
    }

    print(f"\nPredicting correlation between:")
    print(f"  A: {contract_a}")
    print(f"  B: {contract_b}")
    print(f"  Using RAG: {use_rag}\n")

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        result = response.json()
        print("Prediction:")
        print(json.dumps(result, indent=2))
        return result
    else:
        print(f"Error {response.status_code}: {response.text}")
        return None


if __name__ == "__main__":
    # Example 1: Crypto correlation
    print("=" * 80)
    print("Example 1: Cryptocurrency Correlation")
    print("=" * 80)
    predict_correlation(
        contract_a="Will Bitcoin reach $100,000 by end of 2025?",
        contract_b="Will Ethereum reach $10,000 by end of 2025?",
        use_rag=True
    )

    # Example 2: Economic indicators
    print("\n" + "=" * 80)
    print("Example 2: Economic Indicators")
    print("=" * 80)
    predict_correlation(
        contract_a="Will the Fed cut rates in 2025?",
        contract_b="Will US GDP growth exceed 3% in 2025?",
        use_rag=True
    )

    # Example 3: Unrelated contracts
    print("\n" + "=" * 80)
    print("Example 3: Unrelated Contracts")
    print("=" * 80)
    predict_correlation(
        contract_a="Will it snow in New York on Christmas 2025?",
        contract_b="Will SpaceX launch Starship to Mars in 2025?",
        use_rag=True
    )
