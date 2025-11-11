"""Example script to test database connection and queries."""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.database import get_db_session
from src.db.queries import get_correlation_stats, get_similar_correlations


def test_database_connection():
    """Test basic database connection."""
    print("Testing database connection...")

    try:
        with get_db_session() as session:
            stats = get_correlation_stats(session)

            print("\nDatabase Connection: SUCCESS")
            print(f"\nCorrelation Statistics:")
            print(f"  Total correlations: {stats['total_correlations']}")
            print(f"  Average correlation: {stats['avg_underlying_correlation']:.3f}")
            print(f"  Average confidence: {stats['avg_confidence']:.3f}")

            return True
    except Exception as e:
        print(f"\nDatabase Connection: FAILED")
        print(f"Error: {e}")
        return False


def test_rag_retrieval():
    """Test RAG retrieval functionality."""
    print("\n" + "=" * 80)
    print("Testing RAG Retrieval")
    print("=" * 80)

    contract_a = "Bitcoin price prediction"
    contract_b = "Ethereum price prediction"

    try:
        with get_db_session() as session:
            results = get_similar_correlations(
                session=session,
                contract_a_text=contract_a,
                contract_b_text=contract_b,
                top_k=3
            )

            print(f"\nQuery:")
            print(f"  Contract A: {contract_a}")
            print(f"  Contract B: {contract_b}")
            print(f"\nRetrieved {len(results)} similar correlations:\n")

            for i, result in enumerate(results, 1):
                print(f"{i}. {result['contract_a_title']} vs {result['contract_b_title']}")
                print(f"   Correlation: {result['underlying_correlation']:.2f} ({result['correlation_type']})")
                print(f"   Reasoning: {result['reasoning'][:100]}...")
                print()

            return True
    except Exception as e:
        print(f"\nRAG Retrieval: FAILED")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    print("Contract Correlation Database Tests")
    print("=" * 80)

    success = test_database_connection()

    if success:
        test_rag_retrieval()
    else:
        print("\nSkipping RAG test due to connection failure.")
