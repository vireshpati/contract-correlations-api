"""RAG (Retrieval-Augmented Generation) system."""
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from ..db.queries import get_similar_correlations, format_rag_context


def retrieve_context(
    session: Session,
    contract_a: str,
    contract_b: str,
    top_k: int = 5
) -> tuple[str, List[Dict[str, Any]]]:
    """
    Retrieve and format RAG context from historical correlations.

    Returns:
        Tuple of (formatted_context_string, list_of_similar_correlations)
    """
    similar_correlations = get_similar_correlations(
        session=session,
        contract_a_text=contract_a,
        contract_b_text=contract_b,
        top_k=top_k
    )

    formatted_context = format_rag_context(similar_correlations)

    return formatted_context, similar_correlations
