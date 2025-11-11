"""Functional database query builders for RAG retrieval."""
from typing import List, Dict, Any, Callable
from sqlalchemy import select, func, and_, or_
from sqlalchemy.orm import Session
from .schema import ContractCorrelation


# Pure function: Build similarity filter based on contract text
def build_text_similarity_filter(contract_text: str) -> Callable:
    """Create a filter function for text similarity (case-insensitive)."""
    keywords = set(contract_text.lower().split())

    def filter_fn(row: ContractCorrelation) -> bool:
        title_a = (row.contract_a_title or "").lower()
        title_b = (row.contract_b_title or "").lower()
        reasoning = (row.correlation_reasoning or "").lower()

        combined = f"{title_a} {title_b} {reasoning}"
        return any(keyword in combined for keyword in keywords if len(keyword) > 3)

    return filter_fn


# Pure function: Extract keywords from contract text
def extract_keywords(text: str, min_length: int = 4) -> List[str]:
    """Extract meaningful keywords from text."""
    stopwords = {'this', 'that', 'with', 'from', 'have', 'will', 'been', 'their'}
    words = text.lower().split()
    return [w for w in words if len(w) >= min_length and w not in stopwords]


# Query function: Retrieve similar correlations using keyword matching
def get_similar_correlations(
    session: Session,
    contract_a_text: str,
    contract_b_text: str,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve top-k similar contract correlations from database.

    Uses keyword-based similarity search on titles and reasoning.
    """
    keywords_a = extract_keywords(contract_a_text)
    keywords_b = extract_keywords(contract_b_text)

    # Build query with keyword matching
    query = select(ContractCorrelation).where(
        and_(
            ContractCorrelation.is_active == True,
            or_(
                *[ContractCorrelation.contract_a_title.ilike(f"%{kw}%") for kw in keywords_a],
                *[ContractCorrelation.contract_b_title.ilike(f"%{kw}%") for kw in keywords_b],
                *[ContractCorrelation.correlation_reasoning.ilike(f"%{kw}%")
                  for kw in keywords_a + keywords_b]
            )
        )
    ).limit(top_k * 2)  # Get more, then filter

    results = session.execute(query).scalars().all()

    # Convert to dicts and score
    scored_results = [
        {
            "contract_a_title": r.contract_a_title,
            "contract_b_title": r.contract_b_title,
            "underlying_correlation": r.underlying_event_correlation,
            "correlation_type": r.correlation_type,
            "correlation_strength": r.correlation_strength,
            "reasoning": r.correlation_reasoning,
            "common_factors": r.common_factors,
            "causal_relationship": r.causal_relationship,
            "confidence": r.analysis_confidence,
        }
        for r in results[:top_k]
    ]

    return scored_results


# Query function: Get correlation statistics
def get_correlation_stats(session: Session) -> Dict[str, Any]:
    """Get aggregate statistics from correlations table."""
    stats = session.execute(
        select(
            func.count(ContractCorrelation.id).label("total"),
            func.avg(ContractCorrelation.underlying_event_correlation).label("avg_correlation"),
            func.avg(ContractCorrelation.analysis_confidence).label("avg_confidence"),
        )
    ).first()

    return {
        "total_correlations": stats.total,
        "avg_underlying_correlation": float(stats.avg_correlation or 0),
        "avg_confidence": float(stats.avg_confidence or 0),
    }


# Pure function: Format RAG context from results
def format_rag_context(similar_correlations: List[Dict[str, Any]]) -> str:
    """Format retrieved correlations into context string for LLM prompt."""
    if not similar_correlations:
        return "No similar historical correlations found."

    context_parts = ["Here are similar contract correlation analyses:"]

    for i, corr in enumerate(similar_correlations, 1):
        context_parts.append(
            f"\n{i}. {corr['contract_a_title']} vs {corr['contract_b_title']}\n"
            f"   Correlation: {corr['underlying_correlation']:.2f} ({corr['correlation_type']})\n"
            f"   Strength: {corr['correlation_strength']}\n"
            f"   Reasoning: {corr['reasoning']}\n"
            f"   Common factors: {corr['common_factors']}"
        )

    return "\n".join(context_parts)
