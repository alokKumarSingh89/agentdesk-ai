from packages.ai.evaluation.models import (
    RetrievalMetrics,
)

def calculate_retrieval_metrics(
    retrieved_sources: list[str],
    relevant_sources: list[str],
) -> RetrievalMetrics:
    retrieved = set(retrieved_sources)
    relevant = set(relevant_sources)
    
    matches = len(retrieved & relevant)
    
    precision = (
        matches / len(retrieved)
        if retrieved
        else 0.0
    )
    
    recall = (
        matches / len(relevant)
        if relevant
        else 1.0
    )
    
    return RetrievalMetrics(
        precision_at_k=precision,
        recall_at_k=recall,
    )