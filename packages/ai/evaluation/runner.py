from uuid import UUID

from packages.ai.evaluation.metrics import (
    calculate_retrieval_metrics,
)
from packages.ai.evaluation.models import (
    RAGEvaluationCase,
    RAGEvaluationResult,
)
from packages.ai.rag import RAGStatus
from packages.ai.services.rag_service import (
    RAGService,
)

class RAGEvaluationRunner:
    def __init__(
        self,
        rag_service: RAGService,
    ) -> None:
        self.rag_service = rag_service
    
    def evaluate(
        self,
        tenant_id: UUID,
        case: RAGEvaluationCase,
    ) -> RAGEvaluationResult:
        response = self.rag_service.answer(
            tenant_id=tenant_id,
            question=case.question,
        )
        
        retrieved_sources = [
            source.source
            for source in response.sources
        ]
        
        retrieval = calculate_retrieval_metrics(
            retrieved_sources=retrieved_sources,
            relevant_sources=case.relevant_sources,
        )
        
        expected_status = (
            RAGStatus.INSUFFICIENT_CONTEXT
            if case.should_abstain
            else RAGStatus.ANSWERED
        )
        
        status_correct = (
            response.status == expected_status
        )
        
        answer_lower = response.answer.lower()
        
        answer_terms_present = all(
            term.lower() in answer_lower
            for term in case.expected_answer_terms
        )
        
        passed = (
            status_correct
            and answer_terms_present
            and retrieval.recall_at_k == 1.0
            and (
                not case.should_abstain
                or not response.sources
            )
        )
        
        return RAGEvaluationResult(
            case_id=case.id,
            retrieval=retrieval,
            status_correct=status_correct,
            answer_terms_present=answer_terms_present,
            passed=passed,
        )