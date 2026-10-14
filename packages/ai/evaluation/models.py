from pydantic import BaseModel, Field

class RAGEvaluationCase(BaseModel):
    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    relevant_sources: list[str] = Field(
        default_factory=list
    )
    
    expected_answer_terms: list[str] = Field(
        default_factory=list
    )

    should_abstain: bool = False

class RetrievalMetrics(BaseModel):
    precision_at_k: float = Field(ge=0, le=1)
    recall_at_k: float = Field(ge=0, le=1)
    
class RAGEvaluationResult(BaseModel):
    case_id: str

    retrieval: RetrievalMetrics
    
    status_correct: bool
    answer_terms_present: bool

    passed: bool
    