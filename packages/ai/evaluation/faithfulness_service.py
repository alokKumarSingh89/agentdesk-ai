from typing import Protocol

from packages.ai.evaluation.faithfulness import (
    FaithfulnessJudgment,
)


class StructuredFaithfulnessProvider(Protocol):
    def judge_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> FaithfulnessJudgment:
        ...
        

class FaithfulnessService:
    def __init__(
        self,
        provider: StructuredFaithfulnessProvider,
    ) -> None:
        self.provider = provider
    
    def evaluate(
        self,
        answer: str,
        context: str,
    ) -> FaithfulnessJudgment:
        if not answer.strip():
            raise ValueError(
                "Answer cannot be empty."
            )
        
        if not context.strip():
            raise ValueError(
                "Context cannot be empty."
            )

        return self.provider.judge_faithfulness(
            answer=answer,
            context=context,
        )