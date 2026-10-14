
import pytest

from packages.ai.evaluation.faithfulness import (
    FaithfulnessJudgment,
)
from packages.ai.evaluation.faithfulness_service import (
    FaithfulnessService,
)


class FakeJudge:
    def judge_faithfulness(
        self,
        answer: str,
        context: str,
    ) -> FaithfulnessJudgment:
        return FaithfulnessJudgment(
            faithful=False,
            explanation="The timing is unsupported.",
            unsupported_claims=[
                "Refunds take 24 hours."
            ],
        )


def test_unsupported_claim_detected():
    service = FaithfulnessService(
        provider=FakeJudge()
    )

    result = service.evaluate(
        answer="Refunds take 24 hours.",
        context="Refunds take 5 business days.",
    )

    assert result.faithful is False
    assert len(result.unsupported_claims) == 1


def test_empty_context_rejected():
    service = FaithfulnessService(
        provider=FakeJudge()
    )

    with pytest.raises(ValueError):
        service.evaluate(
            answer="Refunds take 24 hours.",
            context="",
        )
