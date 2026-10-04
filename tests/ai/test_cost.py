"""Tests for LLM cost calculation."""
from decimal import Decimal

import pytest

from packages.ai.cost import calculate_llm_cost

def test_calculate_llm_cost() -> None:
    cost = calculate_llm_cost(
        input_tokens=1000,
        output_tokens=500,
        input_price=Decimal("1.00"),
        output_price=Decimal("4.00"),
    )

    # 1000 / 1M * $1 = $0.001
    # 500 / 1M * $4 = $0.002

    assert cost == Decimal("0.003")
    
def test_missing_pricing_returns_none() -> None:
    cost = calculate_llm_cost(
        input_tokens=100,
        output_tokens=200,
        input_price=None,
        output_price=None,
    )

    assert cost is None
    
def test_negative_tokens_are_rejected() -> None:
    with pytest.raises(ValueError):
        calculate_llm_cost(
            input_tokens=-100,
            output_tokens=200,
            input_price=Decimal("1"),
            output_price=Decimal("4"),
        )