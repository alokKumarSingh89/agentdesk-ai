"""LLM cost calculation utilities."""
from decimal import Decimal


ONE_MILLION = Decimal("1000000")

def calculate_llm_cost(
    input_tokens: int,
    output_tokens: int,
    input_price: Decimal | None,
    output_price: Decimal | None,
) -> Decimal | None:
    """
    Estimate LLM cost in USD.

    Prices are expressed per million tokens.

    Returns None when pricing is unavailable.
    """
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError(
            "Token counts cannot be negative."
        )
    if input_price is None or output_price is None:
        return None
    if input_price < 0 or output_price < 0:
        raise ValueError(
            "Token prices cannot be negative."
        )
    
    input_cost = (
        Decimal(input_tokens)
        * input_price
        / ONE_MILLION
    )
    
    output_cost = (
        Decimal(output_tokens)
        * output_price
        / ONE_MILLION
    )
    return input_cost + output_cost