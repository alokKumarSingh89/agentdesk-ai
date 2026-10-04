
from packages.ai.providers.openai_provider import (
    OpenAIProvider,
)


def main() -> None:
    provider = OpenAIProvider()

    print("Welcome to AgentDesk AI")

    question = input("Ask something: ").strip()

    if not question:
        print("Question cannot be empty.")
        return

    try:
        response = provider.generate(question)

    except Exception:
        print(
            "AI request failed. Check your API "
            "configuration and provider access."
        )
        raise SystemExit(1)

    print("\n--- AI RESPONSE ---")
    print(response.content)

    print("\n--- USAGE METADATA ---")
    print(f"Model: {response.model}")

    usage = response.usage

    print(f"Input tokens: {usage.input_tokens}")
    print(f"Output tokens: {usage.output_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    if usage.estimated_cost_usd is not None:
        print(
            f"Estimated cost: "
            f"${usage.estimated_cost_usd:.8f}"
        )
    else:
        print(
            "Estimated cost: unavailable "
            "(configure model pricing)"
        )


if __name__ == "__main__":
    main()
