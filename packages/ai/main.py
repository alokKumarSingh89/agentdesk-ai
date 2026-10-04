
"""AgentDesk AI main entry point."""
from packages.ai.providers.openai_provider import (
    OpenAIProvider,
)


def run_chat(provider: OpenAIProvider) -> None:
    question = input("Ask AgentDesk: ").strip()
    if not question:
        print("Question cannot be empty.")
        return

    response = provider.generate(question)
    print("\n--- AI RESPONSE ---")
    print(response.content)

    print("\n--- USAGE ---")
    print(f"Model: {response.model}")
    print(
        f"Input tokens: "
        f"{response.usage.input_tokens}"
    )
    print(
        f"Output tokens: "
        f"{response.usage.output_tokens}"
    )
    print(
        f"Total tokens: "
        f"{response.usage.total_tokens}"
    )

    cost = response.usage.estimated_cost_usd

    if cost is not None:
        print(f"Estimated cost: ${cost:.8f}")
    else:
        print("Estimated cost: unavailable")
      

def run_inquiry_classifier(
    provider: OpenAIProvider,
) -> None:
    message = input("\nEnter customer message: ").strip()
    inquiry = provider.classify_inquiry(message)

    print("\n--- CLASSIFICATION RESULT ---")

    print(
        inquiry.model_dump_json(indent=2)
    )
    

def main() -> None:
    provider = OpenAIProvider()

    print("\nWelcome to AgentDesk AI")

    print("\n1. General AI Chat")
    print("2. Customer Inquiry Classification")

    choice = input("\nSelect option: ").strip()
    try:
        if choice == "1":
            run_chat(provider)
        elif choice == "2":
            run_inquiry_classifier(provider)
        else:
            print("Invalid option.")
    except ValueError as error:
        print(f"Validation error: {error}")
    except Exception:
        print(
            "AI request failed. Check your API "
            "configuration and provider access."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
