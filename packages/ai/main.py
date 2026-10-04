
"""AgentDesk AI main entry point."""
from packages.ai.providers.openai_provider import (
    OpenAIProvider,
)
from packages.ai.services.inquiry_service import (
    InquiryService,
)
from packages.ai.memory.store import ConversationStore
from packages.ai.services.chat_service import ChatService


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
    service: InquiryService,
) -> None:
    message = input("\nEnter customer message: ").strip()
    inquiry = service.classify(message)

    print("\n--- CLASSIFICATION RESULT ---")

    print(
        inquiry.model_dump_json(indent=2)
    )
    

def run_multi_turn_chat(
    provider: OpenAIProvider,
) -> None:
    store = ConversationStore()
    service = ChatService(
        provider=provider,
        store=store,
        max_history_messages=10,
    )
    conversation_id = service.create_conversation()

    print(f"\nConversation: {conversation_id}")
    print("Type 'exit' to finish.\n")
    while True:
        message = input("You: ").strip()
        if message.lower() == "exit":
            break

        if not message:
            continue
        
        try:
            response = service.send_message(
                conversation_id,
                message,
            )
        except Exception:
            print("AgentDesk: Request failed. Please retry.")
            continue
        
        print(f"\nAgentDesk: {response.content}\n")

        print(
            f"[Tokens: {response.usage.total_tokens}]\n"
        )

def main() -> None:
    # Composition root:
    # Create and connect application dependencies.
    provider = OpenAIProvider()

    inquiry_service = InquiryService(
        provider=provider
    )

    print("\nWelcome to AgentDesk AI")

    print("\n1. General AI Chat")
    print("2. Customer Inquiry Classification")
    print("3. Multi-turn AI Chat")

    choice = input("\nSelect option: ").strip()
    try:
        if choice == "1":
            run_chat(provider)
        elif choice == "2":
            run_inquiry_classifier(inquiry_service)
        elif choice == "3":
            run_multi_turn_chat(provider)
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
