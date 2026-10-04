from packages.ai.providers.openai_provider import (
    OpenAIProvider
)
import openai

def main()->None:
    provider = OpenAIProvider()
    print("Welcome to AgentDesk AI")
    question = input("Ask something: ").strip()
    
    if not question:
        print("Question cannot be empty.")
        return
    
    try:
        answer = provider.generate(question)
        print("\nAgentDesk Response:")
        print(answer)
    except openai.AuthenticationError as exc:
        print(f"Authentication failed: {exc}")

    except openai.RateLimitError as exc:
        print(f"Rate limit or quota exceeded: {exc}")

    except openai.BadRequestError as exc:
        print(f"Invalid API request: {exc}")

    except openai.APIConnectionError as exc:
        print(f"Network/API connection failed: {exc}")

    except openai.APIStatusError as exc:
        print(
            f"OpenAI API error: "
            f"HTTP {exc.status_code}"
        )
    except Exception:
        print(
            "AI request failed. Check your API "
            "configuration and provider access."
        )
        raise SystemExit(1)
    
if __name__ == "__main__":
    main()