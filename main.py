
from agent import agent


def main() -> None:
    user_question = input("Ask a question: ").strip()

    initial_state = {
        "input_question": [user_question],
    }

    result = agent.invoke(initial_state)
    print(result)


if __name__ == "__main__":
    main()
