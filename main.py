from langgraph.types import Command
from agent import build_local_graph, build_graph


def main() -> None:
    agent = build_graph()
    user_question = input("Ask a question: ").strip()

    state = {"input_question": user_question}
    config = {"configurable": {"thread_id": "thread-1"}}

    result = agent.invoke(state, config=config, version="v2")

    while result.interrupts:
        interrupt_payload = result.interrupts[0].value
        question = interrupt_payload.get("clarifying_question", "Please clarify:")

        user_response = input(f"{question}\nYour response: ").strip()
        result = agent.invoke(
            Command(resume=user_response), config=config, version="v2"
        )

    print(result)


if __name__ == "__main__":
    main()
