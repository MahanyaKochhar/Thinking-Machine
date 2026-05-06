from navigator_llm import get_llm
from langgraph.graph import END
from langgraph.types import Command

from state import ClarityClassification, ThinkingMachineState


# Navigator LLM wrapper moved to `navigator_llm.py`.


def user_question_node(state: ThinkingMachineState) -> Command:
    """Store the user's input question in state."""
    questions = state.get("input_question", [])

    if isinstance(questions, str):
        questions = [questions]

    questions = [question.strip() for question in questions if question.strip()]

    if not questions:
        raise ValueError("user_question_node requires an input question.")

    return Command(
        update={"input_question": questions},
        goto="restate_question",
    )


def restate_question_node(state: ThinkingMachineState) -> Command:
    """Restate the user's question using an LLM."""
    user_question = state.get("input_question", [""])[-1]
    prompt = (
        f"Restate this user question in your own words to check your understanding. "
        f"Preserve the user's intent, do not answer it, and return only the "
        f" question in your understanding.\n\nUser question: {user_question}"
    )

    response = get_llm().invoke(prompt)
    restated_question = response.content

    return Command(
        update={"restated_question": restated_question},
        goto="classify_clarity",
    )


def classify_clarity_node(state: ThinkingMachineState) -> Command:
    """Use an LLM to classify the restated question as clear or vague."""
    question = state.get("restated_question", "").strip()
    prompt = (
        f"Classify whether this restated user question is clear enough to "
        f"proceed. Use 'clear' when there is enough context to answer or break "
        f"down the question. Use 'vague' when clarification is needed before "
        f"continuing. Include a concise reason.\n\nRestated question: {question}"
    )

    structured_llm = get_llm().with_structured_output(ClarityClassification)
    classification = structured_llm.invoke(prompt)

    return Command(
        update={"classification": classification},
        goto=END,
    )


def placeholder_node(state: ThinkingMachineState) -> ThinkingMachineState:
    """Temporary placeholder for nodes that will be implemented later."""
    return state
