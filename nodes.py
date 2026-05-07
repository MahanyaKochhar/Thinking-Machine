from navigator_llm import get_llm
from langgraph.types import Command, interrupt
from langgraph.graph import END
from concurrent.futures import ThreadPoolExecutor

from state import ClarityClassification, ComplexityAssessment, ThinkingMachineState



# Navigator LLM wrapper moved to `navigator_llm.py`.


def user_question_node(state: ThinkingMachineState) -> Command:
    """Store the user's input question in state."""
    question = state.get("input_question", "").strip()

    if not question:
        raise ValueError("user_question_node requires an input question.")

    return Command(
        update={"input_question": question},
        goto="restate_question",
    )


def restate_question_node(state: ThinkingMachineState) -> Command:
    """Restate the user's question using an LLM."""
    user_question = state.get("input_question", "").strip()
    
    if not user_question:
        raise ValueError("input_question is required for restate_question_node.")
    
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
    """Use an LLM to classify the restated question as clear or vague.
    
    Routes to 'clarify_question' if vague, or END if clear.
    """
    question = state.get("restated_question", "").strip()
    prompt = (
        f"Classify whether this restated user question is clear enough to "
        f"proceed. Use 'clear' when there is enough context to answer or break "
        f"down the question. Use 'vague' when clarification is needed before "
        f"continuing. Include a concise reason.\n\nRestated question: {question}"
    )

    structured_llm = get_llm().with_structured_output(ClarityClassification)
    classification = structured_llm.invoke(prompt)

    # Route based on clarity classification
    goto = "clarify_question" if classification.understanding == "vague" else "decompose_question"

    return Command(
        update={"classification": classification},
        goto=goto,
    )


def clarify_question_node(state: ThinkingMachineState) -> Command:
    """Generate a clarifying question based on the vague restated question."""
    restated_question = state.get("restated_question", "").strip()
    reason = state.get("classification", {}).reason if state.get("classification") else ""
    
    prompt = (
        f"The user's question is considered vague for the following reason: {reason}\n\n"
        f"Generate a concise, specific clarifying question to ask the user to improve clarity. "
        f"Return only the clarifying question, nothing else.\n\n"
        f"Original question: {restated_question}"
    )
    
    response = get_llm().invoke(prompt)
    clarifying_question = response.content

    return Command(update={"clarifying_question": clarifying_question}, goto="human_review")


def human_review_node(state: ThinkingMachineState) -> Command:
    """Interrupt execution to collect user clarification.

    The resume payload from Command(resume=...) becomes the return value of interrupt().
    After receiving user response, routes back to restate_question with updated input.
    """
    clarifying_question = state.get("clarifying_question", "").strip()
    if not clarifying_question:
        raise ValueError("clarifying_question is required for human_review_node.")

    user_response = interrupt({"clarifying_question": clarifying_question})
    if not isinstance(user_response, str) or not user_response.strip():
        raise ValueError("User clarification is required.")

    original_question = state.get("input_question", "").strip()
    # Append clarification to original question
    updated_question = f"{original_question}\n\nClarification: {user_response.strip()}"
    
    return Command(
        update={"input_question": updated_question},
        goto="restate_question",
    )


def decompose_question_node(state: ThinkingMachineState) -> ThinkingMachineState:
    """Decompose a clear question into sub-questions for detailed analysis."""
    restated_question = state.get("restated_question", "").strip()
    
    if not restated_question:
        raise ValueError("restated_question is required for decomposition.")
    
    prompt = (
        f"Break down this question into specific sub-questions."
        f"Each sub-question must be directly derived from and stay within "
        f"the context of this question only. Do not introduce new topics or external context. "
        f"Return only the sub-questions, one per line.\n\n"
        f"Question: {restated_question}"
    )
    
    response = get_llm().invoke(prompt)
    sub_questions_text = response.content
    
    # Parse sub-questions from the response
    sub_questions = [q.strip() for q in sub_questions_text.split("\n") if q.strip()]

    return Command(
        update={"sub_questions": sub_questions},
        goto="assess_complexity",
    )


def assess_complexity_node(state: ThinkingMachineState) -> Command:
    """Assess the overall complexity of the question and generated sub-questions."""
    restated_question = state.get("restated_question", "").strip()
    sub_questions = state.get("sub_questions", [])

    if not restated_question:
        raise ValueError("restated_question is required for complexity assessment.")

    if not sub_questions:
        raise ValueError("sub_questions are required for complexity assessment.")

    prompt = (
        f"Assess the overall complexity of this question and its sub-questions. "
        f"Rate it as low, medium, or high based on how many distinct ideas, steps, dependencies, "
        f"and domain concepts are involved. Use only the question and the listed sub-questions; "
        f"do not invent anything new.\n\n"
        f"Restated question: {restated_question}\n\n"
        f"Sub-questions:\n" + "\n".join(f"- {sub_question}" for sub_question in sub_questions)
    )

    structured_llm = get_llm().with_structured_output(ComplexityAssessment)
    assessment = structured_llm.invoke(prompt)

    return Command(
        update={"overall_complexity": assessment},
        goto="sub_question_answerer",
    )


def _answer_sub_question(sub_question: str, restated_question: str) -> str:
    """Answer one sub-question using the current restated question context."""
    prompt = (
        f"Answer this sub-question using only the context of the restated question. "
        f"Do not introduce new assumptions beyond what is implied by the question. "
        f"Return a concise answer only.\n\n"
        f"Restated question: {restated_question}\n\n"
        f"Sub-question: {sub_question}"
    )

    response = get_llm().invoke(prompt)
    return response.content.strip()


def sub_question_answerer_node(state: ThinkingMachineState) -> Command:
    """Answer each generated sub-question in parallel and store the responses."""
    restated_question = state.get("restated_question", "").strip()
    sub_questions = state.get("sub_questions", [])

    if not restated_question:
        raise ValueError("restated_question is required for sub-question answering.")

    if not sub_questions:
        raise ValueError("sub_questions are required for sub-question answering.")

    with ThreadPoolExecutor(max_workers=min(8, len(sub_questions))) as executor:
        sub_question_responses = list(
            executor.map(
                lambda sub_question: _answer_sub_question(sub_question, restated_question),
                sub_questions,
            )
        )

    return Command(
        update={"sub_question_responses": sub_question_responses},
        goto=END,
    )
