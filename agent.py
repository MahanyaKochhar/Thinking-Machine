from langgraph.graph import START, StateGraph, END

from nodes import (
    classify_clarity_node,
    # placeholder_node,
    restate_question_node,
    user_question_node,
)
from state import ThinkingMachineState


def build_graph():
    graph = StateGraph(ThinkingMachineState)

    graph.add_node("user_question", user_question_node)
    graph.add_node("restate_question", restate_question_node)
    graph.add_node("classify_clarity", classify_clarity_node)

    # graph.add_node("clarification_question", placeholder_node)
    # graph.add_node("human_clarification", placeholder_node)
    # graph.add_node("sub_questions", placeholder_node)
    # graph.add_node("overall_complexity", placeholder_node)
    # graph.add_node("sub_question_answerer", placeholder_node)
    # graph.add_node("synthesise_answers", placeholder_node)

    graph.add_edge(START, "user_question")
    graph.add_edge("classify_clarity",END)
    return graph.compile()


agent = build_graph()
