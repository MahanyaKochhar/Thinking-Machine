from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    classify_clarity_node,
    clarify_question_node,
    assess_complexity_node,
    decompose_question_node,
    human_review_node,
    restate_question_node,
    answer_sub_question_node,
    aggregate_answers_node,
    user_question_node,
)
from state import ThinkingMachineState


def build_graph(checkpointer=None):
    graph = StateGraph(ThinkingMachineState)

    graph.add_node("user_question", user_question_node)
    graph.add_node("restate_question", restate_question_node)
    graph.add_node("classify_clarity", classify_clarity_node)
    graph.add_node("clarify_question", clarify_question_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("decompose_question", decompose_question_node)
    graph.add_node("assess_complexity", assess_complexity_node)
    graph.add_node("answer_sub_question", answer_sub_question_node)
    graph.add_node("aggregate_answers", aggregate_answers_node)

    graph.add_edge(START, "user_question")
    graph.add_edge("user_question", "restate_question")
    graph.add_edge("aggregate_answers", END)

    return graph.compile(checkpointer=checkpointer)


def build_local_graph():
    return build_graph(checkpointer=MemorySaver())


agent = build_graph()
