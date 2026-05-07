from langgraph.graph import START, StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from nodes import (
    classify_clarity_node,
    clarify_question_node,
    assess_complexity_node,
    decompose_question_node,
    human_review_node,
    restate_question_node,
    sub_question_answerer_node,
    user_question_node,
)
from state import ThinkingMachineState


def build_graph():
    graph = StateGraph(ThinkingMachineState)

    graph.add_node("user_question", user_question_node)
    graph.add_node("restate_question", restate_question_node)
    graph.add_node("classify_clarity", classify_clarity_node)
    graph.add_node("clarify_question", clarify_question_node)
    graph.add_node("human_review", human_review_node)
    graph.add_node("decompose_question", decompose_question_node)
    graph.add_node("assess_complexity", assess_complexity_node)
    graph.add_node("sub_question_answerer", sub_question_answerer_node)

    graph.add_edge(START, "user_question")
    graph.add_edge("user_question", "restate_question")
    graph.add_edge("restate_question", "classify_clarity")
    graph.add_edge("clarify_question", "human_review")
    graph.add_edge("human_review", "restate_question")
    graph.add_edge("decompose_question", "assess_complexity")
    graph.add_edge("assess_complexity", "sub_question_answerer")
    graph.add_edge("sub_question_answerer", END)

    # Compile with MemorySaver for checkpointing
    memory_saver = MemorySaver()
    return graph.compile(checkpointer=memory_saver)


agent = build_graph()
