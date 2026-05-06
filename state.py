from typing import Literal, TypedDict

from pydantic import BaseModel, Field


class ClarityClassification(BaseModel):
    understanding: Literal["clear", "vague"] = Field(
        description="Whether the restated user question is clear enough to answer."
    )
    reason: str = Field(
        description="A concise explanation for why the question is clear or vague."
    )


class ThinkingMachineState(TypedDict, total=False):
    input_question: list[str]
    restated_question: str
    classification: ClarityClassification
    clarifying_question: str
    sub_questions: list[str]
    sub_question_responses: list[str]
