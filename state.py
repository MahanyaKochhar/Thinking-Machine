import operator
from typing import Literal, TypedDict
from typing_extensions import Annotated

from pydantic import BaseModel, Field


class ClarityClassification(BaseModel):
    understanding: Literal["clear", "vague"] = Field(
        description="Whether the restated user question is clear enough to answer."
    )
    reason: str = Field(
        description="A concise explanation for why the question is clear or vague."
    )


class ComplexityAssessment(BaseModel):
    rating: Literal["low", "medium", "high"] = Field(
        description="Overall complexity rating for the question and its sub-questions."
    )
    reason: str = Field(
        description="A concise explanation for the assigned complexity rating."
    )


class ThinkingMachineState(TypedDict, total=False):
    input_question: str
    restated_question: str
    classification: ClarityClassification
    clarifying_question: str
    sub_questions: list[str]
    overall_complexity: ComplexityAssessment
    sub_question: str
    sub_question_responses: Annotated[list[str], operator.add]
    final_answer: str
