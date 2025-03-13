from langgraph.graph import StateGraph, END
from agents import analyze_question, answer_code_question, answer_generic_question, calculate_arithmetic, search_arxiv
from typing import TypedDict


class AgentState(TypedDict):
    input: str
    decision: str
    continue_conversation: bool


def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("analyze", analyze_question)
    workflow.add_node("code_agent", answer_code_question)
    workflow.add_node("generic_agent", answer_generic_question)
    workflow.add_node("arithmetic_agent", calculate_arithmetic)
    workflow.add_node("arxiv_agent", search_arxiv)

    workflow.add_conditional_edges(
        "analyze",
        lambda x: x["decision"],
        {
            "code": "code_agent",
            "general": "generic_agent",
            "arithmetic": "arithmetic_agent",
            "arxiv": "arxiv_agent"
        }
    )

    workflow.set_entry_point("analyze")
    workflow.add_edge("code_agent", END)
    workflow.add_edge("generic_agent", END)
    workflow.add_edge("arithmetic_agent", END)
    workflow.add_edge("arxiv_agent", END)


    return workflow.compile()