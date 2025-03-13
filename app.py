import streamlit as st
from workflow import create_graph
from agents import answer_code_question, answer_generic_question, calculate_arithmetic, search_arxiv
from typing import TypedDict


class UserInput(TypedDict):
    input: str
    continue_conversation: bool


def process_question(state: UserInput):
    graph = create_graph()
    result = graph.invoke({"input": state["input"]})

    agent_name_map = {
        "code": "Code Agent",
        "general": "General Agent",
        "arithmetic": "Arithmetic Agent",
        "arxiv": "Research Agent"

    }
    agent_name = agent_name_map.get(result["decision"], "Unknown Agent")
    st.write("The agent answering your question is:", agent_name)

    if result["decision"] == "code":
        return answer_code_question(state)["output"]
    elif result["decision"] == "general":
        return answer_generic_question(state)["output"]
    elif result["decision"] == "arithmetic":
        return calculate_arithmetic(state)["output"]
    elif result["decision"] == "arxiv":
        return search_arxiv(state)["output"]
    else:
        return "Error: Unable to process the question."
    

def main():
    st.title("Multi-Agent Chatbot")

    user_input = st.text_input("Ask a question:")

    if user_input:
        state = {"input": user_input, "continue_conversation": True}
        answer = process_question(state)
        
        st.subheader("Answer:")
        st.write(answer)

    if not user_input:
        st.info("Please type a question in the text box above.")
        

if __name__ == "__main__":
    main()