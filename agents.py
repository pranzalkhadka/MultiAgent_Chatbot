from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
from dotenv import load_dotenv
import os
from langchain_community.utilities import ArxivAPIWrapper
from langchain_community.tools import ArxivQueryRun

load_dotenv() 

groq_api_key = os.getenv('GROQ_API_KEY')

arxiv_wrapper = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=300)
arxiv_tool = ArxivQueryRun(api_wrapper=arxiv_wrapper)


# def analyze_question(state):
#     llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
#     prompt = PromptTemplate.from_template("""
#     You are an agent that needs to define if a question is a technical code one, a general one, or an arithmetic question.

#     Question: {input}

#     Analyse the question. Only answer with "code" if the question is about technical development.
#     Answer with "general" if the question is not technical.
#     Answer with "arithmetic" if the question involves basic arithmetic (e.g., "2 + 2").

#     Your answer (code/general/arithmetic):
#     """)
#     chain = prompt | llm
#     response = chain.invoke({"input": state["input"]})
#     decision = response.content.strip().lower()
#     return {"decision": decision, "input": state["input"]}


def analyze_question(state):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    prompt = PromptTemplate.from_template("""
    You are an agent that needs to define if a question is a technical code one, a general one, an arithmetic one, or a research paper request.

    Question: {input}

    Analyse the question. Only answer with:
    - "code" if the question is about technical development.
    - "general" if the question is not technical.
    - "arithmetic" if the question involves basic arithmetic (e.g., "2 + 2").
    - "arxiv" if the question is about a research paper or scientific topic.

    Your answer (code/general/arithmetic/arxiv):
    """)
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    decision = response.content.strip().lower()
    return {"decision": decision, "input": state["input"]}


def answer_code_question(state):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    prompt = PromptTemplate.from_template(
        "You are a software engineer. Answer this question with step-by-step details: {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    return {"output": response.content}


def answer_generic_question(state):
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")
    prompt = PromptTemplate.from_template(
        "Give a general and concise answer to the question: {input}"
    )
    chain = prompt | llm
    response = chain.invoke({"input": state["input"]})
    return {"output": response.content}


def calculate_arithmetic(state):
    try:
        expression = re.sub(r'^[^\d]*([\d\+\-\*/\(\)\s]+).*$', r'\1', state["input"])
        result = eval(expression)
        return {"output": str(result)}
    except Exception as e:
        return {"output": f"Error: {str(e)}"}
    

def search_arxiv(state):
    query = state["input"] 
    try:
        response = arxiv_tool.invoke(query)
        return {"output": response}
    except Exception as e:
        return {"output": f"Error fetching Arxiv results: {str(e)}"}