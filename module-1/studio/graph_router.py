
from langchain_ollama import ChatOllama
from langgraph.graph import MessagesState
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# Tool
def multiply(a: int, b: int) -> int:
    """Multiplies a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a * b

# LLM with bound tool
llm = ChatOllama(model="llama3.2:1b")
llm_with_tools = llm.bind_tools([multiply])

# Node
def tool_calling_llm(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# Graph
builder = StateGraph(MessagesState)

builder.add_node("tool_calling_llm", tool_calling_llm)
builder.add_node("tools", ToolNode([multiply]))

builder.add_edge(START, "tool_calling_llm")
builder.add_conditional_edges("tool_calling_llm", tools_condition)
builder.add_edge("tools", END)

graph = builder.compile()
