
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Tool
def add(a: int, b: int) -> int:
    """Adds a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.
    
    Args:
        a: first int
        b: second int
    """
    return a / b

tools = [add, multiply, divide]

# LLM with bound tool
llm = ChatOllama(model="llama3.2:1b")
llm_with_tools = llm.bind_tools(tools)

# Node
def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke([system_message] + state["messages"])]}

# System Message
system_message = SystemMessage(
    content="You are a helpful assistant tasked with performing arithmetic on a set of inputs."
)

# Graph
builder = StateGraph(MessagesState)

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()
