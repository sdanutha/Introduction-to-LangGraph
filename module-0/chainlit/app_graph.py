
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

# Arithmetic functions
def add(a: int, b: int) -> int:
    """Returns the sum of two integers."""
    return a + b

def multiply(a: int, b: int) -> int:
    """Returns the product of two integers."""
    return a * b

def divide(a: int, b: int) -> float:
    """Returns the division of two integers, handling zero division errors."""
    if b == 0:
        raise ValueError("Division by zero is not allowed.")
    return a / b

# Define tools and LLM
tools = [add, multiply, divide]
llm = ChatOllama(model="llama3.2:1b").bind_tools(tools)

# System message
SYSTEM_MESSAGE = SystemMessage(
    content="You are a helpful assistant skilled in arithmetic operations."
)

# LLM processing node
def assistant(state: MessagesState) -> dict:
    response = llm.invoke([SYSTEM_MESSAGE] + state["messages"])
    return {"messages": [response]}

# Build LangGraph workflow
builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# Define edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

# Compile the graph
graph = builder.compile()
