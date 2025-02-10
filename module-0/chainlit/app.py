
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

import chainlit as cl

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

@cl.on_message
async def run_convo(message: cl.Message):
    
    # Create the initial conversation state with the user's message.
    state = {"messages": [HumanMessage(content=message.content)]}
    
    # Invoke the compiled workflow.
    result = graph.invoke(state)
    
    # Extract and send the final message content from the result.
    final_response = result["messages"][-1].content
    await cl.Message(content=final_response).send()
