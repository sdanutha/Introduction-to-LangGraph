
import chainlit as cl
from langchain_core.messages import HumanMessage
import app_graph as g  # Import the compiled LangGraph instance

@cl.on_message
async def run(message: cl.Message):
    
    # Create the initial conversation state with the user's message.
    state = {"messages": [HumanMessage(content=message.content)]}
    
    # Invoke the compiled workflow.
    result = g.graph.invoke(state)
    
    # Extract and send the final message content from the result.
    final_response = result["messages"][-1].content
    await cl.Message(content=final_response).send()
