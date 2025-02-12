
import chainlit as cl
from langchain_core.messages import HumanMessage
import app_graph as g

@cl.on_message
async def handle_message(user_message: cl.Message):
    try:
        
        # Initialize conversation state
        state = {"messages": [HumanMessage(content=user_message.content)]}
        
        # Invoke LangGraph workflow
        result = g.graph.invoke(state)
        
        # Get final response
        response_text = result["messages"][-1].content
        
        # Stream response
        msg = cl.Message(content="")
        for token in response_text:
            await msg.stream_token(token)
        await msg.send()
    
    except Exception as e:
        
        # Handle errors
        await cl.Message(content=f"Error: {str(e)}").send()
