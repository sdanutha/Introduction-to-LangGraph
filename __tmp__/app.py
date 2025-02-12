import chainlit as cl
import logging
import asyncio
from langchain_ollama import ChatOllama
from langchain.schema import HumanMessage, SystemMessage, AIMessage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model Configuration
MODEL_NAME = "llama3.2-vision:11b"

# Initialize the LangChain Ollama model
llm = ChatOllama(model=MODEL_NAME)


@cl.on_chat_start
async def start_chat():
    """Initialize user session and send a welcome message."""
    cl.user_session.set("interaction", [SystemMessage(content="You are a helpful assistant.")])

    msg = cl.Message(content="")
    welcome_text = "Hello! How can I assist you today?"

    # Stream the welcome message
    for token in welcome_text:
        await msg.stream_token(token)

    await msg.send()


async def call_ollama(messages):
    """Call LangChain's Ollama integration asynchronously."""
    try:
        response = await asyncio.to_thread(llm.invoke, messages)
        return response.content
    except Exception as e:
        logger.error(f"Error calling LangChain Ollama: {e}")
        return "I'm sorry, but I encountered an error processing your request."


@cl.step(type="tool")
async def tool(input_message: str, images: list = None):
    """Process user input and interact with LangChain Ollama API."""
    interaction = cl.user_session.get("interaction")

    # Construct HumanMessage with images if provided
    if images:
        user_input = HumanMessage(content=input_message, additional_kwargs={"images": images})
    else:
        user_input = HumanMessage(content=input_message)

    interaction.append(user_input)

    # Call Ollama API asynchronously (non-blocking)
    response_content = await call_ollama(interaction)

    # Update session history
    interaction.append(AIMessage(content=response_content))

    return response_content


@cl.on_message
async def main(message: cl.Message):
    """Handle incoming messages from users."""
    images = [file.path for file in message.elements if "image" in file.mime]

    # Process message with or without images
    response_content = await tool(message.content, images if images else None)

    # Send response as a streamed message
    msg = cl.Message(content="")
    for token in response_content:
        await msg.stream_token(token)

    await msg.send()
