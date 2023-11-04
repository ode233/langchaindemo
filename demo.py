import os

import gradio as gr
from dotenv import load_dotenv
from langchain.agents import load_tools, initialize_agent, AgentType

from chatglm3.chatglmagent import ChatGLMAgent

load_dotenv()

llm = ChatGLMAgent(
    endpoint_url=os.getenv("endpoint_url")
)
tools = load_tools(["arxiv", "searchapi"])

agent = initialize_agent(
    tools, llm, agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION, verbose=True, handle_parsing_errors=True
)


def response(message, history):
    return agent.run(message)


demo = gr.ChatInterface(response)

demo.launch()
