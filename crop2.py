from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
import os
from model import predictCropTool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent

import chainlit as cl

os.environ["OPENAI_API_KEY"] = "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"

llm = ChatOpenAI(model="gpt-3.5-turbo-1106")
tools = [predictCropTool()]
prompt = hub.pull("hwchase17/openai-functions-agent")
prompt.messages[0].prompt.template="Your are an Highly intelligent Crop Predictor AI you can predict the crop based on the given values by the user, u can use the croppredictor tool for the prediction."

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_openai_functions_agent(llm, tools,prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools,memory=memory,verbose=True)

import chainlit as cl

@cl.on_message
async def on_message(message: cl.Message):
    await cl.Message(agent_executor.invoke({"input": message.content})["output"]).send()

    

