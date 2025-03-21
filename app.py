import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, trim_messages
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

st.title("LangChain Chatbot")
st.header("Conversational Model with Memory", divider="red")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

# Model or LLM
#model = ChatOpenAI(model="gtp-4o-mini")
model = ChatGroq(model="llama-3.3-70b-versatile")

# ChatPrompt Template
prompt_template = ChatPromptTemplate(
    [
        SystemMessage(content="You are a personal assistant that taks like a pirate."),
        MessagesPlaceholder(variable_name="messages")
    ]
)

# Messages Trimmer
trimmer = trim_messages(
    strategy="last",
    max_tokens=50,
    token_counter=len,
    include_system=True
)

# define a new graph
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    chat_template_messages = prompt_template.invoke(trimmed_messages)
    completion = model.invoke(chat_template_messages)
    return {"messages": completion}

# Create a new Node
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Runnable Graph
runnable_graph = workflow.compile(checkpointer=st.session_state.memory)

# Graph config
config = {"configurable": {"thread_id": 1}}


if prompt := st.chat_input():

    # UI Messages
    for message in st.session_state.chat_history:
        with st.chat_message('human' if isinstance(message, HumanMessage) else "ai"):
            st.write(message.content)

    # Appends user prompt to chat history        
    st.session_state.chat_history.append(HumanMessage(content=prompt))

    
    completion = runnable_graph.invoke(S
        {"messages": HumanMessage(content=prompt)}, config
    )

    st.session_state.chat_history.append(AIMessage(content=completion["messages"][-1].content))

    completion["messages"][-1].pretty_print()