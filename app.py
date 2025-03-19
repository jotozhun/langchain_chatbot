import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

model = ChatOpenAI(model="gtp-4o-mini")
#model = ChatGroq(model="llama-3.3-70b-versatile")


completion = model.invoke(
    [HumanMessage(content="Hey whats up")])

print(completion)