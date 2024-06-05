# s = "You are a helpful AI assistant."
# q = 'How to explain Internet for a medieval knight?'

import streamlit as st
from utils.llm import LLM
import time

llm = LLM()

def answerInProgress(q,llm):
    
    answer = llm.request(q)

    for word in answer.split(" "):
        yield word + " "
        time.sleep(0.05)

st.title('Open Chat')
# with st.sidebar:
#     st.header('Role')
#     role = st.text_area(label='', placeholder='Example: You are a teacher')

messages = st.container(height=500)
if question := st.chat_input("Ask me anything ..."):
    messages.chat_message("user").write(question)
    messages.chat_message("assistant").write_stream(answerInProgress(question, llm))