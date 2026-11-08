import streamlit as st
from groq import RateLimitError
from langchain_core.messages import AIMessage, HumanMessage
from langchain_groq import ChatGroq
from streamlit_chat import message

from car_agent import build_car_agent

model = ChatGroq(model="openai/gpt-oss-120b", temperature=0.7)
agent = build_car_agent(model)

st.title("Car Advisor")
st.session_state.setdefault(
    "history", [AIMessage("Hi! I'll help you find your next car.")]
)

def render(i, m):
    message(m.content, is_user=isinstance(m, HumanMessage), key=str(i))


for i, m in enumerate(st.session_state.history):
    render(i, m)

if question := st.chat_input("Tell me what you're looking for"):
    chat = st.session_state.history
    chat.append(HumanMessage(question))
    render(len(chat) - 1, chat[-1])

    try:
        with st.spinner("Thinking..."):
            result = agent.invoke({"messages": chat})
    except RateLimitError as e:
        chat.pop()
        wait = e.response.headers.get("retry-after")
        st.warning(
            f"I'm getting too many requests right now. Try again in {wait or 'a few'} seconds."
        )
        st.stop()

    chat.append(AIMessage(result["messages"][-1].content))
    render(len(chat) - 1, chat[-1])
