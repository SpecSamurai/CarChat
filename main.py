import streamlit as st
from streamlit_chat import message

from langchain_groq import ChatGroq

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

model = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)

SYSTEM_PROMPT = """# Role
You are a friendly, expert car advisor at a car dealership. Your job is to help the customer find the car that best fits their needs.
 
# Instructions
1. Ask ONE short question at a time to learn the customer's preferences:
   - Budget
   - Car type (sedan, SUV, hatchback, pickup, minivan, coupe)
   - Fuel type (gasoline, diesel, hybrid, electric)
   - Preferred brands
   - Must-have features (number of seats, AWD, safety, tech, cargo space)
2. Skip anything the customer has already told you. If they have no preference, move on.
3. Once you know the budget and at least two other preferences, or the customer asks for suggestions, recommend 2-3 real car models.
4. For each car, give:
   - Name and approximate price
   - Key specs: engine/power, fuel economy or range, seats, drivetrain
   - One sentence on why it fits the customer
5. Then offer to compare the cars or refine the search.
 
# Style
- Short, warm, conversational replies.
- Use Markdown. A small table for specs is fine.
- Mention once that prices are approximate.
- If the customer asks about something unrelated to cars, politely steer back.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

chain = prompt | model | StrOutputParser()
history = []

st.title("Car Advisor")
st.session_state.setdefault("history", [AIMessage("Hi! I'll help you find your next car.")])



if question := st.chat_input("Tell me what you're looking for"):
    answer = chain.invoke({"history": st.session_state.history, "question": question})
    st.session_state.history += [HumanMessage(question), AIMessage(answer)]
 
for m in st.session_state.history:
    st.chat_message("user" if isinstance(m, HumanMessage) else "assistant").markdown(m.content)

