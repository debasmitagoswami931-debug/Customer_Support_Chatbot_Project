print("CHATBOT APP STARTED")
import streamlit as st
import joblib
import re
import numpy as np

st.set_page_config(
    page_title="Customer Support Chatbot",
    page_icon="🤖",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = joblib.load("chatbot_model.pkl")
    responses = joblib.load("chatbot_responses.pkl")
    return model, responses

model, responses = load_files()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chatbot_reply(user_input):
    cleaned = clean_text(user_input)

    if not cleaned:
        return "Please enter a message."

    probabilities = model.predict_proba([cleaned])[0]
    best_index = np.argmax(probabilities)

    intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < 0.30:
        return (
            "Sorry, I could not understand your question. "
            "Please ask about orders, delivery, payment, "
            "products or returns."
        )

    return responses[intent]

st.title("🤖 Customer Support Chatbot")
st.write(
    "Welcome! Ask me about orders, shipping, "
    "payments, products and returns."
)

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! Welcome to Customer Support. How can I help you?"
        }
    ]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("Type your question here..."):

    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })

    with st.chat_message("user"):
        st.write(prompt)

    reply = chatbot_reply(prompt)

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    with st.chat_message("assistant"):
        st.write(reply)

if st.button("Clear Chat"):
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! How can I help you?"
        }
    ]
    st.rerun()

st.caption(
    "Customer Support Chatbot | Built using Python, "
    "Scikit-learn and Streamlit"
)
