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
    words = set(cleaned.split())

    if not cleaned:
        return "Please enter a message."

    # Return and refund
    if words.intersection({"return", "refund", "exchange", "replace"}):
        return responses["return_refund"]

    # Business hours
    if words.intersection({"hours", "open", "closing", "timing", "schedule"}):
        return responses["business_hours"]

    # Greeting
    if words.intersection({"hello", "hi", "hey", "goodmorning", "goodafternoon"}):
        return responses["greeting"]

    # Goodbye
    if words.intersection({"bye", "goodbye", "farewell"}):
        return responses["goodbye"]

    # Thanks
    if words.intersection({"thanks", "thank", "thankyou", "appreciate"}):
        return responses["thanks"]

    # Human support
    if words.intersection({"agent", "representative", "human", "person"}):
        return responses["human_support"]

    # Order status
    if words.intersection({"tracking", "track", "status", "order"}):
        return responses["order_status"]

    # Shipping and delivery
    if words.intersection({"shipping", "delivery", "delivered", "arrive", "shipment"}):
        return responses["shipping"]

    # Payment
    if words.intersection({"payment", "pay", "paid", "card", "transaction"}):
        return responses["payment"]

    # Products
    if words.intersection({"product", "products", "item", "items", "catalog"}):
        return responses["products"]

    # Machine learning prediction
    probabilities = model.predict_proba([cleaned])[0]
    best_index = np.argmax(probabilities)

    intent = model.classes_[best_index]
    confidence = probabilities[best_index]

    if confidence < 0.30:
        return (
            "Sorry, I could not understand your question. "
            "Please ask about orders, delivery, payment, "
            "products, returns or business hours."
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
