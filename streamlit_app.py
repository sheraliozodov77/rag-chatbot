# streamlit_app.py

import streamlit as st
import requests
import uuid

API_URL = "http://localhost:8000/api/chat"

st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("💬 RAG Chatbot (Memory + LangGraph)")
st.markdown("Ask any question. The bot remembers your conversation and pulls facts from Pinecone-indexed data.")

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []

for role, message in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(message)

query = st.chat_input("Ask me anything...")

if query:
    st.session_state.messages.append(("user", query))
    with st.chat_message("user"):
        st.markdown(query)

    with st.spinner("Thinking..."):
        try:
            res = requests.post(API_URL, json={
                "user_input": query,
                "session_id": st.session_state.session_id
            })

            if res.status_code != 200:
                raise Exception(res.text)

            data = res.json()
            answer = data["response"]
            sources = data.get("sources", [])

            # Show bot message
            st.session_state.messages.append(("assistant", answer))
            with st.chat_message("assistant"):
                st.markdown(answer)

                # Show unique sources
                sources = data.get("sources", [])
                if sources:
                    # st.markdown("---")
                    st.markdown("**Manbalar:**")
                    for source in sources:
                        title = source.get("title", "Noma'lum manba")
                        url = source.get("url", "")
                        if url:
                            st.markdown(f"- [{title}]({url})")
                        else:
                            st.markdown(f"- {title}")

        except Exception as e:
            st.error(f"❌ Error: {e}")
