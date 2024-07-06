from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import HumanMessage
import streamlit as st
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
import time

load_dotenv()

models_to_chat = [
    ChatGoogleGenerativeAI(model="gemini-1.5-flash-001"),
    ChatGoogleGenerativeAI(model="gemini-1.5-pro-001"),
    ChatGroq(model="llama3-70b-8192"),
    ChatGroq(model="llama3-8b-8192"),
    ChatMistralAI(model="mistral-large-latest"),
    ChatMistralAI(model="open-mixtral-8x22b")
]

models_to_see = ChatGoogleGenerativeAI(model="gemini-pro-vision")

def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.02)

def processing():
    if 'responses' not in st.session_state:
        st.session_state['responses'] = [("How can I assist you?", "How can I assist you?")]
    if 'requests' not in st.session_state:
        st.session_state['requests'] = []
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(k=50, return_messages=True)
    if 'models' not in st.session_state:
        st.session_state.models = []

    system_msg_template = SystemMessagePromptTemplate.from_template(template="""Answer the question to the best of your ability""")
    human_msg_template = HumanMessagePromptTemplate.from_template(template="{input}")
    prompt_template = ChatPromptTemplate.from_messages([system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template])

    st.title("LLMs Playground🤖")
    st.subheader("Compare various LLMs!")
    st.logo("https://img.freepik.com/premium-vector/stylish-black-lion-logo-white-background-vector_532963-5993.jpg")
    st.write("This app presents the top LLMs for you to compare✨")

    response_container = st.container()

    query = st.chat_input("Send a message", key="input")

    models_to_talk = [
        {"name": "ChatGoogleGenerativeAI (Gemini 1.5 Flash 001)", "id": 0},
        {"name": "ChatGoogleGenerativeAI (Gemini 1.5 Pro 001)", "id": 1},
        {"name": "ChatGroq (LLama3-70b-8192)", "id": 2},
        {"name": "ChatGroq (LLama3-8b-8192)", "id": 3},
        {"name": "ChatMistralAI (Mistral Large Latest)", "id": 4},
        {"name": "ChatMistralAI (Open Mixtral 8x22b)", "id": 5},
    ]

    col1, col2 = st.columns(2)

    with col1:
        selected_model_1 = st.selectbox("Select a chat model (Left)", models_to_talk, format_func=lambda model: model["name"], placeholder="Choose a model")
    with col2:
        selected_model_2 = st.selectbox("Select a chat model (Right)", models_to_talk, format_func=lambda model: model["name"], placeholder="Choose a model")

    a = selected_model_1["id"]
    b = selected_model_2["id"]

    llm_1 = models_to_chat[a]
    llm_2 = models_to_chat[b]

    conversation_1 = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm_1, verbose=True)
    conversation_2 = ConversationChain(memory=st.session_state.buffer_memory, prompt=prompt_template, llm=llm_2, verbose=True)

    context = "You are a friendly and helpful chatbot."

    if query:
        with st.spinner("typing..."):
            response_1 = conversation_1.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
            response_2 = conversation_2.predict(input=f"Context:\n{context}\n\nQuery:\n{query}")
        
        st.session_state.responses.append((response_1, response_2))
        st.session_state.requests.append(query)
        st.session_state.models.append((selected_model_1["name"], selected_model_2["name"]))

        with response_container:
            if st.session_state['responses']:
                for i in range(len(st.session_state['responses'])):
                    if i < len(st.session_state['requests']):
                        response_1, response_2 = st.session_state['responses'][i+1]
                        query = st.session_state['requests'][i]
                        model_1, model_2 = st.session_state.models[i]

                        with st.container():
                            with col1:
                                st.markdown(f":red[{model_1}]")
                                message(query, is_user=True, key=str(i) + '_user_1')
                                message(response_1, key=str(i) + '_model_1')
                            
                            with col2:
                                st.markdown(f":red[{model_2}]")
                                message(query, is_user=True, key=str(i) + '_user_2')
                                message(response_2, key=str(i) + '_model_2')
    


    