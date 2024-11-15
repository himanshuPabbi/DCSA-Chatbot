import streamlit as st
import openai
from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Set up Streamlit page configuration
st.set_page_config(page_title="DCSA Chatbot", page_icon="🦙", layout="centered", initial_sidebar_state="auto")

# Set the OpenAI API key securely
openai.api_key = st.secrets.openai_key

st.title("Chat with the DCSA Chatbot, powered by LlamaIndex 💬🦙")

# Initialize session state for message history if not already initialized
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": '''I can assist with definitions, standards, and timelines for the exchange of information 
            between vessel sharing partners related to Loadlist and Bayplan, as per DCSA guidelines. 
            Ask me any question related to DCSA, and I will provide detailed, factual answers based on the available knowledge base'''
        }
    ]

# Cache data loading
@st.cache_data(show_spinner=True)
def load_data():
    try:
        # Loading documents from directory
        reader = SimpleDirectoryReader(input_dir="C:\\Users\\DELL\\OneDrive\\Desktop\\llamaindex projects2\\DCSA Chatbot\\Data\\", recursive=True)
        docs = reader.load_data()
        print(docs)

        # Set up the LLM with system prompt
        Settings.llm = OpenAI(
            model="gpt-4",
            temperature=0,
            system_prompt='''You are an expert on DCSA's definitions, standards, and timelines for the exchange of information 
            between vessel sharing partners regarding Loadlist and Bayplan. Provide technical, fact-based responses, 
            and avoid generating unverified features or hallucinated details.'''
        )

        # Create index
        index = VectorStoreIndex.from_documents(docs)
        return index

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

# Load data and initialize index
index = load_data()

if index:
    # Initialize the chat engine if not already set
    if "chat_engine" not in st.session_state.keys():
        st.session_state.chat_engine = index.as_chat_engine(
            chat_mode="condense_question", verbose=True, streaming=True
        )

    # Define message history limits to prevent excessive memory usage
    MAX_MESSAGES = 10
    if len(st.session_state.messages) > MAX_MESSAGES:
        st.session_state.messages = st.session_state.messages[-MAX_MESSAGES:]

    # Prompt user input and save to chat history
    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})

    # Display message history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Loading indicator during response generation
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner('Generating response...'):
                response_stream = st.session_state.chat_engine.stream_chat(prompt)
                st.write_stream(response_stream.response_gen)
                message = {"role": "assistant", "content": response_stream.response}
                st.session_state.messages.append(message)
else:
    st.error("Failed to load data and initialize chatbot.")
