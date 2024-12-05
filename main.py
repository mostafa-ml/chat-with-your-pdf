from groq import Groq
import streamlit as st
import streamlit_feedback
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.faiss import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import tempfile  # To handle temporary files
import os
import shutil


# Initialize Groq client
try:
    client = Groq(api_key=st.secrets['api']['Groq_API_KEY'])
except KeyError:
    st.error("API key for Groq is missing in secrets.")
    st.stop()


def load_documents(path):
    """Load a PDF file and extract its content."""
    loader = PyPDFLoader(path)
    documents = loader.load()
    return documents


def generate_embeddings():
    """Initialize the embedding model."""
    # embedding_model =  SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    embedding_model =  HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding_model


def create_vector_db(documents, file_path):
    """Create a FAISS vector database from documents."""
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)
    vec_db = FAISS.from_documents(documents=docs, embedding=generate_embeddings())
    # Save vec_db
    vec_db.save_local(file_path)
    return vec_db


# Load vec_db
def load_vec_db(file_path):
    return FAISS.load_local(folder_path=file_path, embeddings=generate_embeddings(), allow_dangerous_deserialization=True)


def retrieve_relevant_context(query, vec_db, k=4):
    if vec_db:    # vec_db != None
        # Approximate Nearest Neighbors (ANN) search on vector database
        docs = vec_db.similarity_search(query=query, k=k)
        return docs
    return None    # else:


def get_sys_msg(context):
    if context:
        sys_msg = f"Act as an AI expert who will answer questions about {context}."
        return sys_msg
    return None


# Formate the last context
def format_context(context):
    formatted_context = ""
    for i, doc in enumerate(context):
        formatted_context += f'**Context {i+1}:**\n{doc.page_content}\n\n'
    return formatted_context


# get the response from Llama LLM
def get_chat_response(prompt, chat_history, sys_msg_content, model_name, temperature, MAX_CHAT_HISTORY_LENGTH):
    chat_history.append({"role": "user", "content": prompt})

    # Keep the chat history within the specified max length
    if len(chat_history) > MAX_CHAT_HISTORY_LENGTH:
        chat_history = chat_history[-MAX_CHAT_HISTORY_LENGTH:]

    messages = chat_history

    if sys_msg_content:
        system_msg = [{"role": "system", "content": sys_msg_content}]
        messages = system_msg + messages

    try:
        stream = client.chat.completions.create(
            messages=messages,
            model=model_name,
            temperature=temperature,
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        st.error(f"❌Error while generating response:\n\n{e}")
        return


def main():
    st.title("Ask the AI 🤖")

    # sidebar controls
    model_name = st.sidebar.selectbox(label="Choose the model", options=["llama3-70b-8192", "gemma2-9b-it", "mixtral-8x7b-32768"], index=0)
    temperature = st.sidebar.slider(label="Set Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1)
    MAX_CHAT_HISTORY_LENGTH = int(st.sidebar.number_input(label="Max history length", min_value=1, max_value=10, value=4))

    # Create session state to store and load the chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    else:
        for msg in st.session_state.chat_history:
            st.chat_message(msg["role"]).write(msg['content'])

    uploaded_pdf = st.sidebar.file_uploader(label="Upload a PDF", type=["pdf"])
    vec_db_path = "vector_db"  # Directory or file path for saving the FAISS database

    if uploaded_pdf is not None:
        # If there is not vec_db => create and save it for future use
        if not os.path.exists(vec_db_path):
            print("Reading pdf...")
            with st.spinner("Reading your pdf..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                    temp_file.write(uploaded_pdf.read())  # Write the uploaded file's bytes to the temp file
            
            print("Creating vec_db...")
            with st.spinner("Creating vector database..."):
                documents = load_documents(temp_file.name)
                vec_db = create_vector_db(documents, vec_db_path)
            # Clean up the temporary file
            os.remove(temp_file.name)
    else:
        # To handel the case: when the user removes the uploaded file we need to remove its vector database!!!
        if os.path.exists(vec_db_path):
            shutil.rmtree(vec_db_path)

    def clear_chat():
        st.session_state.chat_history = []
    st.sidebar.button(label="Clear chat", on_click=clear_chat)

    user_input = st.chat_input("Enter your message...", key="user_input")
    if user_input:    # user_input != None
        if user_input.strip():
            st.chat_message("user").write(user_input)

            # If there is vec_db => load it
            if os.path.exists(vec_db_path):
                print("loading vec_db...")
                with st.spinner("Loading vector database..."):
                    vec_db = load_vec_db(vec_db_path)
            else:
                vec_db = None

            with st.spinner("AI is thinking..."):

                context = retrieve_relevant_context(query=user_input, vec_db=vec_db, k=4)
                st.session_state.last_context = context
                sys_msg = get_sys_msg(context)

                response_placeholder = st.empty()    # For dynamic updates
                ai_response = ""
                for chunk in get_chat_response(user_input, st.session_state.chat_history, sys_msg, model_name, temperature, MAX_CHAT_HISTORY_LENGTH):
                    ai_response += chunk
                    response_placeholder.chat_message("assistant").write(ai_response)

                st.session_state.chat_history.append({"role": "assistant", "content": ai_response})

                if st.session_state.last_context:
                    reference = format_context(st.session_state.last_context)
                    st.sidebar.text_area(label='Last query relevant context', value=reference, height=300, disabled=True)
                    st.session_state.last_context = None

                feedback = streamlit_feedback.streamlit_feedback(feedback_type='thumbs',
                                                                optional_text_label='[Optional] please provide an explanation',
                                                                key=f'feedback_{len(st.session_state.chat_history)}')
        else:
            # pass
            st.warning("⚠️Please enter a message!")


if __name__ == "__main__":
    main()
