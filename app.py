
import streamlit as st
import os
from Rag.Rag_project import (
    load_dotenv, 
    CHROMA_PATH, 
    DATA_PATH,
    OpenAIEmbeddings, 
    Chroma, 
    get_retriever, 
    create_rag_chain,
    construct_query,
    load_documents,
    split_text,
    create_vector_store
)

# Set Page Config
st.set_page_config(page_title="Company Brain RAG", page_icon="🤖", layout="wide")

# ---------------------------------
# Password Protection
# ---------------------------------
def check_password():
    """Returns True if the user entered the correct password."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if st.session_state.authenticated:
        return True

    # Show login form
    st.title("🔒 Company Private Data Assistant")
    st.markdown("Please enter the password to access the assistant.")
    
    password = st.text_input("Password", type="password")
    
    if st.button("Login"):
        # Check password from Streamlit Secrets
        correct_password = st.secrets.get("APP_PASSWORD", "admin123")
        if password == correct_password:
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("❌ Incorrect password. Please try again.")
    
    return False

# Block access if not authenticated
if not check_password():
    st.stop()

# App Header
st.title("🤖 Company Private Data Assistant")
st.markdown("---")

# Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for Setup & Management
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # 1. Management - Refresh Database
    if st.button("🔄 Refresh/Re-index Knowledge Base"):
        with st.status("Re-indexing..."):
            st.write("Deleting old database...")
            if os.path.exists(CHROMA_PATH):
                import shutil
                shutil.rmtree(CHROMA_PATH)
            
            st.write("Loading files...")
            docs = load_documents(DATA_PATH)
            st.write(f"Loaded {len(docs)} docs.")
            
            st.write("Splitting...")
            chunks = split_text(docs)
            
            st.write("Creating Vector Store...")
            create_vector_store(chunks)
            st.success("Indexing complete!")
            st.rerun()

    st.markdown("---")
    st.info("Ask questions about company policies, FAQs, or guidelines.")

# Initialize RAG Components
@st.cache_resource
def init_rag():
    # Load API key from Streamlit Cloud Secrets (if deployed)
    # This bridges the gap between local .env and cloud Secrets
    if hasattr(st, "secrets") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    load_dotenv()
    if not os.path.exists(CHROMA_PATH):
        # Initial setup if DB doesn't exist
        docs = load_documents(DATA_PATH)
        chunks = split_text(docs)
        vector_store = create_vector_store(chunks)
    else:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    
    retriever = get_retriever(vector_store)
    chain = create_rag_chain(retriever)
    return chain

rag_chain = init_rag()

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if prompt := st.chat_input("Ask me anything about the company documents..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("🔍 *Thinking and searching internal documents...*")
        
        try:
            # 1. Query Construction
            search_query = construct_query(prompt)
            
            # 2. Run RAG Chain
            response = rag_chain.invoke(search_query)
            
            message_placeholder.markdown(response)
            
            # Add assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": response})
        except Exception as e:
            st.error(f"Error: {e}")
            message_placeholder.markdown("❌ Sorry, something went wrong while processing your request.")
