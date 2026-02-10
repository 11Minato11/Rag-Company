
# ---------------------------------
# 1. Environment & Utilities
# ---------------------------------
import os
import sys
from dotenv import load_dotenv

# ---------------------------------
# 2. Document Loading
# ---------------------------------
from langchain_community.document_loaders import (
    DirectoryLoader,    # Load all files in a folder
    PyPDFLoader,        # Load PDF files
    TextLoader,         # Load plain text files
    Docx2txtLoader,     # Load Word documents
    CSVLoader,          # Load CSV files
    UnstructuredExcelLoader # Load Excel files
)

# ---------------------------------
# 3. Text Splitting
# ---------------------------------
from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,  # Recommended splitter for generic text
    CharacterTextSplitter            # Simpler splitter based on characters
)

# ---------------------------------
# 4. Embeddings & Vector Store
# ---------------------------------
from langchain_openai import OpenAIEmbeddings  # OpenAI's embedding model
from langchain_chroma import Chroma            # Local vector database

# ---------------------------------
# 5. Retrieval
# ---------------------------------
try:
    from langchain.retrievers.multi_query import MultiQueryRetriever
except ImportError:
    try:
        from langchain_community.retrievers import MultiQueryRetriever
    except ImportError:
        # Fallback for very specific environments
        from langchain.retrievers import MultiQueryRetriever


# ---------------------------------
# 6. Generation (LLM)
# ---------------------------------
from langchain_openai import ChatOpenAI        # Chat model (GPT-3.5/4)

# ---------------------------------
# 7. Prompting & Output Parsing
# ---------------------------------
from langchain_core.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate
)
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# ---------------------------------
# 8. Functional Utilities
# ---------------------------------
from operator import itemgetter

# Load environment variables
load_dotenv()

# ---------------------------------
# Configuration
# ---------------------------------
DATA_PATH = r"c:\Users\othma\Documents\Simple_RAG\data\sample_documents"
CHROMA_PATH = r"chroma_db"

# ---------------------------------
# 1. Load Documents
# ---------------------------------
def load_documents(data_path):
    """
    Loads documents from the specified directory.
    Supports PDF, DOCX, and TXT via DirectoryLoader.
    """
    print(f"Loading documents from {data_path}...")
    documents = []
    
    # 1. Load PDFs
    # We use glob="**/*.pdf" to find all PDFs recursively
    pdf_loader = DirectoryLoader(
        data_path, glob="**/*.pdf", loader_cls=PyPDFLoader, show_progress=True
    )
    documents.extend(pdf_loader.load())

    # 2. Load Word Docs (Uncomment if needed)
    # docx_loader = DirectoryLoader(
    #     data_path, glob="**/*.docx", loader_cls=Docx2txtLoader, show_progress=True
    # )
    # documents.extend(docx_loader.load())

    # 3. Load Text Files
    txt_loader = DirectoryLoader(
         data_path, 
         glob="**/*.txt", 
         loader_cls=TextLoader, 
         loader_kwargs={"encoding": "utf-8"},
         show_progress=True
     )
    documents.extend(txt_loader.load())

    # 4. Load Markdown Files
    markdown_loader = DirectoryLoader(
        data_path, 
        glob="**/*.md", 
        loader_cls=TextLoader, 
        loader_kwargs={"encoding": "utf-8"},
        show_progress=True
    )
    documents.extend(markdown_loader.load())
    
    print(f"Loaded {len(documents)} documents.")
    return documents

# ---------------------------------
# 2. Split Text
# ---------------------------------
def split_text(documents):
    """
    Splits documents into smaller chunks for embedding.
    """
    print("Splitting documents...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,       # Size of each chunk in characters
        chunk_overlap=200,     # Overlap to keep context between chunks
        length_function=len,
        is_separator_regex=False,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks

# ---------------------------------
# 3. Embed & Vector Store (Indexing)
# ---------------------------------
def create_vector_store(chunks):
    """
    Embeds documents and stores them in a local ChromaDB.
    Returns the vector store object.
    """
    # 1. Define the embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 2. Create (or load) the Vector Store
    # We use persist_directory to save it to disk so we don't re-embed every time
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=CHROMA_PATH
    )
    print("Vector Store created and saved.")
    return vector_store

# ---------------------------------
# 4. Query Construction (Rewrite)
# ---------------------------------
def construct_query(question):
    """
    Rewrites the user's question into a standalone, search-optimized query.
    Useful for chat history or vague questions.
    """
    llm = ChatOpenAI(temperature=0)
    
    # Prompt to rewrite the question
    rewrite_prompt = ChatPromptTemplate.from_template(
        """Provide a better search query for the user question. 
        Focus on extracting keywords and intent. 
        Do not answer the question, just rewrite it for a search engine.

        User Question: {question}
        
        Search Query:"""
    )
    
    rewriter = rewrite_prompt | llm | StrOutputParser()
    
    # Run the rewrite
    search_query = rewriter.invoke({"question": question})
    return search_query

# ---------------------------------
# 5. Retrieval with Query Translation
# ---------------------------------
def get_retriever(vector_store):
    """
    Creates a retriever with Query Translation (Multi-Query).
    This generates multiple variations of the user's question to find better results.
    """
    
    # 1. The Base Retriever (Basic Semantic Search)
    base_retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5} # Retrieve top 5 chunks
    )

    # 2. Query Translation: Multi-Query Retriever
    # This uses an LLM to generate 3-5 different versions of the user's question
    llm = ChatOpenAI(temperature=0)
    
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever, 
        llm=llm
    )
    
    return multi_query_retriever

# ---------------------------------
# 5. Generation (RAG Chain)
# ---------------------------------
def create_rag_chain(retriever):
    """
    Builds the final RAG chain: defined query -> retrieve context -> generate answer.
    """
    # 1. Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # 2. Define the Prompt Template
    template = """You are a helpful assistant for a company. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    prompt = ChatPromptTemplate.from_template(template)

    # 3. Build the Chain using LCEL (LangChain Expression Language)
    # RunnablePassthrough just passes the input query to the next step
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return rag_chain

# ---------------------------------
# 6. Main Execution
# ---------------------------------
if __name__ == "__main__":
    # Check if vector DB already exists
    if os.path.exists(CHROMA_PATH):
        print("Loading existing vector store...")
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
    else:
        print("Creating new vector store...")
        documents = load_documents(DATA_PATH)
        chunks = split_text(documents)
        vector_store = create_vector_store(chunks)

    # Create the retrieval chain
    retriever = get_retriever(vector_store)
    rag_chain = create_rag_chain(retriever)
    
    # Example User Input
    user_input = "What is the main topic of the documents?" 

    # 1. Query Construction
    print(f"\nUser Input: {user_input}")
    print("Constructing Search Query...")
    search_query = construct_query(user_input)
    print(f"Refined Query: {search_query}")
    
    # 2. Run the chain with the NEW query
    print("Thinking...")
    result = rag_chain.invoke(search_query)
    
    print("\nANSWER:")
    print(result)
