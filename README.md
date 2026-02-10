# 🤖 Company Private Data Assistant (RAG)

A professional Retrieval-Augmented Generation (RAG) system designed to help employees interact with private company documents (PDFs, Markdown, and Text) using OpenAI's LLMs and a local vector database.

## 🚀 Key Features
- **Intelligent Query Translation**: Rewrites vague user questions into search-optimized queries.
- **Advanced Retrieval**: Uses `MultiQueryRetriever` to search the documentation from multiple conceptual angles, ensuring better accuracy.
- **Multi-Format Support**: Automatically ingests and indexes `.pdf`, `.md`, and `.txt` files.
- **Chat Interface**: A modern, responsive web UI built with Streamlit for a seamless user experience.
- **Local Vector Storage**: Uses ChromaDB to store document embeddings locally, maintaining privacy and speed.
- **Knowledge Management**: Simple one-click "Refresh Knowledge Base" button to update the AI when new documents are added.

## 🛠️ Tech Stack
- **Framework**: [LangChain](https://www.langchain.com/)
- **LLM**: OpenAI GPT-4o-mini
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Database**: [ChromaDB](https://www.trychroma.com/)
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Language**: Python 3.11+

## ⚙️ Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/11Minato11/Rag-Company.git
cd Rag-Company
```

### 2. Set Up Virtual Environment
```bash
python -m venv .venv
# Windows:
.\.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configuration
Create a `.env` file in the root directory and add your OpenAI API Key:
```env
OPENAI_API_KEY=sk-proj-your-api-key-here
```

### 5. Add Your Data
Place your company documents (PDFs, Markdown, Text) in the `data/sample_documents/` directory.

### 6. Run the Application
```bash
streamlit run app.py
```

## 🌐 Deployment (Streamlit Cloud)
1. Push this project to a **Private** GitHub repository.
2. Connect your GitHub to [Streamlit Cloud](https://share.streamlit.io/).
3. Add your `OPENAI_API_KEY` to the **Secrets** section in the App Settings.
4. Click Deploy!

## 📂 Project Structure
- `app.py`: The main Streamlit web application.
- `Rag/Rag_project.py`: Core logic for loading, splitting, and retrieving documents.
- `data/`: Folder containing the private documents for the AI to "read".
- `chroma_db/`: Local storage for vectorized document chunks.
- `.gitignore`: Ensures sensitive files like `.env` are never uploaded to GitHub.
