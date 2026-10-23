# Smart Document Assistant — RAG Application

A Retrieval-Augmented Generation (RAG) application that lets users upload one or more PDF documents and ask questions about their content.

## Features
- Upload multiple PDF documents
- Extract and split document text into chunks
- Generate embeddings using Hugging Face
- Retrieve relevant chunks using Chroma vector storage
- Generate context-based answers using Groq LLM
- Display source filenames and page numbers
- Chat history, clear chat, and reset documents
- Built with Streamlit

## Tech Stack
- Python
- Streamlit
- LangChain
- Groq API
- Hugging Face Sentence Transformers
- Chroma Vector Database
- PyPDF

## How It Works
1. Upload PDF documents.
2. Extract text and split it into chunks.
3. Convert chunks into embeddings.
4. Retrieve relevant chunks for the user's question.
5. Pass the retrieved context to the language model.
6. Display the answer and source references.

## Installation

1. Clone this repository.
2. Create and activate a virtual environment.
3. Install dependencies:

   `pip install -r requirements.txt`

4. Create a `.env` file in the project root:

   `GROQ_API_KEY=your_groq_api_key`

5. Run the application:

   `streamlit run app.py`

## Security
Keep API keys in your local `.env` file. Never commit secrets to GitHub.

## Author
Gowthami Ramisetti
