
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)



def load_and_split_pdf(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError("The PDF file was not found.")

    if os.path.getsize(file_path) == 0:
        raise ValueError("The uploaded PDF is empty.")

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        raise ValueError("No pages were found in the PDF.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    if not any(chunk.page_content.strip() for chunk in chunks):
        raise ValueError(
            "No readable text found. The PDF may be scanned or image-only."
        )

    return chunks


def create_vector_store(chunks):
    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )


def ask_question(vectorstore, question):
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 3}
    )

    relevant_docs = retriever.invoke(question)

    context = "\n\n".join(
        doc.page_content for doc in relevant_docs
    )

    prompt = ChatPromptTemplate.from_template("""
Answer using only the provided context.
If the answer is missing, say:
"I don't know based on the provided document."

Context:
{context}

Question:
{question}

Answer:
""")

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({
        "context": context,
        "question": question
    })

    sources = [
        {
            "source": os.path.basename(
                doc.metadata.get("source", "Uploaded document")
            ),
            "page": doc.metadata.get("page", 0) + 1
        }
        for doc in relevant_docs
    ]

    return answer, sources
