
import os
import tempfile
import streamlit as st

from rag_pipeline import (
    load_and_split_pdf,
    create_vector_store,
    ask_question
)

st.set_page_config(
    page_title="Smart Document Assistant",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Smart Document Assistant")

col1, col2 = st.columns(2)

with col1:
    if st.button("🧹 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🔄 Reset Documents"):
        st.session_state.vectorstore = None
        st.session_state.messages = []
        st.rerun()

st.write(
    "Upload one or more PDFs, process them, and ask questions "
    "about their content. Answers are based on your documents."
)

with st.expander("How to use this application"):
    st.markdown("""
    1. Upload one or more PDF documents.
    2. Click **Process Documents**.
    3. Ask a question in the chat box.
    4. Check the source filename and page number.
    5. Use **Clear Chat** to start a new conversation.
    """)

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "messages" not in st.session_state:
    st.session_state.messages = []

uploaded_files = st.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)
    
if uploaded_files:
    if st.button("Process Documents"):
        all_chunks = []
        temp_paths = []

        try:
            with st.spinner("Processing documents..."):
                for uploaded_file in uploaded_files:
                    with tempfile.NamedTemporaryFile(
                        delete=False,
                        suffix=".pdf"
                    ) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        temp_path = temp_file.name

                    temp_paths.append(temp_path)

                    chunks = load_and_split_pdf(temp_path)

                    # Keep the original uploaded filename
                    for chunk in chunks:
                        chunk.metadata["source"] = uploaded_file.name

                    all_chunks.extend(chunks)

                if not all_chunks:
                    st.error("No readable text found in the uploaded PDFs.")
                else:
                    st.session_state.vectorstore = (
                        create_vector_store(all_chunks)
                    )
                    st.session_state.messages = []

                    st.success(
                        f"Processed {len(uploaded_files)} PDF(s) "
                        f"and created {len(all_chunks)} chunks."
                    )

                    st.session_state.processed_files = [
                        file.name for file in uploaded_files
                    ]

        except Exception as e:
            st.error(f"Could not process documents: {e}")

        finally:
            for path in temp_paths:
                if os.path.exists(path):
                    os.remove(path)


                    # Show processed document information
                    if st.session_state.vectorstore is not None:
                        st.success("Documents are ready for questions.")

                        if "processed_files" in st.session_state:
                            st.caption(
                                "Loaded documents: "
                                + ", ".join(st.session_state.processed_files)
                        )



if st.session_state.vectorstore is not None:
    st.subheader("Ask questions")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message.get("sources"):
                st.caption(f"Sources: {message['sources']}")

    question = st.chat_input("Ask something about your document...")

    if question:
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Finding an answer..."):
                try:
                    answer, sources = ask_question(
                        st.session_state.vectorstore,
                        question
                    )

                    st.markdown(answer)
                    if sources:
                        st.markdown("**Sources:**")

                        unique_sources = {
                          (source["source"], source["page"])
                        for source in sources
                        }

                        for filename, page in sorted(unique_sources):
                            st.caption(f"📄 {filename} — Page {page}")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"Could not answer the question: {e}")
