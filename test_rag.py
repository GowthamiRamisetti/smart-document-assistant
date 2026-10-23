
from rag_pipeline import load_and_split_pdf, create_vector_store, ask_question

# Replace this with the path to a PDF on your computer
pdf_path = r"C:\Users\ramis\Desktop\GENAI local chatbox\smart-document Assistant\Machine_Learning_Syllabus_Phases (1).pdf"

chunks = load_and_split_pdf(pdf_path)
print("Number of chunks:", len(chunks))

vectorstore = create_vector_store(chunks)
print("Vector database created successfully!")

question = input("Ask a question about your PDF: ")

answer, sources = ask_question(vectorstore, question)

print("\nAnswer:", answer)
print("\nSources:", sources)
