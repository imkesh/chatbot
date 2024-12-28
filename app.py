import streamlit as st
from pdfprocessor import process_pdf_and_answer_question  # Import the function from pdfprocessor.py

# Streamlit UI
st.title("Question Answering from PDF")

# Text box for asking a question
question = st.text_input("Ask your question:")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

# Check if a PDF is uploaded
if uploaded_file:
    # Process the PDF and answer the question
    if question:
        result = process_pdf_and_answer_question(uploaded_file, question)

        # Display the answer
        st.write(f"**Answer:** {result['answer']}")

        # Display extracted text and chunks
        st.write("Extracted Text (First 1000 characters):")
        st.write(result["extracted_text"][:1000])  # First 1000 characters

        st.write("Text Chunks (First few chunks):")
        st.write(result["text_chunks"][:5])  # First few chunks

        # Display detected entities
        st.write("Entities Detected:")
        st.write(result["entities"])

else:
    st.write("Please upload a PDF to get the answer.")
