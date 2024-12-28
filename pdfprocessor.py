import pdfplumber
from transformers import BertTokenizer, BertForQuestionAnswering, pipeline
import spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# Function to extract text from a PDF
def pdf_to_text(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  
                text += page_text
    return text

# Function to extract paragraphs and key sections (like headings or subheadings)
def improved_chunking(text):
    paragraphs = text.split("\n\n") 
    return paragraphs

# Function to use the model to answer questions
def answer_question(question, context):
    tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
    model = BertForQuestionAnswering.from_pretrained("bert-large-uncased")
    qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
    
    # Get the answer from the model
    result = qa_model(question=question, context=context)
    return result['answer']

# Basic Named Entity Recognition (NER)
def perform_ner(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    entities = [(entity.text, entity.label_) for entity in doc.ents]
    return entities

# Function to retrieve the most relevant context using TF-IDF
def retrieve_relevant_context(query, text_chunks):
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(text_chunks)
    query_vec = vectorizer.transform([query])
    cosine_similarities = np.array(tfidf_matrix * query_vec.T).flatten()
    most_similar_chunk_idx = cosine_similarities.argmax()
    return text_chunks[most_similar_chunk_idx]

# Function to process the PDF and answer questions
def process_pdf_and_answer_question(pdf_file, question):
    pdf_text = pdf_to_text(pdf_file)
    
    text_chunks = improved_chunking(pdf_text)
    
    relevant_context = retrieve_relevant_context(question, text_chunks)
    
    answer = answer_question(question, relevant_context)
    
    entities = perform_ner(relevant_context)
    
    return {
        "answer": answer,
        "extracted_text": pdf_text,
        "text_chunks": text_chunks,
        "entities": entities
    }
