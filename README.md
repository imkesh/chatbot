# Question Answering from PDF

## Overview
This project is a Streamlit application that allows users to upload PDF files and ask questions related to the content of the uploaded file. It extracts text from the PDF, identifies relevant sections, and provides answers using a pre-trained BERT model. Additionally, the app performs basic Named Entity Recognition (NER) on the content.


## Features
1. Upload a PDF file and extract its content for processing.
2. Ask any question related to the uploaded document.
3. Get precise answers using a `BERT`-based transformer model.
4. Retrieve the most relevant sections using TF-IDF.
5. Perform Named Entity Recognition (NER) to extract entities like dates, names, and locations.

## Prerequisites
- Python 3.8 or later
- Required libraries (install using `pip`):
  ```bash
  pip install streamlit pdfplumber transformers spacy scikit-learn numpy
  python -m spacy download en_core_web_sm
