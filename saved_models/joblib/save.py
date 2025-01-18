import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved target encoding models for Gene and Variation
gene_target_encoder = joblib.load('target_encoder_gene.joblib')
variation_target_encoder = joblib.load('target_encoder_variation.joblib')

# Load the saved TF-IDF model for Text
text_tfidf_model = joblib.load('tfidf_vectorizer.joblib')

# Streamlit App
st.title('Preprocessing Input Data')
st.write('Enter values for Gene, Variation, and Text:')

# Input fields for Gene, Variation, and Text
gene_input = st.text_input('Gene:', value='SampleGene')
variation_input = st.text_input('Variation:', value='SampleVariation')
text_input = st.text_area('Text:', value='Sample clinical text here')

# Preprocess input data using the loaded models
if st.button('Preprocess Data'):
    # Target encoding for Gene and Variation
    encoded_gene = gene_target_encoder.transform([gene_input])[0]
    encoded_variation = variation_target_encoder.transform([variation_input])[0]
    
    # TF-IDF transformation for Text
    text_tfidf = text_tfidf_model.transform([text_input])
    
    st.write('Preprocessed Gene (Target Encoded):', encoded_gene)
    st.write('Preprocessed Variation (Target Encoded):', encoded_variation)
    st.write('Preprocessed Text (TF-IDF):', text_tfidf)
